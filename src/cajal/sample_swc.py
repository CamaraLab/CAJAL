"""Functions for sampling points from an SWC reconstruction of a neuron."""

import math
from typing import Callable, Union, Any, Optional
import ot
import numpy as np
import csv
from scipy.spatial.distance import squareform
from pathos.pools import ProcessPool
from multiprocessing import Pool
import numpy.typing as npt
import itertools as it
from scipy.spatial.distance import euclidean, pdist
from tqdm import tqdm

from .swc import (
    NeuronNode,
    NeuronTree,
    SWCForest,
    default_name_validate,
    get_filenames,
    read_swc,
    weighted_depth,
)
from .utilities import Err, T, write_csv_block, cell_iterator_csv, uniform
from .weighted_tree import (
    WeightedTree,
    WeightedTree_of,
    WeightedTreeChild,
    WeightedTreeRoot,
    weighted_depth_wt,
    weighted_dist_from_root,
)
from .types import Distribution, DistanceMatrix  # Matrix, Array

from threadpoolctl import ThreadpoolController

# Warning: Of 509 neurons downloaded from the Allen Brain Initiative
# database, about 5 had a height of at least 1000 nodes. Therefore on
# about 1% of test cases, recursive graph traversal algorithms will
# fail. For this reason we have tried to write our functions in an
# iterative style when possible.


def _count_nodes_helper(
    node_a: NeuronNode, node_b: NeuronNode, stepsize: float, offset: float
) -> tuple[int, float]:
    """
    Count nodes in the tree between `node_a` and `node_b` at the given stepsize/offset.

    :param node_a: A node in the graph being sampled; the parent of node_b.
    :param node_b: A node in the graph being sampled; the child of node_a.
    :param stepsize: The sampling parameter controlling the distance between points \
    sampled from along the line segments of the graph.
    :param offset: The height above `node_a` of the last point which was sampled from the graph. \
    It is assumed that offset < stepsize; otherwise, it would be possible to choose a lower point \
    above node_a which is at least stepsize away from the one above it.

    :return: `num_intermediary nodes`, the number of points on the line segment from `a` to `b`
    which would be returned if one sampled uniformly from the least point already sampled above `a`
    to `b` at a distance of `stepsize` between sampled points.
    :return: `leftover`, the height of the least sampled point on the line segment from `a` to `b`
    after sampling according to this procedure.
    """
    cumulative = (
        euclidean(np.array(node_a.coord_triple), np.array(node_b.coord_triple)) + offset
    )
    num_intermediary_nodes = math.floor(cumulative / stepsize)
    leftover = cumulative - (num_intermediary_nodes * stepsize)
    return num_intermediary_nodes, leftover


def _count_nodes_at_given_stepsize(tree: NeuronTree, stepsize: float) -> int:
    r"""Count how many nodes will be returned if the user samples points at `stepsize`.

    We sample uniformly from `tree`, starting at the root and adding
    all points at (geodesic) depth `stepsize`, 2 \* stepsize, 3 \*
    stepsize, and so on until we reach the end of the graph.

    :return: the number of points which would be sampled at this stepsize.

    """
    treelist = [(tree, 0.0)]
    acc: int = 1
    while bool(treelist):
        new_treelist: list[tuple[NeuronTree, float]] = []
        for tree0, offset in treelist:
            for child_tree in tree0.child_subgraphs:
                nodes, new_offset = _count_nodes_helper(
                    tree0.root, child_tree.root, stepsize, offset
                )
                acc += nodes
                new_treelist.append((child_tree, new_offset))
        treelist = new_treelist
    return acc


def _binary_stepwise_search(forest: SWCForest, num_samples: int) -> float:
    """
    Return the epsilon which will cause exactly `num_samples` points to be sampled.

    We assume the the forest is sampled at `stepsize` epsilon. The
    user should ensure that len(forest) <= num_samples.
    """
    if len(forest) > num_samples:
        raise Exception(
            "More trees in the forest than num_samples. \
        All root nodes of all connected components are returned as sample points, \
        so given this input it is impossible to get `num_samples` sample points. \
        Recommend discarding smaller trees. \
        "
        )

    max_depth = max((weighted_depth(tree) for tree in forest))
    max_reps = 50
    counter = 0
    step_size = max_depth
    adjustment = step_size / 2
    while counter < max_reps:
        num_nodes_this_step_size = sum(
            map(lambda tree: _count_nodes_at_given_stepsize(tree, step_size), forest)
        )
        if num_nodes_this_step_size < num_samples:
            step_size -= adjustment
        elif num_nodes_this_step_size > num_samples:
            step_size += adjustment
        else:
            return step_size
        adjustment /= 2
        counter += 1
    raise Exception("Binary search timed out.")


def get_sample_pts_euclidean(
    forest: SWCForest, step_size: float
) -> list[tuple[npt.NDArray[np.float64], int]]:
    """
    Sample points uniformly throughout the forest, starting at the roots, \
     at the given step size.

    :return: a list of (x,y,z) coordinate triples, \
    represented as numpy floating point \
    arrays of shape (3,). The list length depends (inversely) \
    on the value of `step_size`.
    """
    sample_pts_list: list[npt.NDArray[np.float64]] = []
    for tree in forest:
        sample_pts_list.append(
            (np.array(tree.root.coord_triple), tree.root.structure_id)
        )
    treelist = [(tree, 0.0) for tree in forest]
    while bool(treelist):
        new_treelist: list[tuple[NeuronTree, float]] = []
        for tree, offset in treelist:
            root_triple = np.array(tree.root.coord_triple)
            for child_tree in tree.child_subgraphs:
                child_triple = np.array(child_tree.root.coord_triple)
                dist = euclidean(root_triple, child_triple)
                assert step_size >= offset
                num_nodes, leftover = _count_nodes_helper(
                    tree.root, child_tree.root, step_size, offset
                )
                spacing = (
                    np.linspace(
                        start=step_size - offset,
                        stop=dist - leftover,
                        num=num_nodes,
                        endpoint=True,
                    )
                    / dist
                )
                assert spacing.shape[0] == num_nodes
                for x in spacing:
                    sample_pts_list.append(
                        (
                            (root_triple * x) + (child_triple * (1 - x)),
                            tree.root.structure_id,
                        )
                    )
                assert leftover >= 0
                assert leftover < step_size
                new_treelist.append((child_tree, leftover))
        treelist = new_treelist
    return sample_pts_list


def euclidean_point_cloud(
    forest: SWCForest, num_samples: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
    r"""
    Compute the (Euclidean) point cloud matrix for the forest with n sample points.

    :param forest: The cell to be sampled.
    :param num_samples: How many points to be sampled.
    :return: A rectangular matrix of shape (n,3), and an array of their structure ids.
    """
    if len(forest) >= num_samples:
        pts: list[npt.NDArray[np.float64]] = []
        structure_ids: list[npt.NDArray[np.int32]] = []
        for i in range(num_samples):
            pts.append(np.array(forest[i].root.coord_triple))
            structure_ids.append(forest[i].root.structure_id)
    else:
        step_size = _binary_stepwise_search(forest, num_samples)
        pts, structure_ids = zip(*get_sample_pts_euclidean(forest, step_size))
    return np.stack(pts), np.array(structure_ids, dtype=np.int32)


def icdm_euclidean(
    forest: SWCForest, num_samples: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
    r"""
    Compute the (Euclidean) intracell distance matrix for the forest with n sample points.

    :param forest: The cell to be sampled.
    :param num_samples: How many points to be sampled.
    :return: A condensed (vectorform) matrix of length n\* (n-1)/2.
    """
    x, y = euclidean_point_cloud(forest, num_samples)
    return pdist(x), y


def _sample_at_given_stepsize_wt(
    tree: WeightedTreeRoot, stepsize: float
) -> list[tuple[WeightedTree, float]]:
    r"""Sample points from the tree at the given step size.

    Starting from the root of `tree`, sample points along `tree` at a
    geodesic distance of `stepsize` from the root, 2 \* `stepsize`
    from the root, and so on until the end of the graph.

    In our formulation, a point `p` lying on a line segment from `a`
    to `b` (where `a` is the parent node and `b` is the child node)
    is represented by a pair `(p_dist, b)`, where `p_dist` is the
    distance from `p` to `b`, or the height of `p` above `b` in the
    graph.

    :return: A list of sample points `(h, b)`, where `b` is a node in
    `tree` and `h` is the distance of the sample point above `b`. `h`
    is guaranteed to be less than the distance between `a` and `b`. If
    `b` is the root node of its tree, `h` is guaranteed to be 0.
    """
    treelist: list[tuple[WeightedTree, float]] = [(tree, 0.0)]
    master_list: list[tuple[WeightedTree, float]] = [(tree, 0.0)]
    while bool(treelist):
        new_treelist: list[tuple[WeightedTree, float]] = []
        for tree0, offset in treelist:
            for child_tree in tree0.subtrees:
                cumulative = child_tree.dist + offset
                num_intermediary_nodes = math.floor(cumulative / stepsize)
                leftover = cumulative - (num_intermediary_nodes * stepsize)
                for k in range(num_intermediary_nodes):
                    assert (cumulative - stepsize * (k + 1)) <= child_tree.dist
                    master_list.append((child_tree, cumulative - stepsize * (k + 1)))
                new_treelist.append((child_tree, leftover))
        treelist = new_treelist
    return master_list


def _geodesic_distance_children(
    wt1: WeightedTreeChild, h1: float, wt2: WeightedTreeChild, h2: float
):
    """Compute the geodesic distance between p1 = (wt1,h1) and p2 = (wt2, h2)."""
    depth1 = wt1.depth
    unique_id1 = wt1.unique_id
    wt_parent1 = wt1.parent
    d1 = wt1.dist
    depth2 = wt2.depth
    unique_id2 = wt2.unique_id
    wt_parent2 = wt2.parent
    d2 = wt2.dist
    if unique_id1 == unique_id2:
        return abs(h2 - h1)
    # p1, p2 don't lie over the same child node.
    # Thus, either one is an ancestor of the other
    # (with one of wt1, wt2 strictly in between)
    # or they have a common ancestor, and dist(p1, p2)
    # is the sum of distances from p1, p2 to the common ancestor respectively.
    # These three can be combined into the following problem:
    # there is a minimal node in the weighted tree
    # which lies above both wt1 and wt2, and dist(p1, p2) is the sum
    # of the distances from p1, p2 respectively to that common minimal node.
    # This includes the case where the minimal node is wt1 or wt2.
    # To address these cases in a uniform way we use some cleverness with abs().
    dist1 = -h1
    dist2 = -h2
    while depth1 > depth2:
        dist1 += d1
        if isinstance(wt_parent1, WeightedTreeRoot):
            raise Exception("Nodes constructed have wrong depth.")
        elif isinstance(wt_parent1, WeightedTreeChild):
            depth1 = wt_parent1.depth
            unique_id1 = wt_parent1.unique_id
            d1 = wt_parent1.dist
            wt_parent1 = wt_parent1.parent
        else:
            raise Exception("Case missed.")
    while depth2 > depth1:
        dist2 += d2
        if isinstance(wt_parent2, WeightedTreeRoot):
            raise Exception("Nodes constructed have wrong depth.")
        elif isinstance(wt_parent2, WeightedTreeChild):
            depth2 = wt_parent2.depth
            unique_id2 = wt_parent2.unique_id
            d2 = wt_parent2.dist
            wt_parent2 = wt_parent2.parent
    # Now we know that both nodes have the same height.
    while unique_id1 != unique_id2:
        dist1 += d1
        dist2 += d2
        if isinstance(wt_parent1, WeightedTreeRoot):
            assert dist1 >= 0
            assert dist2 >= 0
            assert isinstance(wt_parent2, WeightedTreeRoot)
            return dist1 + dist2
        elif isinstance(wt_parent1, WeightedTreeChild):
            unique_id1 = wt_parent1.unique_id
            d1 = wt_parent1.dist
            wt_parent1 = wt_parent1.parent
        if isinstance(wt_parent2, WeightedTreeRoot):
            raise Exception("Nodes constructed have wrong depth.")
        elif isinstance(wt_parent2, WeightedTreeChild):
            unique_id2 = wt_parent2.unique_id
            d2 = wt_parent2.dist
            wt_parent2 = wt_parent2.parent
    return abs(dist1) + abs(dist2)


def geodesic_distance(
    wt1: WeightedTree, h1: float, wt2: WeightedTree, h2: float
) -> float:
    """
    Return the geodesic distance between p1=(wt1,h1) and p2=(wt2,h2).

    Here, p1 is a point in a weighted tree which lies at height h1 above wt1.
    Similarly, p2 is a point in a weighted tree which lies at height h2 above wt2.

    :param wt1: A node in a weighted tree.
    :param h1: Represents a point `p1` which lies `h1` above `wt1` in the tree, along \
    the line segment connecting `wt1` to its parent. `h1` is assumed to be less than the \
    distance between `wt1` and `wt1.parent`; or if `wt1` is a root node, \
    `h1` is assumed to be zero.
    :param wt2: A node in a weighted tree.
    :param h2: Represents a point `p2` which lies `h2` above `wt2` in the tree, along \
    the line segment connecting `wt2` to its parent. Similar assumptions as for `h1`.
    """
    if isinstance(wt1, WeightedTreeRoot):
        # If wt1 is a root, we assume h1 is zero and p1 = wt1
        # otherwise the input is not sensible.
        assert h1 == 0.0
        return weighted_dist_from_root(wt2) - h2
    elif isinstance(wt1, WeightedTreeChild):
        # Otherwise, suppose that wt1 is at an unweighted depth of depth1,
        # and that the distance between wt1 and its parent is d1.
        if isinstance(wt2, WeightedTreeRoot):
            # If wt2 is a root, then the approach is dual to what we have just done.
            assert h2 == 0.0
            return weighted_dist_from_root(wt1.parent) + wt1.dist - h1
        elif isinstance(wt2, WeightedTreeChild):
            # So let us consider the case where both wt1, wt2 are child nodes.
            return _geodesic_distance_children(wt1, h1, wt2, h2)


def get_sample_pts_geodesic(
    tree: NeuronTree, num_sample_pts: int
) -> list[tuple[WeightedTree, float]]:
    """
    Sample points uniformly throughout the body of `tree`, starting at \
    the root, returning a list of length `num_sample_pts`.

    "Sample points uniformly" means that there is some scalar `step_size` \
    such that a point `p` on a line segment of `tree` will be in the \
    return list iff its geodesic distance from the origin is an integer \
    multiple of `step_size`.

    :return: a list of pairs (wt, h), where `wt` is a node of `tree`, \
    and `h` is a floating point real number representing a point \
    `p` which lies a distance of `h` above `wt` on the line \
    segment between `wt` and its parent. If `wt` is a child node, \
    `h` is guaranteed to be less than the distance between `wt` \
    and its parent. If `wt` is a root, `h` is guaranteed to be zero.
    """
    assert num_sample_pts >= 0
    if num_sample_pts == 0:
        return []
    weighted_tree = WeightedTree_of(tree)
    if num_sample_pts == 1:
        return [(weighted_tree, 0.0)]
    max_depth = weighted_depth_wt(weighted_tree)
    if max_depth == 0.0:
        return [(weighted_tree, 0.0)] * num_sample_pts
    max_reps = 50
    counter = 0
    step_size = max_depth
    adjustment = step_size / 2
    while counter < max_reps:
        ell = _sample_at_given_stepsize_wt(weighted_tree, step_size)
        num_nodes_this_step_size = len(ell)
        if num_nodes_this_step_size < num_sample_pts:
            step_size -= adjustment
        elif num_nodes_this_step_size > num_sample_pts:
            step_size += adjustment
        else:
            return ell
        adjustment /= 2
        counter += 1
    raise Exception("Binary search timed out.")


def icdm_geodesic(
    tree: NeuronTree, num_samples: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
    r"""
    Compute the intracell distance matrix for `tree` using the geodesic metric.

    Sample `num_samples` many points uniformly throughout the body of `tree`, compute the
    pairwise geodesic distance between all sampled points, and return the matrix of distances.

    :return: A numpy array, a "condensed distance matrix" in the sense of
        :func:`scipy.spatial.distance.squareform`, i.e., an array of shape
        (`num_samples` \* `num_samples` - 1/2, ). Contains the entries in the intracell geodesic
        distance matrix for `tree` lying strictly above the diagonal.
    """
    pts_list = get_sample_pts_geodesic(tree, num_samples)
    dist_list = []
    for i in range(len(pts_list)):
        for j in range(i + 1, len(pts_list)):
            wt1, h1 = pts_list[i]
            wt2, h2 = pts_list[j]
            dist_list.append(geodesic_distance(wt1, h1, wt2, h2))
            assert (
                len(dist_list)
                == math.comb(num_samples, 2) - math.comb(num_samples - i, 2) + j - i
            )
    return np.array(dist_list), np.array(
        [wt.structure_id for (wt, _) in pts_list], dtype=np.int32
    )


def read_preprocess_compute_euclidean(
    file_name: str,
    n_sample: int,
    preprocess: Callable[[SWCForest], Union[Err[T], SWCForest]],
) -> Union[Err[T], tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]]:
    r"""
    Read the swc file, apply a preprocessor, and compute the Euclidean distance matrix.

    Read the \*.swc file `file_name` from disk as an `SWCForest`.
    Apply the function `preprocess` to the forest. If it returns an error, return that error.
    Otherwise, return the intracell distance matrix in vector form.
    """
    loaded_forest, _ = read_swc(file_name)
    forest = preprocess(loaded_forest)
    if isinstance(forest, Err):
        return forest
    if isinstance(forest, NeuronTree):
        icdm = icdm_euclidean([forest], n_sample)
        return icdm
    if isinstance(forest, list):
        if all([isinstance(tree, NeuronTree) for tree in forest]):
            icdm = icdm_euclidean(forest, n_sample)
            return icdm
    raise TypeError(
        "For computing Euclidean distance, preprocessing function must \
        return a list of NeuronTrees or an instance of the Err class. \
        Preprocessing function returned data of type "
        + str(type(forest))
    )


def read_preprocess_compute_geodesic(
    file_name: str,
    n_sample: int,
    preprocess: Callable[[SWCForest], Union[Err[T], NeuronTree]],
) -> Union[Err[T], tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]]:
    r"""
    Read the swc file, apply a preprocessor, and compute the geodesic distance matrix.

    Read the \*.swc file `file_name` from disk as an `SWCForest`. \
    Apply the function `preprocess` to the forest. If preprocessing returns an error,\
    return that error. \
    Otherwise return the preprocessed NeuronTree.

    :param file_name:
    """
    loaded_forest, _ = read_swc(file_name)
    tree = preprocess(loaded_forest)
    if isinstance(tree, Err):
        return tree
    if isinstance(tree, list) or (not isinstance(tree, NeuronTree)):
        raise ValueError(
            "For computing geodesic distance, preprocessing function must \
            return a NeuronTree or an instance of the Err class. \
            Preprocessing function returned data of type "
            + str(type(tree))
        )
    return icdm_geodesic(tree, n_sample)


def compute_icdm_all_euclidean(
    infolder: str,
    out_csv: str,
    out_node_types: str,
    n_sample: int,
    num_processes: int = 8,
    preprocess: Callable[[SWCForest], Union[Err[T], SWCForest]] = lambda forest: forest,
    name_validate: Callable[[str], bool] = default_name_validate,
    write_fn=write_csv_block,
) -> list[tuple[str, Err[T]]]:
    r"""
    Compute the intracell Euclidean distance matrices for all swc cells in `infolder`.

    For each \*.swc file in infolder, read the \*.swc file into memory as an
    SWCForest, `forest`.  Apply a preprocessing function `preprocess` to
    `forest`, which can return either an error message (because the file is for
    whatever reason unsuitable for processing or sampling) or a potentially
    modified SWCForest `processed_forest`. Sample n_sample many points from the
    neuron, evenly spaced, and compute the Euclidean intracell matrix. Write
    the resulting intracell distance matrices for all cells passing the
    preprocessing test to a csv file with path `out_csv`.

    :param infolder: Directory of input \*.swc files.
    :param out_csv: Output file to write the pairwise distances to.
    :param out_node_types: Output file to write the node labels
    :param n_sample: How many points to sample from each cell.
    :param preprocess: `preprocess` is expected to be roughly of the following form:

        #. Apply such-and-such tests of data quality and integrity to the
           SWCForest. (For example, check that the forest has only a single
           connected component, that it has only a single soma node, that it has
           at least one soma node, that it contains nodes from the axon, that
           it does not have any elements whose structure_id is 0 (for
           'undefined'), etc.)
        #. If any of the tests are failed, return an instance
           of :class:`utilities.Err` with a message explaining why the \*.swc
           file was ineligible for sampling.
        #. If all tests are passed, apply a transformation to `forest` and return
           the modified `new_forest`. (For example, filter out all axon nodes to
           focus on the dendrites, or filter out all undefined nodes, or filter out
           all components which have fewer than 10% of the nodes in the
           largest component.)

        If `preprocess(forest)` returns an instance of the
        :class:`utilities.Err` class, this file is not sampled from, and its
        name is added to a list together with the error returned by
        `preprocess`. If `preprocess(forest)` returns a SWCForest, this is what
        will be sampled. By default, no preprocessing is performed, and the
        neuron is processed as-is.

    :param name_validate: A boolean test on strings. Files will be read from the directory
        if name_validate is True (truthy).
    :return: List of pairs (cell_name, error), where cell_name is the cell for
        which sampling failed, and `error` is a wrapper around a message indicating
        why the neuron was not sampled from.
    """
    cell_names, file_paths = get_filenames(infolder, name_validate)
    assert len(cell_names) == len(file_paths)

    def rpce(file_path: str) -> Union[Err[T], npt.NDArray[np.float64]]:
        return read_preprocess_compute_euclidean(file_path, n_sample, preprocess)

    # args = zip([file_paths,repeat(n_sample),repeat(preprocess)])
    # icdms: Iterator[Union[Err[T], npt.NDArray[np.float64]]]
    failed_cells: list[tuple[str, Err[T]]]

    pool = ProcessPool(nodes=num_processes)
    results = tqdm(pool.imap(rpce, file_paths), total=len(cell_names))
    failed_cells = write_fn(
        out_csv, n_sample, zip(cell_names, results), 10, out_node_types=out_node_types
    )
    pool.close()
    pool.join()
    pool.clear()
    return failed_cells


def compute_icdm_all_geodesic(
    infolder: str,
    out_csv: str,
    out_node_types: str,
    n_sample: int,
    num_processes: int = 8,
    preprocess: Callable[
        [SWCForest], Union[Err[T], NeuronTree]
    ] = lambda forest: forest[0],
    name_validate: Callable[[str], bool] = default_name_validate,
    write_fn=write_csv_block,
) -> list[tuple[str, Err[T]]]:
    """
    Compute the intracell geodesic distance matrices for all swc cells in `infolder`.

    This function is substantially the same as \
    :func:`cajal.sample_swc.compute_icdm_all_euclidean` and the user should \
    consult the documentation for that function. However, note that \
    `preprocess`  has a different type signature, it is expected to return a `NeuronTree` \
    rather than an `SWCForest`. There is not a meaningful notion of geodesic distance \
    between points in two different components of a graph.

    The default preprocessing is to take the largest component.
    """
    cell_names, file_paths = get_filenames(infolder, name_validate)

    def rpcg(file_path) -> Union[Err[T], npt.NDArray[np.float64]]:
        return read_preprocess_compute_geodesic(file_path, n_sample, preprocess)

    failed_cells: list[tuple[str, Err[T]]]
    pool = ProcessPool(nodes=num_processes)
    results = tqdm(pool.imap(rpcg, file_paths), total=len(cell_names))
    failed_cells = write_fn(
        out_csv, n_sample, zip(cell_names, results), 10, out_node_types=out_node_types
    )
    pool.close()
    pool.join()
    pool.clear()
    return failed_cells


controller = ThreadpoolController()


@controller.wrap(limits=1, user_api="blas")
def fused_gromov_wasserstein(
    cell1_dmat: DistanceMatrix,
    cell1_distribution: Distribution,
    cell1_node_types: npt.NDArray[np.int32],
    cell2_dmat: DistanceMatrix,
    cell2_distribution: Distribution,
    cell2_node_types: npt.NDArray[np.int32],
    penalty_dictionary: dict[tuple[int, int], float],
    worst_case_gw_increase: Optional[float] = None,
    **kwargs,
):
    """
    Compute the fused Gromov-Wasserstein distance between cells.

    Penalties for mismatched node types should be supplied by the user.
    """
    penalty_matrix = np.zeros(
        shape=(cell1_node_types.shape[0], cell2_node_types.shape[0])
    )
    for (i, j), p in penalty_dictionary.items():
        penalty_matrix += (
            np.logical_and(
                (cell1_node_types == i)[:, np.newaxis],
                (cell2_node_types == j)[np.newaxis, :],
            )
            * p
        )
        penalty_matrix += (
            np.logical_and(
                (cell1_node_types == j)[:, np.newaxis],
                (cell2_node_types == i)[np.newaxis, :],
            )
            * p
        )

    if worst_case_gw_increase is not None:
        (plan, log) = ot.gromov.gromov_wasserstein(
            cell1_dmat,
            cell2_dmat,
            p=cell1_distribution,
            q=cell2_distribution,
            symmetric=True, log=True)
        G0 = plan
        scalar = worst_case_gw_increase * log['gw_dist']/((penalty_matrix * plan).sum())
        penalty_matrix *= scalar
    else:
        G0 = cell1_distribution[:,np.newaxis] * cell2_distribution[np.newaxis,:]
    
    return ot.fused_gromov_wasserstein(
        penalty_matrix,
        cell1_dmat,
        cell2_dmat,
        p=cell1_distribution,
        q=cell2_distribution,
        G0=G0,
        **kwargs,
    )

# @controller.wrap(limits=1, user_api="blas")
# def fused_gromov_wasserstein_adaptive(
#     cell1_dmat: DistanceMatrix,
#     cell1_distribution: Distribution,
#     cell1_node_types: npt.NDArray[np.int32],
#     cell2_dmat: DistanceMatrix,
#     cell2_distribution: Distribution,
#     cell2_node_types: npt.NDArray[np.int32],
#     penalty_dictionary: dict[tuple[int, int], float],
#     max_gw_increase_percentage: float,
#     **kwargs,
# ):
#     """
#     Compute the fused Gromov-Wasserstein distance between cells.

#     Penalties for mismatched node types should be supplied by the user.
#     """
#     penalty_matrix = np.zeros(
#         shape=(cell1_node_types.shape[0], cell2_node_types.shape[0])
#     )
#     for (i, j), p in penalty_dictionary.items():
#         penalty_matrix += (
#             np.logical_and(
#                 (cell1_node_types == i)[:, np.newaxis],
#                 (cell2_node_types == j)[np.newaxis, :],
#             )
#             * p
#         )
#         penalty_matrix += (
#             np.logical_and(
#                 (cell1_node_types == j)[:, np.newaxis],
#                 (cell2_node_types == i)[np.newaxis, :],
#             )
#             * p
#         )

#     (plan, log) = ot.gromov.gromov_wasserstein(
#         cell1_dmat,
#         cell2_dmat,
#         p=cell1_distribution,
#         q=cell2_distribution,
#         symmetric=True, log=True)

#     penalty_matrix *= log['gw_dist']/( (penalty_matrix * plan).sum() )
#     return ot.fused_gromov_wasserstein(
#         penalty_matrix,
#         cell1_dmat,
#         cell2_dmat,
#         p=cell1_distribution,
#         q=cell2_distribution,
#         G0=plan
#         **kwargs,
#     )

def _init_fgw_pool(
    cells: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]],
    node_types: npt.NDArray[np.int32],
    penalty_dictionary: dict[tuple[int, int], float],
    worst_case_gw_increase: Optional[float],
    kwargs: dict[str, Any],
):
    """
    Set a few global variables so that the parallel process pool can access them.

    This is a private function.
    """
    global _CELLS
    _CELLS = cells
    global _NODE_TYPES
    _NODE_TYPES = node_types
    global _WORST_CASE_GW_INCREASE
    _WORST_CASE_GW_INCREASE=worst_case_gw_increase
    global _KWARGS
    _KWARGS = kwargs
    global _PENALTY_DICTIONARY
    _PENALTY_DICTIONARY = penalty_dictionary


def _fgw_index(p: tuple[int, int]):
    """Compute the Fused GW distance between cells i and j in the master cell list."""
    i, j = p
    (_, log) = fused_gromov_wasserstein(
        _CELLS[i][0],
        _CELLS[i][1],
        _NODE_TYPES[i],
        _CELLS[j][0],
        _CELLS[j][1],
        _NODE_TYPES[j],
        _PENALTY_DICTIONARY,
        _WORST_CASE_GW_INCREASE,
        **_KWARGS,
    )
    return (i, j, log["fgw_dist"])


def _sort_distances(dmat, node_types):
    soma_nodes = node_types == 1
    if np.any(soma_nodes):
        distance_from_soma_nodes = np.sum(dmat[soma_nodes, :], axis=0)
        min_index = np.argmin(distance_from_soma_nodes)
        sort_by = np.argsort(dmat[min_index])
        dmat = dmat[:, sort_by][sort_by, :]
        node_types = node_types[sort_by]
        return (dmat, node_types)
    else:
        distances = np.sum(dmat, axis=0)
        min_index = np.argmin(distances)
        sort_by = np.argsort(dmat[min_index])
        dmat = dmat[:, sort_by][sort_by, :]
        node_types = node_types[sort_by]
        return (dmat, node_types)


def gw_cost(A, a, B, b, P):
    """
    Compute the GW cost of the given transport plan.

    (A,a) and (B, b) are metric measure spaces. P is a transport plan.
    """
    c_A = ((A * A) @ a) @ a
    c_B = ((B * B) @ b) @ b
    return c_A + c_B - 2 * ((A @ P @ B) * P).sum()


def gw_dist(A, a, B, b, P):
    """
    Compute the GW distance of the given transport plan.

    We distinguish between GW distance and GW cost. GW distance is a metric,
    and GW cost is simpler to work with.
    """
    return math.sqrt(gw_cost(A, a, B, b, P)) / 2


def gw_cost_unif(A, a, B, b):
    """Compute the GW cost of the uniform transport plan."""
    c_A = ((A * A) @ a) @ a
    c_B = ((B * B) @ b) @ b
    Aa = A @ a
    Bb = B @ b
    return (
        c_A
        + c_B
        - 2
        * (
            np.multiply(Aa[:, np.newaxis], Bb[np.newaxis, :], order="C")
            * (a[:, np.newaxis] * b[np.newaxis, :])
        ).sum()
    )


def gw_cost_upper_bound(A_dmat, a_distr, A_node_types, B_dmat, b_distr, B_node_types):
    """
    Compute a simple upper bound to the GW cost between spaces.

    This function was written for the simple case where (A_dmat, a_distr) and
    (B_dmat, b_distr) are metric measure spaces of the same dimensions. It
    will fail if A_dmat is the wrong size.
    """
    A_dmat_sorted, a_distr_sorted = _sort_distances(A_dmat, a_distr)
    B_dmat_sorted, b_distr_sorted = _sort_distances(B_dmat, b_distr)
    gw_cost(
        A_dmat_sorted,
        a_distr_sorted,
        B_dmat_sorted,
        b_distr_sorted,
        np.eye(N=A_dmat.shape[0]),
    )


def fused_gromov_wasserstein_parallel(
    intracell_csv_loc: str,
    swc_node_types: str,
    fgw_dist_csv_loc: str,
    num_processes: int,
    soma_dendrite_penalty: float,
    basal_apical_penalty: float,
    penalty_dictionary: Optional[dict[tuple[int, int], float]] = None,
    chunksize: int = 20,
    sample_points_npz: Optional[str] = None,
    worst_case_gw_increase: Optional[float] = None,
    **kwargs,
):
    """
    Compute the fused GW distance pairwise in parallel between many neurons.

    :param intracell_csv_loc: The path to the file where the sampled points are stored.
    :param swc_node_types: The path to the swc node type file.
    :param fgw_dist_csv_loc: Where you want the fused GW distances to be written.
    :param num_processes: How many parallel processes you want this to run on.
    :param soma_dendrite_penalty: This represents the penalty paid by the transport plan
    for aligning a soma node with a dendrite node. By choosing this coefficient
    sufficiently large, the algorithm favors transport plans which align soma nodes
    to soma nodes and dendrite nodes to dendrite nodes. Choosing the coefficient
    to be too large may be counterproductive.
    :param basal_apical_penalty: The penalty paid by the transport plan for aligning
    a basal dendrite node with an apical dendrite node, if this distinction is
    indeed captured in the morphological reconstructions.
    :param penalty_dictionary: The user can choose the penalty
    to align nodes of any two different types. For example, if their
    data contains nodes with structure id's 3,4 and 5, the user
    can impose a penalty for joining a node of type 3 to a node of type 4,
    4 to 5, and 3 to 5. If this parameter is supplied then
    the previous two parameters are ignored as this parameter overrides them;
    the user can reproduce the behavior by adding penalty keys for (1,3), (1,4)
    and (3,4) appropriately.
    :param chunksize: A parallelization parameter, the
    number of jobs fed to each process at a time.
    """
    cells: list[tuple[DistanceMatrix, Distribution]]
    node_types: npt.NDArray[np.int32]
    names: list[str]

    if sample_points_npz is None:
        cell_names_dmats = list(cell_iterator_csv(intracell_csv_loc))
        node_types = np.load(swc_node_types)
        cells = [(c := cell, uniform(c.shape[0])) for _, cell in cell_names_dmats]
        names = [name for name, _ in cell_names_dmats]
    else:
        a = np.load(sample_points_npz)
        cells = a["dmats"]
        k = cells.shape[1]
        n = int(math.ceil(math.sqrt(k * 2)))
        u = uniform(n)
        cells = [(squareform(cell,force='tosquareform'), u) for cell in cells]
        node_types = a["structure_ids"]
        names = a["names"]
        a.close()

    num_cells = len(names)
    # List of pairs (A, a) where A is a square matrix and `a` a probability distribution
    # compute pairwise fGW distances between all objects

    index_pairs = it.combinations(
        iter(range(num_cells)), 2
    )  # object pairs to compute fGW / OT for
    total_num_pairs = int(
        (num_cells * (num_cells - 1)) / 2
    )  # total number of object pairs to compute (for progress bar)
    kwargs["log"] = True

    if penalty_dictionary is None:
        penalty_dictionary = dict()
        penalty_dictionary[(1, 3)] = (soma_dendrite_penalty,)
        penalty_dictionary[(1, 4)] = (soma_dendrite_penalty,)
        penalty_dictionary[(3, 4)] = basal_apical_penalty

    with Pool(
        initializer=_init_fgw_pool,
        initargs=(cells, node_types, penalty_dictionary, worst_case_gw_increase, kwargs),
        processes=num_processes,
    ) as pool:
        res = pool.imap_unordered(_fgw_index, index_pairs, chunksize=chunksize)
        # store GW distances
        fgw_dmat = np.zeros((num_cells, num_cells))
        for i, j, fgw_dist in tqdm(res, total=total_num_pairs, position=0, leave=True):
            fgw_dmat[i, j] = fgw_dist
            fgw_dmat[j, i] = fgw_dist

    with open(fgw_dist_csv_loc, "w") as outfile:
        csvwrite = csv.writer(outfile)
        csvwrite.writerow(["first_object", "second_object", "gw_distance"])
        for i, j in it.combinations(iter(range(num_cells)), 2):
            csvwrite.writerow([names[i], names[j], str(fgw_dmat[i, j])])

    return fgw_dmat
