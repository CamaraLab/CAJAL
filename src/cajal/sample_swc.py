"""Functions for sampling points from an SWC reconstruction of a neuron."""

import math
from typing import Callable, Union
import numpy as np
from pathos.pools import ProcessPool
import numpy.typing as npt
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
from .utilities import Err, T, write_csv_block
from .weighted_tree import (
    WeightedTree,
    WeightedTree_of,
    WeightedTreeChild,
    WeightedTreeRoot,
    weighted_depth_wt,
    weighted_dist_from_root,
)

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
