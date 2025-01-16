"""
Helper functions.
"""

from dataclasses import dataclass
import csv
from scipy.spatial.distance import squareform
from scipy.sparse import coo_array
import itertools as it
from typing import Iterator, Iterable, Optional, TypeVar, Generic, Union, Callable
from sklearn.neighbors import NearestNeighbors
import leidenalg
import community as community_louvain
import igraph as ig
import networkx as nx
from math import ceil, sqrt
from .cajal_types import DistanceMatrix

from scipy.sparse.csgraph import dijkstra

import numpy as np
import numpy.typing as npt


def read_gw_dists(
    gw_dist_file_loc: str, header: bool
) -> tuple[list[str], dict[tuple[str, str], float]]:
    r"""
    Read a GW distance matrix into memory.

    :param gw_dist_file_loc: A file path to a Gromov-Wasserstein distance matrix. \
    The distance matrix should be a CSV file with at least three columns and possibly \
    a single header line (which is ignored). All \
    following lines should be two strings cell_name1, cell_name2 followed by a \
    floating point real number. Entries after the three columns are discarded.

    :param header: If `header` is True, the very first line of the file is discarded. \
    If `header` is False, all lines are assumed to be relevant.

    :returns: A pair (cell_names, gw_dist_dictionary), where \
    cell_names is a list of cell names in alphabetical order, gw_dist_dictionary \
    is a dictionary of the GW distances  which can be read like \
    gw_dist_dictionary[(cell_name1, cell_name2)], where cell_name1 and cell_name2 \
    are in alphabetical order. gw_dist_list is a vector-form array (rank 1) of the \
    GW distances.
    """
    gw_dist_dict: dict[tuple[str, str], float] = {}
    with open(gw_dist_file_loc, "r", newline="") as gw_file:
        csvreader = csv.reader(gw_file, delimiter=",")
        if header:
            _ = next(csvreader)
        for line in csvreader:
            first_cell, second_cell, gw_dist_str = line[0:3]
            gw_dist = float(gw_dist_str)
            first_cell, second_cell = sorted([first_cell, second_cell])
            gw_dist_dict[(first_cell, second_cell)] = gw_dist
    all_cells_set = set()
    for cell_1, cell_2 in gw_dist_dict:
        all_cells_set.add(cell_1)
        all_cells_set.add(cell_2)
    all_cells = sorted(list(all_cells_set))
    return all_cells, gw_dist_dict


def dist_mat_of_dict(
    gw_dist_dictionary: dict[tuple[str, str], float],
    cell_names: Iterable[str],
    as_squareform: bool = True,
) -> npt.NDArray[np.float64]:
    """
    Given a distance dictionary and a list of cell names, return a square distance \
    matrix containing the pairwise GW distances between all cells in `cell_names`, and \
    in the same order.\

    It is assumed that the keys in `gw_dist_dictionary` are in alphabetical order.
    """
    if cell_names is None:
        names = set()
        for key in gw_dist_dictionary:
            names.add(key[0])
            names.add(key[1])
        cell_names = sorted(names)
    dist_list: list[float] = []
    for first_cell, second_cell in it.combinations(cell_names, 2):
        first_cell, second_cell = sorted([first_cell, second_cell])
        dist_list.append(gw_dist_dictionary[(first_cell, second_cell)])
    arr = np.array(dist_list, dtype=np.float64)
    if as_squareform:
        return squareform(arr, force="tomatrix")
    return arr


def update_names(f: Callable, gw_dist_dict: dict):
    """
    Given f a function and gw_dists a dictionary of pairwise GW distances,
    return a new gw distance dictionary with entry (a,b) replaced with (f(a),f(b)) if f(a)<f(b)
    or (f(b),f(a)) if f(b)<f(a).

    If f is not injective then the resulting distance dictionary may be unusable.
    The function raises an exception on collisions.

    Codomain of f is assumed to work with `<`.
    """
    gw_dist_dict1 = dict()
    for i, j in gw_dist_dict:
        i1 = f(i)
        j1 = f(j)
        if i1 == j1:
            raise Exception("f is not injective.")
        elif j1 < i1:
            tmp = j1
            j1 = i1
            i1 = tmp
        assert i1 < j1
        gw_dist_dict1[(i1, j1)] = gw_dist_dict[(i, j)]
    return gw_dist_dict1


def read_gw_dists_pd(gw_dist_file_loc: str, header: bool):
    import pandas as pd

    cell_names, cell_dict = read_gw_dists(gw_dist_file_loc, header)
    gw_dmat = dist_mat_of_dict(cell_dict, cell_names, as_squareform=True)
    return pd.Series(
        np.ndarray.flatten(gw_dmat),
        index=pd.MultiIndex.from_product(
            (cell_names, cell_names), names=["first", "second"]
        ),
    )


def read_gw_couplings(gw_coupling_mat_npz_loc: str):
    """Read Gromov-Wasserstein coupling matrices into memory.
    :param gw_coupling_mat_npz_loc: A filepath to an npz file
        with fields "first_names", "second_names", "coo_data", "coo_row",
        "coo_col". The names are not expected to be in any kind of sorted order.
    :returns: A dictionary whose keys are pairs (name1, name2) and whose values
        are scipy coo arrays.
    """
    with np.load(gw_coupling_mat_npz_loc) as npz:
        first_names = npz["first_names"]
        second_names = npz["second_names"]
        coo_data = npz["coo_data"]
        coo_row = npz["coo_row"]
        coo_col = npz["coo_col"]

    gw_coupling_mats = dict()
    iterator = zip(first_names, second_names, coo_data, coo_row, coo_col)
    for first_name, second_name, data, row, col in iterator:
        coo = coo_array((data, (row, col)))
        first_name, second_name = sorted([first_name, second_name])
        gw_coupling_mats[(first_name, second_name)] = coo
    return gw_coupling_mats


# def read_gw_couplings(
#     gw_couplings_file_loc: str, header: bool
# ) -> dict[tuple[str, str], npt.NDArray[np.float64]]:
#     """
#     Read a list of Gromov-Wasserstein coupling matrices into memory.
#     :param header: If True, the first line of the file will be ignored.
#     :param gw_couplings_file_loc: name of a file holding a list of GW coupling matrices in \
#     COO form. The files should be in csv format. Each line should be of the form
#     `cellA_name, cellA_sidelength, cellB_name, cellB_sidelength, num_nonzero,
#     (data), (row), (col)`
#     where `data` is a sequence of `num_nonzero` many floating point real numbers,
#     `row` is a sequence of `num_nonzero` many integers (row indices), and
#     `col` is a sequence of `num_nonzero` many integers (column indices).
#     :return: A dictionary mapping pairs of names (firstcell, secondcell) to the GW \
#     matrix of the coupling. `firstcell` and `secondcell` are in alphabetical order.
#     """

#     gw_coupling_mat_dict: dict[tuple[str, str], npt.NDArray[np.float64]] = {}
#     with open(gw_couplings_file_loc, "r", newline="") as gw_file:
#         csvreader = csv.reader(gw_file, delimiter=",")
#         linenum = 1
#         if header:
#             _ = next(csvreader)
#             linenum += 1
#         for line in csvreader:
#             cellA_name = line[0]
#             cellA_sidelength = int(line[1])
#             cellB_name = line[2]
#             cellB_sidelength = int(line[3])
#             num_non_zero = int(line[4])
#             rest = line[5:]
#             if 3 * num_non_zero != len(rest):
#                 raise Exception(
#                     "On line " + str(linenum) + " data not in COO matrix form."
#                 )
#             data = [float(x) for x in rest[:num_non_zero]]
#             rows = [int(x) for x in rest[num_non_zero : (2 * num_non_zero)]]
#             cols = [int(x) for x in rest[(2 * num_non_zero) :]]
#             coo = coo_array(
#                 (data, (rows, cols)), shape=(cellA_sidelength, cellB_sidelength)
#             )
#             linenum += 1
#             if cellA_name < cellB_name:
#                 gw_coupling_mat_dict[(cellA_name, cellB_name)] = coo
#             else:
#                 gw_coupling_mat_dict[(cellB_name, cellA_name)] = coo_array.transpose(
#                     coo
#                 )
#     return gw_coupling_mat_dict


T = TypeVar("T")


@dataclass
class Err(Generic[T]):
    code: T


def write_csv_block(
    out_csv: str,
    sidelength: int,
    dist_mats: Iterator[tuple[str, Union[Err[T], npt.NDArray[np.float64]]]],
    batch_size: int,
    **kwargs,
) -> list[tuple[str, Err[T]]]:
    """
    :param sidelength: The side length of all matrices in dist_mats.
    :param dist_mats: an iterator over pairs (name, arr), where arr is an
    vector-form array (rank 1) or an error code.
    """
    fused = "out_node_types" in kwargs

    failed_cells: list[tuple[str, Err[T]]] = []
    if fused:
        node_types: list[npt.NDArray[np.int32]] = []

    with open(out_csv, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=",")
        firstline = ["cell_id"] + [
            "d_%d_%d" % (i, j) for i, j in it.combinations(range(sidelength), 2)
        ]
        csvwriter.writerow(firstline)
        while next_batch := list(it.islice(dist_mats, batch_size)):
            good_cells: list[list[Union[str, float]]] = []
            for name, cell in next_batch:
                if isinstance(cell, Err):
                    failed_cells.append((name, cell))
                elif fused:
                    good_cells.append([name] + cell[0].tolist())
                    node_types.append(cell[1])
                else:  # Not fused
                    good_cells.append([name] + cell.tolist())
            csvwriter.writerows(good_cells)

    if fused:
        stacks = np.stack(node_types)
        with open(kwargs["out_node_types"], "wb") as f:
            np.save(f, stacks)
    return failed_cells


def write_npz(
    out_npz: str,
    sidelength: int,
    dist_mats: Iterator[tuple[str, Union[Err[T], npt.NDArray[np.float64]]]],
    batch_size: int,
    **kwargs,
) -> list[tuple[str, Err[T]]]:
    """
    Write the stream to an npz file. This writing method keeps all data in memory
    at one time so it may be inappropriate for situations where the point clouds are
    large or there are many cells.

    :param sidelength: The side length of all matrices in dist_mats.
    :param dist_mats: an iterator over pairs (name, arr), where arr is an
    vector-form array (rank 1) or an error code.
    """

    fused = "out_node_types" in kwargs
    names = []
    dmats = []
    failed_cells: list[tuple[str, Err[T]]] = []
    if fused:
        node_types: list[npt.NDArray[np.int32]] = []

    for name, result in dist_mats:
        if isinstance(result, Err):
            failed_cells.append((name, result))
        else:
            names.append(name)
            if fused:
                dmats.append(result[0])
                node_types.append(result[1])
            else:
                dmats.append(result)

    with open(out_npz, "wb") as npzfile:
        if fused:
            np.savez(npzfile, names=names, dmats=dmats, structure_ids=node_types)
        else:
            np.savez(npzfile, names=names, dmats=dmats)

    return failed_cells


def knn_graph(dmat: npt.NDArray[np.float64], nn: int) -> npt.NDArray[np.int_]:
    """
    :param dmat: squareform distance matrix
    :param nn: (nearest neighbors) - in the returned graph, nodes v and w will be \
    connected if v is one of the `nn` nearest neighbors of w, or conversely.
    :return: A (1,0)-valued adjacency matrix for a nearest neighbors graph, same shape as dmat.
    """
    a = np.argpartition(dmat, nn + 1, axis=0)
    sidelength = dmat.shape[0]
    graph = np.zeros((sidelength, sidelength), dtype=np.int_)
    for i in range(graph.shape[1]):
        graph[a[0 : (nn + 1), i], i] = 1
    graph = np.maximum(graph, graph.T)
    np.fill_diagonal(graph, 0)
    return graph


def louvain_clustering(
    gw_mat: npt.NDArray[np.float64], nn: int
) -> npt.NDArray[np.int_]:
    """
    Compute clustering of cells based on GW distance, using Louvain clustering on a
    nearest-neighbors graph

    :param gw_mat: NxN distance matrix of GW distance between cells
    :param nn: number of neighbors in nearest-neighbors graph
    :return: numpy array of shape (num_cells,) the cluster assignment for each cell
    """
    nn_model = NearestNeighbors(n_neighbors=nn, metric="precomputed")
    nn_model.fit(gw_mat)
    adj_mat = nn_model.kneighbors_graph(gw_mat).todense()
    np.fill_diagonal(adj_mat, 0)

    graph = nx.convert_matrix.from_numpy_array(adj_mat)
    # louvain_clus_dict is a dictionary whose keys are nodes of `graph` and whose
    # values are natural numbers indicating communities.
    louvain_clus_dict = community_louvain.best_partition(graph, randomize=False)
    louvain_clus = np.array([louvain_clus_dict[x] for x in range(gw_mat.shape[0])])
    return louvain_clus


def leiden_clustering(
    gw_mat: npt.NDArray[np.float64],
    nn: int = 5,
    resolution: Optional[float] = None,
    seed: Optional[int] = None,
) -> npt.NDArray[np.int_]:
    """
    Compute clustering of cells based on GW distance, using Leiden clustering on a
    nearest-neighbors graph

    :param gw_mat: NxN distance matrix of GW distance between cells
    :param nn: number of neighbors in nearest-neighbors graph
    :param resolution: If None, use modularity to get optimal partition.
        If float, get partition at set resolution.
    :param seed: Seed for the random number generator.
        Uses a random seed if nothing is specified.
    :return: numpy array of cluster assignment for each cell
    """
    nn_model = NearestNeighbors(n_neighbors=nn, metric="precomputed")
    nn_model.fit(gw_mat)
    adj_mat = nn_model.kneighbors_graph(gw_mat).todense()
    np.fill_diagonal(adj_mat, 0)

    graph = ig.Graph.Adjacency((adj_mat > 0).tolist())
    graph.es["weight"] = adj_mat[adj_mat.nonzero()]
    graph.vs["label"] = range(adj_mat.shape[0])

    if resolution is None:
        leiden_clus = np.array(
            leidenalg.find_partition_multiplex(
                [graph], leidenalg.ModularityVertexPartition, seed=seed
            )[0]
        )
    else:
        leiden_clus = np.array(
            leidenalg.find_partition_multiplex(
                [graph],
                leidenalg.CPMVertexPartition,
                resolution_parameter=resolution,
                seed=seed,
            )[0]
        )
    return leiden_clus


def identify_medoid(
    cell_names: list, gw_dist_dict: dict[tuple[str, str], float]
) -> str:
    """
    Identify the medoid cell in cell_names.
    """
    return cell_names[
        np.argmin(
            dist_mat_of_dict(gw_dist_dict, cell_names, as_squareform=True).sum(axis=0)
        )
    ]


def cap(a: npt.NDArray[np.float64], c: float) -> npt.NDArray[np.float64]:
    """
    Return a copy of `a` where values above `c` in `a` are replaced with `c`.
    """
    a1 = np.copy(a)
    a1[a1 >= c] = c
    return a1


def step_size(icdm: npt.NDArray[np.float64]) -> float:
    """
    Heuristic to estimate the step size a neuron was sampled at.
    :param icdm: Vectorform distance matrix.
    """
    return float(np.min(icdm))


def orient(
    medoid: str,
    obj_name: str,
    iodm: npt.NDArray[np.float64],
    gw_coupling_mat_dict: dict[tuple[str, str], coo_array],
) -> npt.NDArray[np.float64]:
    """
    :param medoid: String naming the medoid object, its key in iodm
    :param obj_name: String naming the object to be compared to
    :param iodm: intra-object distance matrix given in square form
    :param gw_coupling_mat_dict: maps pairs (objA_name, objB_name) to scipy COO matrices
    :return: "oriented" squareform distance matrix
    """
    if obj_name < medoid:
        gw_coupling_mat = gw_coupling_mat_dict[(obj_name, medoid)]
    else:
        gw_coupling_mat = gw_coupling_mat_dict[(medoid, obj_name)].transpose()

    i_reorder = np.argmax(gw_coupling_mat.todense(), axis=0)
    return iodm[i_reorder][:, i_reorder]


def avg_shape(
    obj_names: list[str],
    gw_dist_dict: dict[tuple[str, str], float],
    iodms: dict[str, npt.NDArray[np.float64]],
    gw_coupling_mat_dict: dict[tuple[str, str], coo_array],
):
    """
    Compute capped and uncapped average distance matrices. \
    In both cases the distance matrix is rescaled so that the minimal distance between two points \
    is 1. The "capped" distance matrix has a max distance of 2.

    :param obj_names: Keys for the gw_dist_dict and iodms.
    :param gw_dist_dict: Dictionary mapping ordered pairs (cellA_name, cellB_name) \
    to Gromov-Wasserstein distances.
    :param iodms: (intra-object distance matrices) - \
    Maps object names to intra-object distance matrices. Matrices are assumed to be given \
    in vector form rather than squareform.
    :param gw_coupling_mat_dict: Dictionary mapping ordered pairs (cellA_name, cellB_name) to \
    Gromov-Wasserstein coupling matrices from cellA to cellB.
    """
    num_objects = len(obj_names)
    medoid = identify_medoid(obj_names, gw_dist_dict)
    medoid_matrix = iodms[medoid]
    # Rescale to unit step size.
    ss = step_size(medoid_matrix)
    assert ss > 0
    medoid_matrix = medoid_matrix / step_size(medoid_matrix)
    dmat_accumulator_uncapped = np.copy(medoid_matrix)
    dmat_accumulator_capped = cap(medoid_matrix, 2.0)
    others = (obj for obj in obj_names if obj != medoid)
    for obj_name in others:
        iodm = iodms[obj_name]
        # Rescale to unit step size.
        iodm = iodm / step_size(iodm)
        reoriented_iodm = squareform(
            orient(
                medoid,
                obj_name,
                squareform(iodm, force="tomatrix"),
                gw_coupling_mat_dict,
            ),
            force="tovector",
        )
        # reoriented_iodm is not a distance matrix - it is a "pseudodistance matrix".
        # If X and Y are sets and Y is a metric space, and f : X -> Y, then \
        # d_X(x0, x1) := d_Y(f(x0),f(x1)) is a pseudometric on X.
        dmat_accumulator_uncapped += reoriented_iodm
        dmat_accumulator_capped += cap(reoriented_iodm, 2.0)
    # dmat_avg_uncapped can have any positive values, but none are zero,
    # because medoid_matrix is not zero anywhere.
    # dmat_avg_capped has values between 0 and 2, exclusive.
    return (
        dmat_accumulator_capped / num_objects,
        dmat_accumulator_uncapped / num_objects,
    )


def avg_shape_spt(
    cell_names: list[str],
    gw_dist_dict: dict[tuple[str, str], float],
    icdms: dict[str, npt.NDArray[np.float64]],
    gw_coupling_mat_dict: dict[tuple[str, str], coo_array],
    k: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Given a set of cells together with their intracell distance matrices and
    the (precomputed) pairwise GW coupling matrices between cells, construct a
    morphological "average" of cells in the cluster. This function:

    * aligns all cells in the cluster with each other using the coupling matrices
    * takes a "local average" of all intracell distance matrices, forming a
      distance matrix which models the average local connectivity structure of the neurons
    * draw a neighborh
    * draws a minimum spanning tree through the intracell distance graph,
      allowing us to visualize this average morphology

    :param cell_names: The cluster you want to take the average of,
        expressed as a list of names of cells in the cluster. These should be
        names that occur in the keys for the other dictionary arguments.
    :param gw_dist_dict: Dictionary mapping ordered pairs (cellA_name, cellB_name)
        to Gromov-Wasserstein distances between them, as returned by
        cajal.utilities.dist_mat_of_dict.
    :param icdms: (intra-cell distance matrices) -
        Maps cell names to intra-cell distance matrices. Matrices are assumed to be given
        in vector form rather than squareform. Intracell distances are
        computed by any of the sampling functions in sample_swc, sample_seg, etc.
        and are read from file by cell_iterator_csv.
    :param gw_coupling_mat_dict: Dictionary mapping ordered pairs (cellA_name, cellB_name) to
        Gromov-Wasserstein coupling matrices from cellA to cellB, with
        cellA_name < cellB_name lexicographically
    :param k: how many neighbors in the nearest-neighbors graph in step 3
    :returns: A pair (adjacency_matrix, confidence) where adjacency_matrix
        is a Numpy matrix of shape (n, n)  (where n is the number of points in each sampled cell)
        and confidence is an array of shape (n)  adjacency_matrix has values between 0 and 2.
        When "confidence" at a node in the average graph is high, the node is not
        very close to its nearest neighbor.  We can think of this as saying that
        this node in the averaged graph is a kind of poorly amalgamated blend of
        different features in different graphs.  Conversely, when confidence is
        low, and the node is close to its nearest neighbor, we interpret this as
        meaning that this node and its nearest neighbor appear together in many
        of the graphs being averaged, so this is potentially a good
        representation of some edge that really appears in many of the graphs.
    """
    dmat_avg_capped, dmat_avg_uncapped = avg_shape(
        cell_names, gw_dist_dict, icdms, gw_coupling_mat_dict
    )
    dmat_avg_uncapped = squareform(dmat_avg_uncapped)
    # So that 0s along diagonal don't get caught in min
    np.fill_diagonal(dmat_avg_uncapped, np.max(dmat_avg_uncapped))
    # When confidence at a node in the average graph is high, the node is not
    # very close to its nearest neighbor.  We can think of this as saying that
    # this node in the averaged graph is a kind of poorly amalgamated blend of
    # different features in different graphs.  Conversely, when confidence is
    # low, and the node is close to its nearest neighbor, we interpret this as
    # meaning that this node and its nearest neighbor appear together in many
    # of the graphs being averaged, so this is potentially a good
    # representation of some edge that really appears in many of the graphs.
    confidence = np.min(dmat_avg_uncapped, axis=0)
    d = squareform(dmat_avg_capped)
    G = knn_graph(d, k)
    d = np.multiply(d, G)
    # Get shortest path tree

    spt = dijkstra(d, directed=False, indices=0, return_predecessors=True)
    # Get graph representation by only keeping distances on edges from spt
    mask = np.array([True] * (d.shape[0] * d.shape[1])).reshape(d.shape)
    for i in range(1, len(spt[1])):
        if spt[1][i] == -9999:
            print("Disconnected", i)
            continue
        mask[i, spt[1][i]] = False
        mask[spt[1][i], i] = False
    retmat = squareform(dmat_avg_capped)
    retmat[mask] = 0
    return retmat, confidence


def cell_iterator_csv(
    intracell_csv_loc: str,
    as_squareform=True,
) -> Iterator[tuple[str, DistanceMatrix]]:
    """
    :param intracell_csv_loc: A full file path to a csv file.
    :param as_squareform: If True, return a square distance matrix.
        If False, return a vectorform distance matrix.

    :return: an iterator over cells in the csv file, given as tuples of the form
        (name, dmat).
    """
    icdm_csv_validate(intracell_csv_loc)
    with open(intracell_csv_loc, "r", newline="") as icdm_csvfile:
        csv_reader = csv.reader(icdm_csvfile, delimiter=",")
        # Assume a header
        next(csv_reader)
        while ell := next(csv_reader, None):
            cell_name = ell[0]
            arr = np.array([float(x) for x in ell[1:]], dtype=np.float64)
            if as_squareform:
                arr = squareform(arr, force="tomatrix")
            yield cell_name, arr


def icdm_csv_validate(intracell_csv_loc: str) -> None:
    """
    Raise an exception if the file in intracell_csv_loc fails to pass formatting tests.

    If formatting tests are passed, the function returns none.

    :param intracell_csv_loc: The (full) file path for the CSV file containing the intracell
        distance matrix.

    The file format for an intracell distance matrix is as follows:

    * A line whose first character is '#' is discarded as a comment.
    * The first line which is not a comment is discarded as a "header" - this line may
          contain the column titles for each of the columns.
    * Values separated by commas. Whitespace is not a separator.
    * The first value in the first non-comment line should be the string 'cell_id', and
          all values in the first column after that should be a unique identifier for that cell.
    * All values after the first column should be floats.
    * Not including the cell id in the first column, each row except the header should contain
          the entries of an intracell distance matrix lying strictly above the diagonal,
          as in the footnotes of
          https://docs.scipy.org/doc/scipy/reference/\
          generated/scipy.spatial.distance.squareform.html
    """
    with open(intracell_csv_loc, "r", newline="") as icdm_infile:
        csv_reader = csv.reader(icdm_infile, delimiter=",")
        header = next(csv_reader)
        while header[0] == "#":
            header = next(csv_reader)
        if header[0] != "cell_id":
            raise ValueError("Expects header on first line starting with 'cell_id' ")
        linenum = 1
        for line in csv_reader:
            if line[0] == "#":
                continue
            for value in line[1:]:
                try:
                    float(value)
                except ValueError:
                    print(
                        "Unexpected value at file line "
                        + str(linenum)
                        + ", could not convert value"
                        + str(value)
                        + " to a float"
                    )
                    raise

            line_length = len(header[1:])
            side_length = ceil(sqrt(2 * line_length))
            if side_length * (side_length - 1) != 2 * line_length:
                raise ValueError(
                    "Line " + str(linenum) + " is not in upper triangular form."
                )
            linenum += 1


def uniform(n: int) -> npt.NDArray[np.float64]:
    """Compute the uniform distribution on n points, as a vector of floats."""
    return np.ones((n,), dtype=float) / n


def n_c_2(n: int):
    """Compute the number of ordered pairs of distinct elements in the set {1,...,n}."""
    return (n * (n - 1)) // 2
