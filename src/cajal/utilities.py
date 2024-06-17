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

from scipy.sparse import coo_matrix
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


def read_gw_couplings(
    gw_couplings_file_loc: str, header: bool
) -> dict[tuple[str, str], npt.NDArray[np.float64]]:
    """
    Read a list of Gromov-Wasserstein coupling matrices into memory.
    :param header: If True, the first line of the file will be ignored.
    :param gw_couplings_file_loc: name of a file holding a list of GW coupling matrices in \
    COO form. The files should be in csv format. Each line should be of the form
    `cellA_name, cellA_sidelength, cellB_name, cellB_sidelength, num_nonzero, (data), (row), (col)`
    where `data` is a sequence of `num_nonzero` many floating point real numbers,
    `row` is a sequence of `num_nonzero` many integers (row indices), and
    `col` is a sequence of `num_nonzero` many integers (column indices).
    :return: A dictionary mapping pairs of names (firstcell, secondcell) to the GW \
    matrix of the coupling. `firstcell` and `secondcell` are in alphabetical order.
    """

    gw_coupling_mat_dict: dict[tuple[str, str], npt.NDArray[np.float64]] = {}
    with open(gw_couplings_file_loc, "r", newline="") as gw_file:
        csvreader = csv.reader(gw_file, delimiter=",")
        linenum = 1
        if header:
            _ = next(csvreader)
            linenum += 1
        for line in csvreader:
            cellA_name = line[0]
            cellA_sidelength = int(line[1])
            cellB_name = line[2]
            cellB_sidelength = int(line[3])
            num_non_zero = int(line[4])
            rest = line[5:]
            if 3 * num_non_zero != len(rest):
                raise Exception(
                    "On line " + str(linenum) + " data not in COO matrix form."
                )
            data = [float(x) for x in rest[:num_non_zero]]
            rows = [int(x) for x in rest[num_non_zero : (2 * num_non_zero)]]
            cols = [int(x) for x in rest[(2 * num_non_zero) :]]
            coo = coo_array(
                (data, (rows, cols)), shape=(cellA_sidelength, cellB_sidelength)
            )
            linenum += 1
            if cellA_name < cellB_name:
                gw_coupling_mat_dict[(cellA_name, cellB_name)] = coo
            else:
                gw_coupling_mat_dict[(cellB_name, cellA_name)] = coo_array.transpose(
                    coo
                )
    return gw_coupling_mat_dict


T = TypeVar("T")


@dataclass
class Err(Generic[T]):
    code: T


def write_csv_block(
    out_csv: str,
    sidelength: int,
    dist_mats: Iterator[tuple[str, Union[Err[T], npt.NDArray[np.float64]]]],
    batch_size: int,
) -> list[tuple[str, Err[T]]]:
    """
    :param sidelength: The side length of all matrices in dist_mats.
    :param dist_mats: an iterator over pairs (name, arr), where arr is an
    vector-form array (rank 1) or an error code.
    """
    failed_cells: list[tuple[str, Err[T]]] = []
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
                else:
                    good_cells.append([name] + cell.tolist())
            csvwriter.writerows(good_cells)
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
    louvain_clus_dict = community_louvain.best_partition(graph)
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
    return np.min(icdm)


def orient(
    medoid: str,
    obj_name: str,
    iodm: npt.NDArray[np.float64],
    gw_coupling_mat_dict: dict[tuple[str, str], coo_matrix],
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
        gw_coupling_mat = coo_matrix.transpose(gw_coupling_mat_dict[(medoid, obj_name)])

    i_reorder = np.argmax(gw_coupling_mat.todense(), axis=0)
    return iodm[i_reorder][:, i_reorder]


def avg_shape(
    obj_names: list[str],
    gw_dist_dict: dict[tuple[str, str], float],
    iodms: dict[str, npt.NDArray[np.float64]],
    gw_coupling_mat_dict: dict[tuple[str, str], coo_matrix],
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
    obj_names: list[str],
    gw_dist_dict: dict[tuple[str, str], float],
    iodms: dict[str, npt.NDArray[np.float64]],
    gw_coupling_mat_dict: dict[tuple[str, str], coo_matrix],
    k: int,
):
    """
    Given a set of cells together with their intracell distance matrices and
    the (precomputed) pairwise GW coupling matrices between cells, construct a
    morphological "average" of cells in the cluster. This function:

    * aligns all cells in the cluster with each other using the coupling matrices
    * takes a "local average" of all intracell distance matrices, forming a
      distance matrix which models the average local connectivity structure of the neurons
    * draws a minimum spanning tree through the intracell distance graph,
      allowing us to visualize this average morphology

    :param obj_names: Keys for the gw_dist_dict and iodms; unique identifiers for the cells.
    :param gw_dist_dict: Dictionary mapping ordered pairs (cellA_name, cellB_name) \
        to Gromov-Wasserstein distances between them.
    :param iodms: (intra-object distance matrices) - \
        Maps object names to intra-object distance matrices. Matrices are assumed to be given \
        in vector form rather than squareform.
    :gw_coupling_mat_dict: Dictionary mapping ordered pairs (cellA_name, cellB_name) to \
        Gromov-Wasserstein coupling matrices from cellA to cellB.
    :param k: how many neighbors in the nearest-neighbors graph.
    """
    dmat_avg_capped, dmat_avg_uncapped = avg_shape(
        obj_names, gw_dist_dict, iodms, gw_coupling_mat_dict
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
