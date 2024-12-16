"""
This file contains functionality related to applying Fused Gromov-Wasserstein
to SWC morphology reconstructions of neurons, the functions are specialized
for this purpose. For other applications such as images with multiple color channels,
see sample_seg.py.
"""

import numpy as np
import numpy.typing as npt
import itertools as it
from threadpoolctl import ThreadpoolController
from typing import Optional, Any
import csv
from scipy.spatial.distance import squareform
import math
import ot
from tqdm import tqdm
from multiprocessing import Pool

from .cajal_types import DistanceMatrix, Distribution, MetricMeasureSpace
from .utilities import uniform, cell_iterator_csv, n_c_2
from .gw_cython import gw


def to_penalty_matrix(
    penalty_dictionary: dict[tuple[int, int], float], cell1_node_types, cell2_node_types
):
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
    return penalty_matrix


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

    penalty_matrix = to_penalty_matrix(
        penalty_dictionary, cell1_node_types, cell2_node_types
    )

    if worst_case_gw_increase is not None:
        (plan, log) = ot.gromov.gromov_wasserstein(
            cell1_dmat,
            cell2_dmat,
            p=cell1_distribution,
            q=cell2_distribution,
            symmetric=True,
            log=True,
        )
        G0 = plan
        denom = (penalty_matrix * plan).sum()
        if denom <= 0:
            penalty_matrix = penalty_matrix
        else:
            scalar = worst_case_gw_increase * log["gw_dist"] / denom
            penalty_matrix *= scalar
    else:
        G0 = cell1_distribution[:, np.newaxis] * cell2_distribution[np.newaxis, :]

    return ot.fused_gromov_wasserstein(
        penalty_matrix,
        cell1_dmat,
        cell2_dmat,
        p=cell1_distribution,
        q=cell2_distribution,
        G0=G0,
        **kwargs,
    )


# _CELLS: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]
# _NODE_TYPES: npt.NDArray[np.int32]
# _PENALTY_DICTIONARY: dict[tuple[int, int], float]
# _WORST_CASE_GW_INCREASE: Optional[float]
# _KWARGS: dict


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
    _CELLS: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = cells  # type: ignore[name-defined]
    global _NODE_TYPES
    _NODE_TYPES: npt.NDArray[np.int32] = node_types  # type: ignore[name-defined]
    global _WORST_CASE_GW_INCREASE
    _WORST_CASE_GW_INCREASE: Optional[float] = worst_case_gw_increase  # type: ignore[name-defined]
    global _KWARGS
    _KWARGS: dict[str, Any] = kwargs  # type: ignore[name-defined]
    global _PENALTY_DICTIONARY
    _PENALTY_DICTIONARY: dict[tuple[int, int], float] = penalty_dictionary  # type: ignore[name-defined]


def _fgw_index(p: tuple[int, int]):
    """Compute the Fused GW distance between cells i and j in the master cell list."""
    i, j = p
    (_, log) = fused_gromov_wasserstein(
        _CELLS[i][0],  # type: ignore[name-defined]
        _CELLS[i][1],  # type: ignore[name-defined]
        _NODE_TYPES[i],  # type: ignore[name-defined]
        _CELLS[j][0],  # type: ignore[name-defined]
        _CELLS[j][1],  # type: ignore[name-defined]
        _NODE_TYPES[j],  # type: ignore[name-defined]
        _PENALTY_DICTIONARY,  # type: ignore[name-defined]
        _WORST_CASE_GW_INCREASE,  # type: ignore[name-defined]
        **_KWARGS,  # type: ignore
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


def _load(
    intracell_csv_loc: str, swc_node_types: str, sample_points_npz: Optional[str]
) -> tuple[list[MetricMeasureSpace], npt.NDArray[np.int32], list[str]]:
    """
    :returns: A list of cells, a 2d- array of their node types, and a list of their names.
    """
    if sample_points_npz is None:
        cell_names_dmats = list(cell_iterator_csv(intracell_csv_loc))
        node_types = np.load(swc_node_types)
        cells = [(c := cell, uniform(c.shape[0])) for _, cell in cell_names_dmats]
        names = [name for name, _ in cell_names_dmats]
    else:
        a = np.load(sample_points_npz)
        cell_dmats = a["dmats"]
        k = cell_dmats.shape[1]
        n = int(math.ceil(math.sqrt(k * 2)))
        u = uniform(n)
        cells = [(squareform(cell, force="tosquareform"), u) for cell in cell_dmats]
        node_types = a["structure_ids"]
        names = a["names"].tolist()
        a.close()

    return (cells, node_types, names)


def fused_gromov_wasserstein_estimate_costs(
    cells,
    swc_node_types: list[npt.NDArray[np.int64]],
    sample_size: int,
    penalty_dictionary: dict[tuple[int, int], float],
):
    n = len(cells)
    n_pairs = n_c_2(n)
    k = int(n_pairs / sample_size)
    to_sample = it.islice(it.combinations(zip(cells, swc_node_types), 2), 0, None, k)
    ell = []
    # u = uniform(node_types.shape[1])
    for ((dmat1, distr1), nt1), ((dmat2, distr2), nt2) in to_sample:
        coupling_mat, gw_dist = gw(dmat1, distr1, dmat2, distr2)
        penalty_matrix = to_penalty_matrix(penalty_dictionary, nt1, nt2)
        ell.append(
            float((penalty_matrix * coupling_mat).sum()) / (4 * gw_dist * gw_dist)
        )
    return np.median(np.array(ell))


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
    dynamically_adjust: bool = False,
    sample_size: int = 100,
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
    node_types: list[npt.NDArray[np.int32]]
    names: list[str]
    cells, node_types, names = _load(
        intracell_csv_loc, swc_node_types, sample_points_npz
    )
    num_cells = len(names)
    index_pairs = it.combinations(
        iter(range(num_cells)), 2
    )  # object pairs to compute fGW / OT for
    total_num_pairs = n_c_2(num_cells)
    kwargs["log"] = True
    if penalty_dictionary is None:
        penalty_dictionary = dict()
        penalty_dictionary[(1, 3)] = soma_dendrite_penalty
        penalty_dictionary[(1, 4)] = soma_dendrite_penalty
        penalty_dictionary[(3, 4)] = basal_apical_penalty
    if (not dynamically_adjust) and (worst_case_gw_increase is not None):
        estimate = fused_gromov_wasserstein_estimate_costs(
            cells, node_types, sample_size, penalty_dictionary  # type: ignore
        )
        if estimate == 0:
            raise Exception(
                "Most sampled transport plans have zero node penalty, \
                            add node penalties for more nodes or set worst_case_gw_increase to None"
            )
        for key in penalty_dictionary:
            penalty_dictionary[key] = (
                penalty_dictionary[key] * worst_case_gw_increase / estimate
            )

        worst_case_gw_increase = None
        # A bit confusing but this controls the *dynamic* estimation
        # of weights, and we don't want to estimate them dynamically.

    with Pool(
        initializer=_init_fgw_pool,
        initargs=(
            cells,
            node_types,
            penalty_dictionary,
            worst_case_gw_increase,
            kwargs,
        ),
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
