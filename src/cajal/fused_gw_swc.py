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


def _to_penalty_matrix(
    penalty_dictionary: dict[tuple[int, int], float], cell1_node_types, cell2_node_types
):
    """
    Convert the given penalty dictionary into a cost matrix for fused Gromov-Wasserstein.
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

    :param cell1_dmat: A squareform distance matrix of shape (n,n).
    :param cell1_distribution: A probability distribution of shape (n).
    :param cell1_node_types: A vector of integer structure id's (type labels) of shape (n)
    :param cell2_dmat: A squareform distance matrix of shape (m,m).
    :param cell2_distribution: A probability distribution of shape (m).
    :param cell2_node_types: A vector of integer structure id's (type labels) of shape (m)
    :param penalty_dictionary: A dictionary whose keys are pairs (i,j) of distinct
        structure ids
        for points occurring in the sample data (with i < j) and whose values are non-negative
        floating point
        numbers, representing the "fused" penalty for aligning a node of type i with a node of
        type j. Pairs (i,j) which aren't in the penalty dictionary have penalty weight 0.
    :param worst_case_gw_increase: This parameter is meant to give a more interpretable and
        intuitively accessible way to control the fused GW penalty in situations where it is
        difficult to assess the appropriate order of magnitude for the values in the penalty matrix
        *a priori*. The notion of fused GW involves a compromise between minimizing distortion
        (GW cost) and minimizing label misalignment (conflicts in node type labels).
        Adding a penalty for label misalignment will tend to increase the distortion of the
        transport plan, because the algorithm now has to balance both of these considerations,
        and the higher the penalty for label misalignment, the higher the distortion of the
        associated transport plan will be, because the algorithm will focus primarily on aligning
        node types. Therefore, we offer a way to control the maximum increase in distortion
        (above and beyond the distortion associated to the ordinary, classical GW transport plan)
        due to the additional constraint of ensuring label alignment.

        If worst_case_gw_increase is None (the default)
        then the values in penalty_dictionary are taken as *absolute* penalties, and the fused GW
        cost matrix is directly computed from the weights supplied. If worst_case_gw_increase is a
        non-negative floating point number then the values in penalty_dictionary are interpreted in
        a *relative* way, so that only the ratios of one value to another become meaningful - for
        example, if penalty_dictionary[(1,3)] = 10.0 and penalty_dictionary[(3,4)] = 2.0, then the
        resulting fused GW cost matrix will have the property that aligning
        a soma node with a basal dendrite is 5 times more costly than aligning a basal dendrite node
        with an apical dendrite node.
        The *absolute* values of the fused GW cost matrix are determined by the following heuristic,
        which guarantees that
        the GW cost of the final matrix for the fused GW cost is at most a factor of
        `worst_case_gw_increase` greater than
        that of the ordinary GW distance. For instance, if the user supplies
        `worst_case_gw_increase = 0.50`, then the transport plan found by the fused GW algorithm is
        guaranteed to have a GW cost at most 50% higher than the
        transport plan found by ordinary GW.
    :param kwargs: This function wraps the implementation of Fused GW provided by the
        Python Optimal Transport library and all additional keyword arguments supplied by the user
        are passed to that function. See the documentation
        `here <https://pythonot.github.io/all.html#ot.fused_gromov_wasserstein>`_ for keyword
        arguments which can be used to customize the behavior of the algorithm.
    """

    penalty_matrix = _to_penalty_matrix(
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
        if denom > 0:
            penalty_matrix *= (worst_case_gw_increase * log["gw_dist"] / denom)
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
    _CELLS = cells  # type: ignore[name-defined]
    global _NODE_TYPES
    _NODE_TYPES = node_types  # type: ignore[name-defined]
    global _WORST_CASE_GW_INCREASE
    _WORST_CASE_GW_INCREASE = worst_case_gw_increase  # type: ignore[name-defined]
    global _KWARGS
    _KWARGS = kwargs  # type: ignore[name-defined]
    global _PENALTY_DICTIONARY
    _PENALTY_DICTIONARY = penalty_dictionary  # type: ignore[name-defined]


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


def _fused_gromov_wasserstein_estimate_costs(
    cells,
    swc_node_types: list[npt.NDArray[np.int64]],
    sample_size: int,
    penalty_dictionary: dict[tuple[int, int], float],
    quantile: float
):
    n = len(cells)
    n_pairs = n_c_2(n)
    k = int(n_pairs / sample_size)
    to_sample = it.islice(it.combinations(zip(cells, swc_node_types), 2), 0, None, k)
    ell = []
    # u = uniform(node_types.shape[1])
    for ((dmat1, distr1), nt1), ((dmat2, distr2), nt2) in to_sample:
        coupling_mat, gw_dist = gw(dmat1, distr1, dmat2, distr2)
        penalty_matrix = _to_penalty_matrix(penalty_dictionary, nt1, nt2)
        ell.append(
            (gw_dist, float((penalty_matrix * coupling_mat).sum()) / (4 * gw_dist * gw_dist))
        )
    a, b = zip(*ell)
    a = np.array(a)
    b = np.array(b)
    return np.quantile(b, quantile)

    # return np.max(b[a <= np.quantile(a, quantile)])


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
    quantile: float = 0.15,
    **kwargs,
):
    """
    Compute the fused GW distance pairwise in parallel between many neurons.

    :param intracell_csv_loc: The path to the file where the sampled points are stored.
    :param swc_node_types: The path to the swc node type file, expected to be in npy format;
        consistent with the files written by functions in the sample_swc module.
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
    :param penalty_dictionary: For the meaning of this parameter, see
        the documentation for :func:`cajal.fused_gw_swc.fused_gromov_wasserstein`.
        If penalty_dictionary is None, it is automatically
        constructed as a function of the arguments `soma_dendrite_penalty` and
        `apical_dendrite_penalty`. If this parameter is supplied then
        the previous two parameters are ignored as this parameter overrides them;
        the user can reproduce the behavior by adding penalty keys for (1,3), (1,4)
        and (3,4) appropriately. The
    :param chunksize: A parallelization parameter, the
        number of jobs fed to each process at a time.
    :param worst_case_gw_increase: The meaning of this parameter is closely
        related to the parameter documented in
        :func:`cajal.fused_gw_swc.fused_gromov_wasserstein`, but see
        the documentation for `dynamically_adjust`.
    :param dynamically_adjust: If `dynamically_adjust` is True, then
        the argument `worst_case_gw_increase` is passed directly to
        the function :func:`cajal.fused_gw_swc.fused_gromov_wasserstein` for
        each pair of arguments. However, this would mean that a different
        cost matrix will be computed for each pair of cells, so one is not
        computing the same notion of fused GW throughout the data. We regard it
        as more statistically appropriate to use the same fixed parameters
        for fused GW throughout all cell pairs in the data. If `dynamically_adjust` is False
        (the default) then the effect of `worst_case_gw_increase` is to set a global
        cost matrix for all cell pairs, chosen such that for a pair of cells
        in the same neighborhood of the GW space, the increase in distortion will be at most
        a factor of `worst_case_gw_increase`. (This is a statistical heuristic,
        this is not guaranteed.)
    :param sample_size: Only relevant if `dynamically_adjust` is True. Indicates
        the number of cell pairs to sample in order to estimate the distribution of GW costs.
    :param quantile: Only relevant if `dynamically_adjust` is True.
        This informs the notion of "in the same neighborhood" described in `dynamically_adjust`.
        The cost matrix is constructed by looking at cells whose GW cost is less than the
        given quantile in the sample distribution.
    :param kwargs: See documentation for :func:`cajal.fused_gw_swc.fused_gromov_wasserstein`
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
        estimate = _fused_gromov_wasserstein_estimate_costs(
            cells, node_types, sample_size, penalty_dictionary, quantile  # type: ignore
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
            print(penalty_dictionary[key])
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
