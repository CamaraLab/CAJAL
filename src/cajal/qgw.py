# std lib dependencies
import itertools as it
import time
import csv
from typing import Iterable, Iterator, Collection
from math import sqrt

import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# external dependencies
import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy import cluster  # noqa: E402

# from scipy.sparse import coo_array  # noqa: E402
from multiprocessing import Pool  # noqa: E402

from .slb import slb2 as slb_cython  # noqa: E402
from .gw_cython import (  # noqa: E402
    frobenius,
    quantized_gw_cython,
)
from .run_gw import _batched, cell_iterator_csv  # noqa: E402


# SLB
def _init_slb_pool(sorted_cells):
    """
    Initialize the parallel SLB computation by declaring a global variable
    accessible from all processes.
    """

    global _SORTED_CELLS
    _SORTED_CELLS = sorted_cells


def _global_slb_pool(p: tuple[int, int]):
    """
    Given input p= (i,j), compute the SLB distance between cells i
    and j in the global list of cells.
    """

    i, j = p
    return (i, j, slb_cython(_SORTED_CELLS[i], _SORTED_CELLS[j]))


def slb_parallel_memory(
    cell_dms: Collection[npt.NDArray[np.float_]],
    num_processes: int,
    chunksize: int = 20,
) -> npt.NDArray[np.float_]:
    """
    Compute the SLB distance in parallel between all cells in `cell_dms`.
    :param cell_dms: A collection of distance matrices
    :param num_processes: How many Python processes to run in parallel
    :param chunksize: How many SLB distances each Python process computes at a time
    """
    cell_dms_sorted = [np.sort(squareform(cell, force="tovector")) for cell in cell_dms]
    N = len(cell_dms)
    with Pool(
        initializer=_init_slb_pool,
        initargs=(cell_dms_sorted,),
        processes=num_processes,
    ) as pool:
        slb_dists = pool.imap_unordered(
            _global_slb_pool, it.combinations(iter(range(N)), 2), chunksize=chunksize
        )
        arr = np.zeros((N, N))
        for i, j, x in slb_dists:
            arr[i, j] = x
            arr[j, i] = x

    return arr


def slb_parallel(
    intracell_csv_loc: str,
    num_processes: int,
    out_csv: str,
    chunksize: int = 20,
) -> None:
    """
    Compute the SLB distance in parallel between all cells in the csv file `intracell_csv_loc`.
    The files are expected to be formatted according to the format in
    :func:`cajal.run_gw.icdm_csv_validate`.

    :param cell_dms: A collection of distance matrices
    :param num_processes: How many Python processes to run in parallel
    :param chunksize: How many SLB distances each Python process computes at a time
    """
    names, cell_dms = zip(*cell_iterator_csv(intracell_csv_loc))
    slb_dmat = slb_parallel_memory(cell_dms, num_processes, chunksize)
    ij = it.combinations(range(len(names)), 2)
    with open(out_csv, "w", newline="") as outfile:
        csv_writer = csv.writer(outfile)
        csv_writer.writerow(["first_object", "second_object", "slb_dist"])
        batches = _batched(
            ((names[i], names[j], str(slb_dmat[i, j])) for i, j in ij), 2000
        )
        for batch in batches:
            csv_writer.writerows(batch)


class quantized_icdm:
    """
    This class represents a "quantized" intracell distance matrix, i.e.,
    a metric measure space which has been equipped with a given clustering;
    it contains additional data which allows for the rapid computation
    of pairwise GW distances across many cells.

    Users should only need to understand how to call the main constructor.
    """

    n: int
    # 2 dimensional square matrix of side length n.
    icdm: npt.NDArray[np.float64]
    # "distribution" is a dimensional vector of length n,
    # a probability distribution on points of the space
    distribution: npt.NDArray[np.float64]
    ns: int
    # A square sub-matrix of icdm, the distance matrix between sampled points. Of side length ns.
    sub_icdm: npt.NDArray[np.float64]
    # q_indices is a 1-dimensional array of integers of length ns+1. For i,j < ns,
    # icdm[sample_indices[i],sample_indices[j]]==sub_icdm[i,j].
    # sample_indices[ns]==n.
    q_indices: npt.NDArray[np.int_]
    # The quantized distribution; a 1-dimensional array of length ns.
    q_distribution: npt.NDArray[np.float64]
    c_A: float
    c_As: float
    A_s_a_s: npt.NDArray[np.float64]
    # This field is equal to np.dot(np.dot(np.multiply(icdm,icdm),distribution),distribution)

    def __init__(
        self,
        cell_dm: npt.NDArray[np.float64],
        p: npt.NDArray[np.float64],
        num_clusters: int,
    ):
        """
        :param cell_dm: An intracell distance matrix in squareform.
        :param p: A probability distribution on the points of the metric space
        :param num_clusters: How many clusters to subdivide the cell into; the more clusters,
        the more accuracy, but the longer the computation.
        """
        assert len(cell_dm.shape) == 2
        self.n = cell_dm.shape[0]
        cell_dm_sq = np.multiply(cell_dm, cell_dm)
        self.c_A = np.dot(np.dot(cell_dm_sq, p), p)
        Z = cluster.hierarchy.linkage(squareform(cell_dm), method="centroid")
        clusters = cluster.hierarchy.fcluster(
            Z, num_clusters, criterion="maxclust", depth=0
        )
        actual_num_clusters: int = len(set(clusters))
        self.ns = actual_num_clusters
        indices: npt.NDArray[np.int_] = np.argsort(clusters)
        original_cell_dm = cell_dm
        cell_dm = cell_dm[indices, :][:, indices]
        p = p[indices]
        q: list[float]
        q = []
        clusters = np.sort(clusters)
        for i in range(1, actual_num_clusters + 1):
            permutation = np.nonzero(clusters == i)[0]
            this_cluster = cell_dm[permutation, :][:, permutation]
            medoid = np.argmin(sum(this_cluster))
            new_local_indices = np.argsort(this_cluster[medoid])
            cell_dm[permutation, :] = cell_dm[permutation[new_local_indices], :]
            cell_dm[:, permutation] = cell_dm[:, permutation[new_local_indices]]
            indices[permutation] = indices[permutation[new_local_indices]]
            p[permutation] = p[permutation[new_local_indices]]
            q.append(np.sum(p[permutation]))
        self.icdm = np.asarray(cell_dm, order="C")
        self.distribution = p
        q_arr = np.array(q, dtype=np.float64, order="C")
        self.q_distribution = q_arr
        assert abs(np.sum(q_arr) - 1.0) < 1e-7
        medoids = np.nonzero(np.r_[1, np.diff(clusters)])[0]
        A_s = cell_dm[medoids, :][:, medoids]
        assert np.all(np.equal(original_cell_dm[:, indices][indices, :], cell_dm))
        self.sub_icdm = np.asarray(A_s, order="C")
        self.q_indices = np.asarray(
            np.nonzero(np.r_[1, np.diff(clusters), 1])[0], order="C"
        )
        self.c_As = np.dot(np.multiply(A_s, A_s), q_arr) @ q_arr
        self.A_s_a_s = np.dot(A_s, q_arr)


def quantized_gw(A: quantized_icdm, B: quantized_icdm):
    """
    Compute the quantized Gromov-Wasserstein distance between two quantized metric measure spaces.
    """
    T_rows, T_cols, T_data = quantized_gw_cython(
        A.distribution,
        A.sub_icdm,
        A.q_indices,
        A.q_distribution,
        A.A_s_a_s,
        A.c_As,
        B.distribution,
        B.sub_icdm,
        B.q_indices,
        B.q_distribution,
        B.A_s_a_s,
        B.c_As,
    )

    P = sparse.coo_matrix((T_data, (T_rows, T_cols)), shape=(A.n, B.n)).tocsr()
    gw_loss = A.c_A + B.c_A - 2.0 * frobenius(A.icdm, P.dot(P.dot(B.icdm).T))
    return sqrt(gw_loss) / 2.0


def _block_quantized_gw(indices):
    # Assumes that the global variable _QUANTIZED_CELLS has been declared, as by
    # init_qgw_pool
    (i0, i1), (j0, j1) = indices

    gw_list = []
    for i in range(i0, i1):
        A = _QUANTIZED_CELLS[i]
        for j in range(j0, j1):
            if i < j:
                B = _QUANTIZED_CELLS[j]
                gw_list.append((i, j, quantized_gw(A, B)))
    return gw_list


def _init_qgw_pool(quantized_cells: list[quantized_icdm]):
    """
    Initialize the parallel quantized GW computation by declaring a global variable
    accessible from all processes.
    """
    global _QUANTIZED_CELLS
    _QUANTIZED_CELLS = quantized_cells


def _quantized_gw_index(p: tuple[int, int]):
    """
    Given input p= (i,j), compute the quantized GW distance between cells i
    and j in the global list of quantized cells.
    """
    i, j = p
    return (i, j, quantized_gw(_QUANTIZED_CELLS[i], _QUANTIZED_CELLS[j]))


def quantized_gw_parallel(
    intracell_csv_loc: str,
    num_processes: int,
    num_clusters: int,
    out_csv: str,
    chunksize: int = 20,
    verbose: bool = False,
) -> None:
    """
    Compute the quantized Gromov-Wasserstein distance in parallel between all cells in a family
    of cells.
    :param intracell_csv_loc: path to a CSV file containing the cells to process
    :param num_processes: number of Python processes to run in parallel
    :param num_clusters: Each cell will be partitioned into `num_clusters` many clusters.
    :out_csv: file path where a CSV file containing the quantized GW distances will be written
    :chunksize: How many q-GW distances should be computed at a time by each parallel process.
    """
    names, cell_dms = zip(*cell_iterator_csv(intracell_csv_loc))
    quantized_cells = [
        quantized_icdm(
            cell_dm, np.ones((cell_dm.shape[0],)) / cell_dm.shape[0], num_clusters
        )
        for cell_dm in cell_dms
    ]
    N = len(quantized_cells)
    index_pairs = it.combinations(iter(range(N)), 2)

    gw_time = 0.0
    fileio_time = 0.0
    gw_start = time.time()
    with Pool(
        initializer=_init_qgw_pool, initargs=(quantized_cells,), processes=num_processes
    ) as pool:
        gw_dists = pool.imap_unordered(
            _quantized_gw_index, index_pairs, chunksize=chunksize
        )
        gw_stop = time.time()
        gw_time += gw_stop - gw_start
        with open(out_csv, "w", newline="") as outcsvfile:
            csvwriter = csv.writer(outcsvfile)
            csvwriter.writerow(["first_object", "second_object", "quantized_gw"])
            gw_start = time.time()
            t = _batched(gw_dists, 2000)
            for block in t:
                block = [(names[i], names[j], gw_dist) for (i, j, gw_dist) in block]
                gw_stop = time.time()
                gw_time += gw_stop - gw_start
                csvwriter.writerows(block)
                gw_start = time.time()
                fileio_time += gw_start - gw_stop
    return


def _cutoff_of(
    slb_dmat: npt.NDArray[np.float_],
    median: float,
    gw_dmat: npt.NDArray[np.float_],
    gw_known: npt.NDArray[np.bool_],
    nn: int,
) -> npt.NDArray[np.float_]:
    # maxval = np.max(gw_dmat)
    gw_copy = np.copy(gw_dmat)
    # gw_copy[~gw_known]=maxval
    gw_copy[~gw_known] = (slb_dmat + median)[~gw_known]
    gw_copy.partition(nn + 1, axis=1)
    return gw_copy[:, nn + 1]


def _tuple_iterator_of(
    X: npt.NDArray[np.int_], Y: npt.NDArray[np.int_]
) -> Iterator[tuple[int, int]]:
    b = set()
    for i, j in map(tuple, np.stack((X, Y), axis=1, dtype=int).astype(int)):
        if i < j:
            b.add((i, j))
        else:
            b.add((j, i))
    return iter(b)


def _get_indices(
    slb_dmat: npt.NDArray[np.float_],
    gw_dmat: npt.NDArray[np.float_],
    gw_known: npt.NDArray[np.bool_],
    accuracy: float,
    nearest_neighbors: int,
) -> list[tuple[int, int]]:
    """
    Based on the SLB distance matrix and the partially known GW distance matrix,
    and the desired accuracy, return a list of cell pairs which we should compute.
    This function does not return *all* cell pairs that must be computed for the desired accuracy;
    it is expected that the function will be called *repeatedly*, and that a new list of
    cell pairs will be given every time, roughly in descending order of priority;
    when the empty list is returned, this indicates that the gw distance
    table is already at the desired accuracy, and the loop should terminate.

    :param slb_dmat: the SLB distance matrix in squareform, but this would make sense for
    \any lower bound for gw_dmat
    :param gw_dmat: A partially defined
    Gromov-Wasserstein distance matrix in squareform, we should have
     gw_dmat >= slb_dmat almost everywhere where gw_known is true for this to make sense
    (I have not yet checked how this behaves in the case where some values of slb_dmat
    are greater than gw_dmat); should be zero elsewhere.
    :param gw_known: A matrix of Booleans which is true where the entries of `gw_dmat` are
    correct/valid and false where the entries are not meaningful/do not yet have the
    correct value
    :param accuracy: This is a real number between 0 and 1 inclusive. If the accuracy is 1,
    then pairwise cells will continue to be computed until all remaining uncomputed
    cell pairs have an SLB distance which is strictly higher than anything on the list of \
    `nearest_neighbors` many nearest neighbors of every point; thus the reported array of
    distances is guaranteed to be correct out to the first `nearest_neighbors` nearest neighbors
    of every point.
    """
    gw_vf = squareform(gw_dmat)
    N = gw_dmat.shape[0]
    bins = 200
    if np.all(gw_vf == 0.0):
        ind_y = np.argsort(slb_dmat, axis=1)[:, 1 : nearest_neighbors + 1]
        ind_x = np.broadcast_to(np.arange(N)[:, np.newaxis], (N, nearest_neighbors))
        xy = np.reshape(np.stack((ind_x, ind_y), axis=2, dtype=int), (-1, 2))
        return list(_tuple_iterator_of(xy[:, 0], xy[:, 1]))

    # Otherwise, we assume that at least the initial values have been computed.
    slb_vf = squareform(slb_dmat)

    errors = (gw_vf - slb_vf)[gw_vf > 0]
    error_quantiles = np.quantile(
        errors, np.arange(bins + 1).astype(float) / float(bins)
    )
    median = error_quantiles[int(bins / 2)]
    # cutoff = _cutoff_of(gw_dmat,gw_known,nearest_neighbors)
    cutoff = _cutoff_of(slb_dmat, median, gw_dmat, gw_known, nearest_neighbors)

    acceptable_injuries = (nearest_neighbors * N) * (1 - accuracy)
    # We want the expectation of injury to be below this.
    candidates = (~gw_known) & (slb_dmat <= cutoff[:, np.newaxis])
    X, Y = np.nonzero(candidates)
    candidate_count = X.shape[0]
    threshold = cutoff[X] - slb_dmat[X, Y]
    assert np.all(threshold >= 0)
    index_sort = np.argsort(threshold)
    quantiles = np.digitize(threshold, error_quantiles).astype(float) / float(bins)

    sq = np.sort(quantiles)
    K1 = int(np.searchsorted(np.cumsum(sq), acceptable_injuries))
    K2 = int(np.searchsorted(quantiles, 0.5))
    K = min(K1, K2)

    block_size = N * 5

    assert candidate_count == quantiles.shape[0]
    if K == quantiles.shape[0]:
        return []

    if (candidate_count - K) < block_size:
        from_index = K
    else:
        from_index = int((candidate_count + K) / 2)
        # from_index = candidate_count - block_size
        assert from_index >= K
        assert from_index < candidate_count
    indices = index_sort[from_index:]

    return list(_tuple_iterator_of(X[indices], Y[indices]))


def _update_dist_mat(
    gw_dist_iter: Iterable[tuple[int, int, float]],
    dist_mat: npt.NDArray[np.float_],
    dist_mat_known: npt.NDArray[np.bool_],
) -> None:
    """
    Write the values in `gw_dist_iter` to the matrix `dist_mat` and update the matrix
    `dist_mat_known` to reflect these known values.

    :param gw_dist_iter: An iterator over ordered triples (i,j,d) where i, j are array
    indices and d is a float.
    :param dist_mat: A distance matrix. The matrix is modified by this function;
    we set dist_mat[i,j]=d for all (i,j,d) in `gw_dist_iter`; similarly dist_mat[j,i]=d.
    :param dist_mat_known: An array of booleans recording what GW distances are known.
    This matrix is modified by this function.
    """
    for i, j, gw_dist in gw_dist_iter:
        dist_mat[i, j] = gw_dist
        dist_mat[j, i] = gw_dist
        dist_mat_known[i, j] = True
        dist_mat_known[j, i] = True
    return


def combined_slb_quantized_gw_memory(
    cell_dms: Collection[npt.NDArray[np.float_]],  # Squareform
    num_processes: int,
    num_clusters: int,
    accuracy: float,
    nearest_neighbors: int,
    verbose: bool,
    chunksize: int = 20,
):
    """
    Compute the pairwise SLB distances between each pair of cells in `cell_dms`.
    Based on this initial estimate of the distances, compute the quantized GW distance between
    the nearest with `num_clusters` many clusters until the correct nearest-neighbors list is
    obtained for each cell with a high degree of confidence.

    The idea is that for the sake of clustering we can avoid
    computing the precise pairwise distances between cells which are far apart,
    because the clustering will not be sensitive to changes in large
    distances. Thus, we want to compute as precisely as possible the pairwise
    GW distances for (say) the 30 nearest neighbors of each point, and use a
    rough estimation beyond that.

    :param cell_dms: a list or tuple of square distance matrices
    :param num_processes: How many Python processes to run in parallel
    :param num_clusters: Each cell will be partitioned into `num_clusters` many
    clusters for the quantized Gromov-Wasserstein distance computation.
    :param chunksize: Number of pairwise cell distance computations done by
    each Python process at one time.
    :param out_csv: path to a CSV file where the results of the computation will be written
    :accuracy: This is a real number between 0 and 1, inclusive.
    :param nearest_neighbors: The algorithm tries to compute only the
    quantized GW distances between pairs of cells if one is within the first
    `nearest_neighbors` neighbors of the other; for all other values,
    the SLB distance is used to give a rough estimate.
    """

    N = len(cell_dms)
    np_arange_N = np.arange(N)
    slb_dmat = slb_parallel_memory(cell_dms, num_processes, chunksize)

    # Partial quantized Gromov-Wasserstein table, will be filled in gradually.
    qgw_dmat = np.zeros((N, N), dtype=float)
    qgw_known = np.full(shape=(N, N), fill_value=False)
    qgw_known[np_arange_N, np_arange_N] = True

    quantized_cells = [
        quantized_icdm(
            cell_dm, np.ones((cell_dm.shape[0],)) / cell_dm.shape[0], num_clusters
        )
        for cell_dm in cell_dms
    ]
    # Debug
    total_cells_computed = 0
    with Pool(
        initializer=_init_qgw_pool, initargs=(quantized_cells,), processes=num_processes
    ) as pool:
        indices = _get_indices(
            slb_dmat, qgw_dmat, qgw_known, accuracy, nearest_neighbors
        )
        while len(indices) > 0:
            if verbose:
                print(len(indices))
            total_cells_computed += len(indices)
            qgw_dists = pool.imap_unordered(
                _quantized_gw_index, indices, chunksize=chunksize
            )
            _update_dist_mat(qgw_dists, qgw_dmat, qgw_known)
            assert np.count_nonzero(qgw_known) == 2 * total_cells_computed + N
            indices = _get_indices(
                slb_dmat, qgw_dmat, qgw_known, accuracy, nearest_neighbors
            )
    return slb_dmat, qgw_dmat, qgw_known


def combined_slb_quantized_gw(
    input_icdm_csv_location: str,
    gw_out_csv_location: str,
    num_processes: int,
    num_clusters: int,
    accuracy: float,
    nearest_neighbors: int,
    verbose: bool = False,
    chunksize: int = 20,
) -> None:
    """
    This is a wrapper around :func:`cajal.run_gw.combined_slb_quantized_gw_memory` with
    some associated file/IO.

    :param input_icdm_csv_location: file path to a csv file. For format for the icdm
    see :func:`cajal.run_gw.icdm_csv_validate`.
    :param gw_out_csv_location: Where to write the output GW distances.
    :return: None.

    For all other parameters see the docstring for
    :func:`cajal.run_gw.combined_slb_quantized_gw_memory`.
    """

    names, cell_dms = zip(*cell_iterator_csv(input_icdm_csv_location))
    slb_dmat, qgw_dmat, qgw_known = combined_slb_quantized_gw_memory(
        cell_dms,
        num_processes,
        num_clusters,
        accuracy,
        nearest_neighbors,
        verbose,
        chunksize,
    )

    median_error = np.median((qgw_dmat - slb_dmat)[qgw_known])
    slb_estimator = slb_dmat + median_error
    qgw_dmat[~qgw_known] = slb_estimator[~qgw_known]
    ij = it.combinations(range(len(names)), 2)
    out = (
        (names[i], names[j], qgw_dmat[i, j], "QGW" if qgw_known[i, j] else "EST")
        for i, j in ij
    )
    batched_out = _batched(out, 1000)
    with open(gw_out_csv_location, "w", newline="") as outfile:
        csv_writer = csv.writer(outfile)
        for batch in batched_out:
            csv_writer.writerows(batch)
