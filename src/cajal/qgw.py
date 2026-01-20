"""Functions for computing the quantized Gromov-Wasserstein distance and the SLB between \
metric measure spaces, and related utilities for file IO and parallel computation."""

# std lib dependencies
import csv
import itertools as it
import sys
from math import sqrt
from typing import Collection, Iterable, Iterator, Literal, NewType, Optional, Set

if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm  # type: ignore[assignment]

# external dependencies
from multiprocessing import Pool

import numpy as np
import numpy.typing as npt
from scipy import cluster, sparse
from scipy.spatial.distance import pdist, squareform

from .gw_cython import qgw_init_cost, quantized_gw_cython
from .cajal_types import (
    Array,
    DistanceMatrix,
    Distribution,  # MetricMeasureSpace
    Matrix,
)
from .run_gw import (
    _batched,
    cell_iterator_csv,
    uniform,
)
from .run_gw import _batched, cell_iterator_csv, uniform

from .slb import l2


def distance_inverse_cdf(dist_mat: Array, measure: Array):
    """
    Compute the cumulative inverse distance function for a metric measure space.

    :param dX: Vectorform (one-dimensional) distance matrix for a space, of
    length N * (N-1)/2, where N is the number of points in dX.
    :param measure: Probability distribution on points of X, array of length N,
    entries are nonnegative and sum to one.

    :return: The inverse cumulative distribution function of the space, what
    Memoli calls "f_X^{-1}"; intuitively, a real valued function on the unit
    interval such that f_X^{-1}(u) is the distance `d` in X such that u is the
    proportion of pairs points in X that are at least as close together as `d`.
    """
    if len(dist_mat.shape) > 2:
        raise Exception("Array shape is " + str(dist_mat.shape) + ", should be 1D")
    elif len(dist_mat.shape) == 2 and dist_mat.shape[0] == dist_mat.shape[1]:
        dist_mat = squareform(dist_mat, force="tovector")

    index_X = np.argsort(dist_mat)
    dX = np.sort(dist_mat)
    mX_otimes_mX_sq = np.matmul(measure[:, np.newaxis], measure[np.newaxis, :])
    mX_otimes_mX = (
        2 * squareform(mX_otimes_mX_sq, force="tovector", checks=False)[index_X]
    )

    f = np.insert(dX, 0, 0.0)
    u = np.insert(mX_otimes_mX, 0, measure @ measure)

    return (f, u)


def slb_distribution(
    dX: Array,
    mX: Distribution,
    dY: Array,
    mY: Distribution,
):
    """
    Compute the SLB distance between two cells equipped with a choice of distribution.

    :param dX: Vectorform distance matrix for a space X, of length N * (N-1)/2,
        (where N is the number of points in X)
    :param mX: Probability distribution vector on X.
    :param dY: Vectorform distance matrix, of length M * (M-1)/2
        (where M is the number of points in X)
    :param mY: Probability distribution vector on X.
    """
    f, u = distance_inverse_cdf(dX, mX)
    g, v = distance_inverse_cdf(dY, mY)
    cum_u = np.cumsum(u)
    cum_v = np.cumsum(v)
    return 0.5 * sqrt(l2(f, u, cum_u, g, v, cum_v))


# SLB
def _init_slb_pool(sorted_cells, distributions):
    """
    Initialize the parallel SLB computation.

    Declares a global variable accessible from all processes.
    """
    global _SORTED_CELLS
    _SORTED_CELLS = list(zip(sorted_cells, distributions))


def _global_slb_pool(p: tuple[int, int]):
    """Compute the SLB distance between cells p[0] and p[1] in the global cell list."""
    i, j = p
    dX, mX = _SORTED_CELLS[i]  # type: ignore[name-defined]
    dY, mY = _SORTED_CELLS[j]  # type: ignore[name-defined]
    return (i, j, slb_distribution(dX, mX, dY, mY))
    # return (i, j, slb_cython(_SORTED_CELLS[i], _SORTED_CELLS[j]))


def slb_parallel_memory(
    cell_dms: Collection[DistanceMatrix],
    cell_distributions: Optional[Iterable[Distribution]],
    num_processes: int,
    chunksize: int = 20,
) -> DistanceMatrix:
    """
    Compute the SLB distance in parallel between all cells in `cell_dms`.

    :param cell_dms: A collection of distance matrices.
    :param cell_distributions: A collection of distributions on the cells, should
        be the same length as the collection of cell dms.
    :param num_processes: How many Python processes to run in parallel
    :param chunksize: How many SLB distances each Python process computes at a time

    :return: a square matrix giving pairwise SLB distances between points.
    """
    if cell_distributions is None:
        cell_distributions = [uniform(cell_dm.shape[0]) for cell_dm in cell_dms]
    cell_dms_sorted = [np.sort(squareform(cell, force="tovector")) for cell in cell_dms]
    N = len(cell_dms_sorted)

    with Pool(
        initializer=_init_slb_pool,
        initargs=(cell_dms_sorted, cell_distributions),
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
    out_csv: str,
    num_processes: int,
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
    distributions = [uniform(cell.shape[0]) for cell in cell_dms]
    slb_dmat = slb_parallel_memory(cell_dms, distributions, num_processes, chunksize)
    NN = len(names)
    total_num_pairs = int((NN * (NN - 1)) / 2)
    ij = tqdm(it.combinations(range(NN), 2), total=total_num_pairs)
    with open(out_csv, "w", newline="") as outfile:
        csv_writer = csv.writer(outfile)
        csv_writer.writerow(["first_object", "second_object", "slb_dist"])
        batches = _batched(
            ((names[i], names[j], str(slb_dmat[i, j])) for i, j in ij), 2000
        )
        for batch in batches:
            csv_writer.writerows(batch)


def quantize_icdm_reduced(
    A: DistanceMatrix,
    p: Distribution,
    clustering: npt.NDArray[np.int_],
    compute_gw_loss: bool,
) -> tuple[DistanceMatrix, Distribution]:
    """Cluster the cells of A based on the given clustering and return a \
    new matrix / distribution pair (metric measure space) based on the clustering.

    The new metric measure place has as its points the medoids of the original clusters,
    and the distances are inherited from the original space.

    The function is named "reduced" because it discards information
    which is relevant to the computation of the quantized GW of
    Chowdhury, Miller and Needham. Their concept is an upper bound for
    GW, this notion of "reduced quantized GW" is not. Because GW is a
    metric, it is possible to bound the error of this approximation in
    terms of the clustering. If the radius of each cluster (as
    measured from the medoid) is at most :math:`epsilon`, then the
    GW distance of the pairing will be at most epsilon.
    (For any clustering there is an obvious transport plan whose GW
    distortion can be computed easily.)

    The order of points in the new space reflects the numerical ordering of
    values in the `clustering` vector, i.e., if `clustering==[4,2,5,4,2]` then
    the first point in p will correspond to cluster 2, the second point will correspond \
    to cluster 4 and the third point will correspond to cluster 5.

    """
    medoid_list = []
    new_dist_probs = []
    for i in sorted(set(clustering)):
        indices = clustering == i
        local_dmat = A[:, indices][indices, :]
        medoid_list.append(np.argmin(sum(local_dmat)))
        new_dist_probs.append(np.sum(p[indices]))
    medoid_indices = np.array(medoid_list)
    medoid_icdm = A[medoid_indices, :][:, medoid_indices]
    return medoid_icdm, np.array(new_dist_probs)


class quantized_icdm:
    """
    A "quantized" intracell distance matrix.

    A metric measure space which has been equipped with a given clustering; it
    contains additional data which allows for the rapid computation of pairwise
    GW distances across many cells. Users should only need to understand how to
    use the constructor. Usage of this class will result in high memory usage if
    the number of cells to be constructed is large.

    :param cell_dm: An intracell distance matrix in squareform.
    :param p: A probability distribution on the points of the metric space
    :param num_clusters: How many clusters to subdivide the cell into; the more
        clusters, the more accuracy, but the longer the computation.
    :param clusters: Labels for a clustering of the points in the cell. If no clustering
        is supplied, one will be derived by hierarchical clustering until
        `num_clusters` clusters are formed. If a clustering is supplied, then
        `num_clusters` is ignored.
    """

    n: int
    # 2 dimensional square matrix of side length n.
    icdm: DistanceMatrix
    # "distribution" is a dimensional vector of length n,
    # a probability distribution on points of the space
    distribution: Distribution
    # The number of clusters in the quantized cell, which is *NOT* guaranteed
    # to be equal to the value of "clusters" specified in the constructor. Check this
    # field when iterating over clusters rather than assuming it has the number of clusters
    # given by the argument `clusters` to the constructor.
    ns: int
    # A square sub-matrix of icdm, the distance matrix between sampled points. Of side length ns.
    sub_icdm: DistanceMatrix
    # q_indices is a 1-dimensional array of integers of length ns+1. For i,j < ns,
    # icdm[sample_indices[i],sample_indices[j]]==sub_icdm[i,j].
    # sample_indices[ns]==n.
    q_indices: npt.NDArray[np.int_]
    # The quantized distribution; a 1-dimensional array of length ns.
    q_distribution: Distribution
    # This field is equal to np.dot(np.dot(np.multiply(icdm,icdm),distribution),distribution)
    c_A: float
    c_As: float
    A_s_a_s: Array

    @staticmethod
    def _sort_icdm_and_distribution(
        cell_dm: DistanceMatrix,
        p: Distribution,
        clusters: npt.NDArray[np.int_],
    ) -> tuple[DistanceMatrix, Distribution, npt.NDArray[np.int_]]:
        """
        Sort the cell distance matrix so that points in the same cluster are grouped \
        together and the points of each cell are in descending order.

        :param clusters: A vector of integer cluster labels telling which
            cluster each point belongs to, cluster labels are assumed to be contiguous and
            start at 1.

        :return: A sorted cell distance matrix, distribution, and a vector of
            integers marking the initial starting points of each cluster. (This
            has one more element than the number of distinct clusters, the last
            element is the length of the cell.)
        """
        indices: npt.NDArray[np.int_] = np.argsort(clusters)
        cell_dm = cell_dm[indices, :][:, indices]
        clusters = clusters[indices]
        p = p[indices]

        for i in set(clusters):
            permutation = np.nonzero(clusters == i)[0]
            this_cluster = cell_dm[permutation, :][:, permutation]
            medoid = np.argmin(sum(this_cluster))
            new_local_indices = np.argsort(this_cluster[medoid])
            cell_dm[permutation, :] = cell_dm[permutation[new_local_indices], :]
            cell_dm[:, permutation] = cell_dm[:, permutation[new_local_indices]]
            indices[permutation] = indices[permutation[new_local_indices]]
            p[permutation] = p[permutation[new_local_indices]]
            # q.append(np.sum(p[permutation]))

        q_indices = np.asarray(
            np.nonzero(np.r_[1, np.diff(np.sort(clusters)), 1])[0], order="C"
        )

        return (np.asarray(cell_dm, order="C"), p, q_indices)

    def __init__(
        self,
        cell_dm: DistanceMatrix,
        p: Distribution,
        num_clusters: int,
        clusters: Optional[npt.NDArray[np.int_]] = None,
    ):
        """Class constructor."""
        # Validate the data.
        assert len(cell_dm.shape) == 2

        self.n = cell_dm.shape[0]

        if clusters is None:
            # Cluster the data and set icdm, distribution, and ns.
            Z = cluster.hierarchy.linkage(squareform(cell_dm), method="centroid")
            clusters = cluster.hierarchy.fcluster(
                Z, num_clusters, criterion="maxclust", depth=0
            )

        icdm, distribution, q_indices = quantized_icdm._sort_icdm_and_distribution(
            cell_dm, p, clusters
        )

        self.icdm = icdm
        self.distribution = distribution
        self.ns = len(set(clusters))
        self.q_indices = q_indices

        clusters_sort = np.sort(clusters)

        # Compute the quantized distribution.
        q = []
        for i in range(self.ns):
            q.append(np.sum(distribution[q_indices[i] : q_indices[i + 1]]))
        q_arr = np.array(q, dtype=np.float64, order="C")
        self.q_distribution = q_arr
        assert abs(np.sum(q_arr) - 1.0) < 1e-7
        medoids = np.nonzero(np.r_[1, np.diff(clusters_sort)])[0]

        A_s = icdm[medoids, :][:, medoids]
        # assert np.all(np.equal(original_cell_dm[:, indices][indices, :], cell_dm))
        self.sub_icdm = np.asarray(A_s, order="C")
        self.c_A = np.dot(np.dot(np.multiply(icdm, icdm), distribution), distribution)
        self.c_As = np.dot(np.multiply(A_s, A_s), q_arr) @ q_arr
        self.A_s_a_s = np.dot(A_s, q_arr)

    @staticmethod
    def of_tuple(p):
        """Unpack the tuple p and apply the class constructor."""
        cell_dm, p, num_clusters, clusters = p
        return quantized_icdm(cell_dm, p, num_clusters, clusters)

    @staticmethod
    def of_ptcloud(
        X: Matrix,
        distribution: Distribution,
        num_clusters: int,
        method: Literal["kmeans"] | Literal["hierarchical"] = "kmeans",
    ):
        """Construct a quantized icdm from a point cloud."""
        dmat = squareform(pdist(X), force="tomatrix")
        if method == "hierarchical":
            return quantized_icdm(dmat, distribution, num_clusters)
        # Otherwise use kmeans.
        # TODO: This will probably give way shorter than the amount of cells.
        _, clusters = cluster.vq.kmeans2(X, num_clusters, minit="++")
        return quantized_icdm(dmat, distribution, None, clusters)


def quantized_gw(
    A: quantized_icdm,
    B: quantized_icdm,
    initial_plan: Optional[npt.NDArray[np.float64]] = None,
) -> tuple[sparse.csr_matrix, float]:
    """
    Compute the quantized Gromov-Wasserstein distance between two quantized metric measure spaces.

    :param initial_plan: An initial guess at a transport plan from A.sub_icdm to B.sub_icdm.
    """
    if initial_plan is None:
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
    else:
        init_cost = -2 * (A.sub_icdm @ initial_plan @ B.sub_icdm)
        T_rows, T_cols, T_data = qgw_init_cost(
            A.distribution,
            A.sub_icdm,
            A.q_indices,
            A.q_distribution,
            A.c_As,
            B.distribution,
            B.sub_icdm,
            B.q_indices,
            B.q_distribution,
            B.c_As,
            init_cost,
        )

    P = sparse.coo_matrix((T_data, (T_rows, T_cols)), shape=(A.n, B.n)).tocsr()
    gw_loss = A.c_A + B.c_A - 2.0 * float(np.tensordot(A.icdm, P.dot(P.dot(B.icdm).T)))
    return P, sqrt(max(gw_loss, 0)) / 2.0


def _block_quantized_gw(indices):
    # Assumes that the global variable _QUANTIZED_CELLS has been declared, as by
    # init_qgw_pool
    (i0, i1), (j0, j1) = indices

    gw_list = []
    for i in range(i0, i1):
        A = _QUANTIZED_CELLS[i]  # type: ignore[name-defined]
        for j in range(j0, j1):
            if i < j:
                B = _QUANTIZED_CELLS[j]  # type: ignore[name-defined]
                gw_list.append((i, j, quantized_gw(A, B)))
    return gw_list


def _init_qgw_pool(quantized_cells: list[quantized_icdm]):
    """Initialize the parallel quantized GW computation by declaring a global variable \
    accessible from all processes."""
    global _QUANTIZED_CELLS
    _QUANTIZED_CELLS = quantized_cells  # type: ignore[name-defined]


def _quantized_gw_index(p: tuple[int, int]) -> tuple[int, int, float]:
    """Given input p= (i,j), compute the quantized GW distance between cells i \
    and j in the global list of quantized cells."""
    i, j = p
    return (
        i,
        j,
        quantized_gw(_QUANTIZED_CELLS[i], _QUANTIZED_CELLS[j])[1],
    )  # type: ignore[name-defined]


def quantized_gw_parallel_memory(
    quantized_cells: Collection[quantized_icdm],
    num_processes: int,
    chunksize: int = 20,
):
    """
    Compute the quantized Gromov-Wasserstein distances between all pairs of cells in the list.

    Coupling matrices will be discarded.
    """
    N = len(quantized_cells)
    total_num_pairs = int((N * (N - 1)) / 2)
    # index_pairs = tqdm(it.combinations(iter(range(N)), 2), total=total_num_pairs)
    index_pairs = it.combinations(iter(range(N)), 2)
    gw_dists: Iterator[tuple[int, int, float]]
    with Pool(
        initializer=_init_qgw_pool, initargs=(quantized_cells,), processes=num_processes
    ) as pool:
        gw_dists = tqdm(
            pool.imap_unordered(_quantized_gw_index, index_pairs, chunksize=chunksize),
            total=total_num_pairs,
        )  # type: ignore[assignment]
        gw_dists_list = list(gw_dists)
    gw_dists_list.sort(key=lambda p: p[0] * N + p[1])
    return gw_dists_list


def quantized_gw_parallel(
    intracell_csv_loc: str,
    num_processes: int,
    num_clusters: int,
    out_csv: str,
    chunksize: int = 20,
    verbose: bool = False,
    write_blocksize: int = 100,
) -> None:
    """
    Compute the quantized Gromov-Wasserstein distance in parallel between all cells in a family \
    of cells.

    Read icdms from file, quantize them, compute pairwise qGW
    distances between icdms, and write the result to file.

    :param intracell_csv_loc: path to a CSV file containing the cells to process
    :param num_processes: number of Python processes to run in parallel
    :param num_clusters: Each cell will be partitioned into `num_clusters` many clusters.
    :param out_csv: file path where a CSV file containing
         the quantized GW distances will be written
    :param chunksize: How many q-GW distances should be computed at a time by each parallel process.
    """
    if verbose:
        print("Reading files...")
        cells = [cell for cell in tqdm(cell_iterator_csv(intracell_csv_loc))]
        names, cell_dms = zip(*cells)
        del cells
    else:
        names, cell_dms = zip(*cell_iterator_csv(intracell_csv_loc))
    if verbose:
        print("Quantizing intracell distance matrices...")
    with Pool(processes=num_processes) as pool:
        args = [
            (cell_dm, uniform(cell_dm.shape[0]), num_clusters, None)
            for cell_dm in cell_dms
        ]
        quantized_cells = list(
            tqdm(pool.imap(quantized_icdm.of_tuple, args), total=len(names))
        )

    print("Computing pairwise Gromov-Wasserstein distances...")
    gw_dists = quantized_gw_parallel_memory(quantized_cells, num_processes, chunksize)

    with open(out_csv, "w", newline="") as outcsvfile:
        csvwriter = csv.writer(outcsvfile)
        csvwriter.writerow(["first_object", "second_object", "quantized_gw"])
        for i, j, gw_dist in gw_dists:
            csvwriter.writerow((names[i], names[j], gw_dist))


# A BooleanSquareMatrix is a square matrix of booleans.
BooleanSquareMatrix = NewType("BooleanSquareMatrix", npt.NDArray[np.bool_])


def _cutoff_of(
    slb_dmat: DistanceMatrix,
    median: float,
    gw_dmat: DistanceMatrix,
    gw_known: BooleanSquareMatrix,
    nn: int,
):
    # maxval = np.max(gw_dmat)
    gw_copy = np.copy(gw_dmat)
    # gw_copy[~gw_known]=maxval
    gw_copy[~gw_known] = (slb_dmat + median)[~gw_known]
    gw_copy.partition(nn + 1, axis=1)
    return gw_copy[:, nn + 1]


def _tuple_set_of(
    X: npt.NDArray[np.int_], Y: npt.NDArray[np.int_]
) -> Set[tuple[int, int]]:
    b = set()
    tuple_iter: Iterator[tuple[int, int]] = map(
        tuple, np.stack((X, Y), axis=1).astype(int)
    )  # type: ignore[arg-type]
    for i, j in tuple_iter:
        if i < j:
            b.add((i, j))
        elif i > j:
            b.add((j, i))
        # Currently, the case i == j is possible, and may be observed
        # if there are distinct cells whose
        # slb distance is zero.
    return b
