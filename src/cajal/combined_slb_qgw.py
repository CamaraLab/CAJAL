"""Classes and helper functions for a main function \
'combined_slb_quantized_gw' which first computes the pairwise \
SLB for all cells, then computes a \
fraction of the pairwise qGW values \
until an acceptable threshold of accuracy is met."""

import csv
import itertools as it
from multiprocessing import Pool
from typing import Collection, Iterable, NewType, Optional

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import squareform
from tqdm.notebook import tqdm

from .qgw import (
    Array,
    _init_qgw_pool,
    _quantized_gw_index,
    _tuple_set_of,
    quantized_icdm,
    slb_parallel_memory,
)
from .run_gw import (
    DistanceMatrix,
    Distribution,
    Matrix,
    _batched,
    cell_iterator_csv,
    uniform,
)

# A BooleanSquareMatrix is a square matrix of booleans.
BooleanSquareMatrix = NewType("BooleanSquareMatrix", npt.NDArray[np.bool_])


def _get_quantiles(A: DistanceMatrix, n: int) -> tuple[Array, npt.NDArray[np.int_]]:
    """
    Compute the quantiles of the entries of A that do not lie along the diagonal.

    As an example, suppose that `A` has no zeros off the diagonal and that `n==200`.
    In this case, `quantile_bin_edges` will have 201 entries, and the
    values of `quantile_bin_indices` will range between 0 and 201 inclusive.
    Values from 1 to 200 (inclusive) will represent the internals of
    the interval between

    :return: A pair `(quantile_bin_edges, quantile_bin_indices)`,
        where `quantile_bin_edges` is an `Array` of length `n+1`
        whose bins partition the elements of A into n groups, and
        `quantile_bin_indices` is a matrix of integers of the same
        shape as `A` whose values range from `0` to `n+1` inclusive.
        Values in the range 1..n inclusive
    """
    even_intervals = np.arange(n + 1).astype(float) / float(n)
    quantile_bin_edges = np.quantile(
        squareform(A, force="tovector"), even_intervals, overwrite_input=True
    )
    quantile_bin_indices = np.digitize(A, quantile_bin_edges)
    return quantile_bin_edges, quantile_bin_indices


def _off_diag(N: int):
    A = np.full((N, N), True, np.bool_)
    A[np.arange(N), np.arange(N)] = False
    return A


def _bin_index_vectors_of_bin_index_matrix(A: npt.NDArray[np.int_]):
    # bin_index_vectors[0] includes all values in the minimal quantile, in particular
    # all values along the diagonal; it makes sense to ignore these.
    bin_index_vectors = [np.nonzero(A == 0 & _off_diag(A.shape[0]))]
    for i in range(1, np.max(A) + 1):
        bin_index_vectors.append(np.nonzero(i == A))
    return bin_index_vectors


SamplingNumber = NewType("SamplingNumber", int)


class _Error_Distribution:
    def __init__(self, SLB_dmat: DistanceMatrix, slb_bins: int, sn: SamplingNumber):
        # The SLB distance matrix will be broken into
        # `slb_bins` many equally sized bins, and the error distribution will be estimated
        # within each bin. If `slb_bins == 1`, this amounts to assuming that the
        # conditional error is independent of SLB.
        # slb_bins should not be chosen too large relative to the
        # total number of cell pairs, or each bin will have very few
        # samples from which to estimate the error distribution within
        # that bin.
        self._slb_bins = slb_bins
        # This determines how many quantiles each SLB_bin will be split into;
        # it determines with what precision we try to estimate the conditional probability.
        # A default of `sn=200` will result in estimating the
        # conditional probability to the nearest half-percent.
        self._error_bins = sn
        # _quantile_bin_edges is the `Array` of endpoints for the SLB bins.
        # _quantile_bin_indices is a square matrix of integer bin
        # labels telling which SLB quantile bin the corresponding cell pair belongs to.
        self._quantile_bin_edges, self._quantile_bin_indices = _get_quantiles(
            SLB_dmat, slb_bins
        )
        assert self._quantile_bin_edges.shape == (slb_bins + 1,)
        assert self._quantile_bin_indices.shape == SLB_dmat.shape
        assert np.max(self._quantile_bin_indices) == slb_bins + 1
        self._bin_index_vectors = _bin_index_vectors_of_bin_index_matrix(
            self._quantile_bin_indices
        )
        self._error_distribution = np.zeros((slb_bins + 2, self._error_bins + 1))
        self._median = np.zeros((slb_bins + 2,))

    def update_distribution(
        self,
        # GW_dmat: DistanceMatrix,
        observed_error: Matrix,
        GW_known: BooleanSquareMatrix,
    ) -> None:
        """Update the internal inferred conditional distribution of \
        errors based on the observed empirical distribution of errors.

        :param observed_error: A square distance matrix where
        `observed_error[i,j]` is the error `qGW[i, j] - SLB_dmat[i,
        j]` where `GW_known[i, j]` is true; otherwise, it's
        unimportant what the value is.

        """
        for i in range(1, self._slb_bins + 2):
            X, Y = self._bin_index_vectors[i]
            GW_known_i = GW_known[X, Y]
            known_pairs = np.nonzero(GW_known_i)
            unknown_pairs = np.nonzero(np.logical_not(GW_known_i))
            Xk, Yk = X[known_pairs], Y[known_pairs]
            assert np.all(GW_known[Xk, Yk])
            Xuk, Yuk = X[unknown_pairs], Y[unknown_pairs]
            assert not np.any(GW_known[Xuk, Yuk])
            observed_error_i = observed_error[Xk, Yk]
            self._error_distribution[i] = (
                np.quantile(
                    observed_error_i,
                    np.arange(self._error_bins + 1).astype(float)
                    / float(self._error_bins),
                )
                if Xk.shape[0] > 0
                else np.zeros((self._error_bins + 1,))
            )
            self._median[i] = np.median(observed_error_i) if Xk.shape[0] > 0 else 0.0
            # self.estimator_matrix[Xk, Yk] = GW_dmat[Xk, Yk]
            # self.estimator_matrix[Xuk, Yuk] = self._SLB_dmat[Xuk, Yuk] + self._median[i]

    def get_median(
        self, X_indices: npt.NDArray[np.int_], Y_indices: npt.NDArray[np.int_]
    ):
        bins = self._quantile_bin_indices[X_indices, Y_indices]
        return self._median[bins]

    def cdf(
        self,
        X_indices: npt.NDArray[np.int_],
        Y_indices: npt.NDArray[np.int_],
        values: Array,
    ):
        bins = self._quantile_bin_indices[X_indices, Y_indices]
        assert bins.shape == values.shape
        indices = np.array(
            [
                np.searchsorted(self._error_distribution[i], v)
                for i, v in zip(bins, values)
            ]
        )
        probabilities = indices / self._error_bins
        probabilities = np.maximum(
            probabilities, np.zeros(probabilities.shape), dtype=np.float64
        )
        probabilities = np.minimum(
            probabilities, np.ones(probabilities.shape), dtype=np.float64
        )
        return probabilities


def _nn_indices_from_slb(slb_dmat: DistanceMatrix, nearest_neighbors: int):
    N = slb_dmat.shape[0]
    # The following line of code was changed from:
    # ind_y = np.argsort(slb_dmat, axis=1)[:, 1 : nearest_neighbors + 1]
    # to include the 0th column, because if a row of slb_dmat contains
    # zeros off the diagonal (i.e. if the reported SLB between two
    # distinct cells is zero) then the 0th column of ind_y may
    # represent a pair (i,j) for distinct i,j.
    ind_y = np.argsort(slb_dmat, axis=1)[:, : nearest_neighbors + 1]
    ind_x = np.broadcast_to(np.arange(N)[:, np.newaxis], (N, nearest_neighbors + 1))
    xy = np.reshape(np.stack((ind_x, ind_y), axis=2), (-1, 2))
    b = _tuple_set_of(xy[:, 0], xy[:, 1])
    return b


def _sample_indices_by_bin(slb_dmat: DistanceMatrix, slb_bins: int, sn: SamplingNumber):
    slb_quantile_bins = np.quantile(
        squareform(slb_dmat, force="tovector"),
        np.arange(slb_bins + 1).astype(float) / float(slb_bins),
        overwrite_input=True,
    )
    # The values in here will range from 0 to slb_bins + 1,
    # possibly inclusive on the left (but not necessarily),
    # definitely inclusive on the right.
    slb_quantiles = np.digitize(slb_dmat, slb_quantile_bins)
    bin_samples: list[list[tuple[int, int]]] = [[] for i in range(slb_bins + 2)]
    for i in range(slb_dmat.shape[0]):
        for j in range(i + 1, slb_dmat.shape[1]):
            quantile = slb_quantiles[i, j]
            if len(bin_samples[quantile]) < sn:
                bin_samples[quantile].append((i, j))
    return set(it.chain.from_iterable(bin_samples))


def _get_initial_indices(
    slb_dmat: DistanceMatrix,
    nearest_neighbors: int,
    slb_bins: int,
    sn: SamplingNumber,
):
    b1 = _nn_indices_from_slb(slb_dmat, nearest_neighbors)
    b2 = _sample_indices_by_bin(slb_dmat, slb_bins, sn)
    return b1.union(b2)


def _indices_from_cdf_prob(
    N: int,
    X: npt.NDArray[np.int_],
    Y: npt.NDArray[np.int_],
    cdf_prob: Array,
    nearest_neighbors: int,
    accuracy: float,
    exp_decay: float,
) -> list[tuple[int, int]]:
    # This array is crucial. It contains the list of cell pair indices
    # to be computed in order of priority - in *descending* order of
    # cdf_prob.  When cdf_prob is large, say 70%, it means that 70% of
    # the time, the observed error will be less than that value.  If
    # cdf_prob >= 50%, it means that there is at least a 50% chance
    # that the actual GW value will be *less* than the cutoff, and we
    # should compute the value of the cell pair.  If cdf_prob is 30%,
    # then most probably the actual GW value is larger than the cutoff
    # and so the cell pair is not a nearest neighbor; however, there
    # is a 30% chance that the actual GW value is less than the
    # cutoff, and so the expected injury of the cell pair to the
    # nearest neighbors list is 30%. Thus, we should prioritize the
    # cells by descending order of cdf_prob.
    undershooting_prob_indices = np.argsort(-cdf_prob)
    median_estimate_cutoff_index = int(
        np.searchsorted(-cdf_prob[undershooting_prob_indices], -0.5)
    )
    total_expected_injuries = np.sum(cdf_prob)
    incremental_expected_injuries = np.cumsum(cdf_prob[undershooting_prob_indices])
    acceptable_injuries = (nearest_neighbors * N) * (1 - accuracy)
    acceptable_injury_index = int(
        np.searchsorted(
            incremental_expected_injuries,
            total_expected_injuries - acceptable_injuries,
        )
    )
    cutoff_index = max(acceptable_injury_index, median_estimate_cutoff_index)
    block_size = 2500
    if cutoff_index <= block_size:
        indices = undershooting_prob_indices[:cutoff_index]
        return list(_tuple_set_of(X[indices], Y[indices]))

    indices = undershooting_prob_indices[: int(cutoff_index / exp_decay)]
    return list(_tuple_set_of(X[indices], Y[indices]))


def cutoff_of(estimator_matrix: DistanceMatrix, nn: int):
    """Return the vector of current cutoffs for the nn-th nearest neighbor."""
    return np.sort(estimator_matrix, axis=1)[:, nn + 1]


def unknown_indices_of(gw_known: BooleanSquareMatrix):
    """Return upper-triangular indices (i,j) where gw is unknown."""
    Xuk_ts, Yuk_ts = np.nonzero(np.logical_not(gw_known))
    upper_triangular = Xuk_ts <= Yuk_ts
    Xuk = Xuk_ts[upper_triangular]
    Yuk = Yuk_ts[upper_triangular]
    return Xuk, Yuk


def estimator_matrix_of(
    slb_dmat: DistanceMatrix,
    gw_dmat: DistanceMatrix,
    gw_known: BooleanSquareMatrix,
    ed: _Error_Distribution,
    Xuk: npt.NDArray[np.int_],
    Yuk: npt.NDArray[np.int_],
):
    """
    Compute a best-estimate gw matrix.

    The matrix agrees with gw_dmat where those values are known, and
    elsewhere is a best guess imputed from SLB and the inferred error
    distribution of SLB vs GW.
    """
    estimator_matrix = np.copy(slb_dmat)
    estimator_matrix[gw_known] = gw_dmat[gw_known]
    conditional_median_error = ed.get_median(Xuk, Yuk)
    estimator_matrix[Xuk, Yuk] += conditional_median_error
    return estimator_matrix


def _get_indices(
    ed: _Error_Distribution,
    slb_dmat: DistanceMatrix,
    gw_dmat: DistanceMatrix,
    gw_known: BooleanSquareMatrix,
    accuracy: float,
    nearest_neighbors: int,
    exp_decay: float,
    sn: SamplingNumber,
) -> list[tuple[int, int]]:
    """
    Compute cell pair indices for a single iteration of the main loop.

    Based on the SLB distance matrix and the partially known GW
    distance matrix, and the desired accuracy, return a list of cell
    pairs which we should compute.  This function does not return
    *all* cell pairs that must be computed for the desired accuracy;
    it is expected that the function will be called *repeatedly*, and
    that a new list of cell pairs will be returned every time, with the
    list sorted in descending order of priority. When the empty list
    is returned, this indicates that the gw distance table is already
    at the desired accuracy, and the loop should terminate.

    :param slb_dmat: the SLB distance matrix in squareform, but this would make sense for
        any lower bound for gw_dmat
    :param gw_dmat: A partially defined
        Gromov-Wasserstein distance matrix in squareform, we should
        have gw_dmat >= slb_dmat almost everywhere where gw_known is
        true for this to make sense (I have not yet checked how this
        behaves in the case where some values of slb_dmat are greater
        than gw_dmat); should be zero elsewhere.  :param gw_known: A
        matrix of Booleans which is true where the entries of
        `gw_dmat` are correct/valid and false where the entries are
        not meaningful/do not yet have the correct value :param
        accuracy: This is a real number between 0 and 1 inclusive. If
        the accuracy is 1, then pairwise cells will continue to be
        computed until all remaining uncomputed cell pairs have an SLB
        distance which is strictly higher than anything on the list of
        `nearest_neighbors` many nearest neighbors of every point;
        thus the reported array of distances is guaranteed to be
        correct out to the first `nearest_neighbors` nearest neighbors
        of every point.
    """
    # xy is of shape (a,2)
    # We assume that at least the initial values have been computed.
    N = slb_dmat.shape[0]
    # ed maintains an internal state representing the conditional error distribution.
    Xuk, Yuk = unknown_indices_of(gw_known)
    estimator_matrix = estimator_matrix_of(slb_dmat, gw_dmat, gw_known, ed, Xuk, Yuk)
    cutoff = cutoff_of(estimator_matrix, nearest_neighbors)
    distance_below_threshold = cutoff[:, np.newaxis] - slb_dmat
    cdf_prob = ed.cdf(Xuk, Yuk, distance_below_threshold[Xuk, Yuk])
    return _indices_from_cdf_prob(
        N, Xuk, Yuk, cdf_prob, nearest_neighbors, accuracy, exp_decay
    )


def _update_dist_mat(
    gw_dist_iter: Iterable[tuple[int, int, float]],
    dist_mat: npt.NDArray[np.float64],
    dist_mat_known: npt.NDArray[np.bool_],
) -> None:
    """
    Write the values in `gw_dist_iter` to the matrix `dist_mat` and update the matrix \
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


# def _validate_error_distribution_sampling_method(
#     x: Literal["nearest neighbors"] | SamplingNumber,
# ) -> None:
#     if isinstance(x, str) and x != "nearest_neighbors":
#         raise ValueError(
#             "error_distribution_sampling_method is "
#             + str(x)
#             + "function only supports 'nearest_neighbors'"
#         )
#     elif not isinstance(x, SamplingNumber):
#         raise ValueError(
#             "Unrecognized data type for error_distribution_sampling_method"
#         )
#     return None


def combined_slb_quantized_gw_memory(
    cell_dms: Collection[DistanceMatrix],  # Squareform
    cell_distributions: Optional[Iterable[Distribution]],
    num_processes: int,
    num_clusters: int,
    accuracy: float,
    nearest_neighbors: int,
    verbose: bool,
    chunksize: int = 20,
    exp_decay: float = 2.0,
    slb_bins: int = 5,
    sn: SamplingNumber = SamplingNumber(200),
):
    """
    Compute a heuristic approximation to nearest neighbors of cells.

    Compute the pairwise SLB distances between each pair of cells in
    `cell_dms`.  Based on this initial estimate of the distances,
    compute the quantized GW distance between the nearest with
    `num_clusters` many clusters until the correct nearest-neighbors
    list is obtained for each cell with a high degree of confidence.

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
    :param accuracy: This is a real number between 0 and 1, inclusive.
    :param nearest_neighbors: The algorithm tries to compute only the
        quantized GW distances between pairs of cells if one is within the first
        `nearest_neighbors` neighbors of the other; for all other values,
        the SLB distance is used to give a rough estimate.
    :param exp_decay:
        This parameter controls the number of cells computed per
        iteration of the main loop.  At each iteration of the loop,
        the estimated error distribution of SLB vs QGW is re-estimated
        based on newly collected data, and this distribution informs
        the choice of what cell pairs to compute next and how many
        cell pairs still have to be computed.  Each iteration, we
        create a list of all cell pairs which we think still have to
        be computed and order them by priority, and then compute
        (1/exp_decay) of them, so the total number of iterations is
        logarithmic in the number of cell pairs. For example, when
        exp_decay=2.0, we propose a list of M cell pairs to compute
        the QGW of, then compute half of those, then we recompute the
        inferred probability distribution and repeat.  Iterations have
        a constant overhead (a fraction of a second) which is likely
        to be relatively insubstantial when large numbers of cells are
        involved; if iterations are between 5-10 minutes then the
        overhead per iteration will be likely
        negligible.
    :param error_distribution_sampling_method: Controls the approach
        used to infer the error distribution of SLB vs GW.  If
        "nearest neighbors", then we use the computed GW values for
        the smallest SLB values to estimate the distribution. If
        SamplingNumber(n) we additionally sample n values randomly through
        every half-percentile of the SLB distribution. This is more
        accurate and only adds a constant to the runtime, so it should
        be preferred.
    """
    N = len(cell_dms)
    np_arange_N = np.arange(N)
    if cell_distributions is None:
        cell_distributions = [uniform(cell.shape[0]) for cell in cell_dms]
    slb_dmat = slb_parallel_memory(
        cell_dms, cell_distributions, num_processes, chunksize
    )

    slb_bins = 5
    sn = SamplingNumber(max(sn, 1))
    ed = _Error_Distribution(slb_dmat, slb_bins, sn)

    # Partial quantized Gromov-Wasserstein table, will be filled in gradually.
    qgw_dmat = np.zeros((N, N), dtype=float)
    qgw_known = BooleanSquareMatrix(np.full(shape=(N, N), fill_value=False))
    qgw_known[np_arange_N, np_arange_N] = True

    quantized_cells = [
        quantized_icdm(cell_dm, uniform(cell_dm.shape[0]), num_clusters)
        for cell_dm in cell_dms
    ]
    total_cells_computed = 0
    with Pool(
        initializer=_init_qgw_pool, initargs=(quantized_cells,), processes=num_processes
    ) as pool:
        # This function does not have side effects.
        indices = _get_initial_indices(
            slb_dmat,
            nearest_neighbors,
            slb_bins,
            sn,
        )
        while len(indices) > 0:
            if verbose:
                print(
                    "Cell pairs computed so far: "
                    + str((np.count_nonzero(qgw_known) - N) / 2)
                )
                print("Cell pairs to be computed this iteration: " + str(len(indices)))

            total_cells_computed += len(indices)
            qgw_dists = pool.imap_unordered(
                _quantized_gw_index, indices, chunksize=chunksize
            )
            # State mutation warning!
            # This function reads the queue qgw_dists and modifies qgw_dmat and qgw_known.
            _update_dist_mat(qgw_dists, qgw_dmat, qgw_known)
            assert np.count_nonzero(qgw_known) == 2 * total_cells_computed + N
            # This function does not have side effects.
            ed.update_distribution(Matrix(qgw_dmat - slb_dmat), qgw_known)
            indices = _get_indices(
                ed,
                slb_dmat,
                qgw_dmat,
                qgw_known,
                accuracy,
                nearest_neighbors,
                exp_decay,
                sn,
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
    exp_decay: float = 2.0,
    slb_bins: int = 5,
    sn: SamplingNumber = SamplingNumber(200),
) -> None:
    """
    Read icdms from file, call :func:`cajal.qgw.combined_slb_quantized_gw_memory`, write to file.

    For all parameters not listed here see the docstring for
    :func:`cajal.qgw.combined_slb_quantized_gw_memory`.

    :param input_icdm_csv_location: file path to a csv file. For format for the icdm
        see :func:`cajal.run_gw.icdm_csv_validate`.
    :param gw_out_csv_location: Where to write the output GW distances.
    :return: None.
    """
    if verbose:
        print("Reading files...")
        names, cell_dms = zip(*tqdm(cell_iterator_csv(input_icdm_csv_location)))
    else:
        names, cell_dms = zip(*cell_iterator_csv(input_icdm_csv_location))
    slb_dmat, qgw_dmat, qgw_known = combined_slb_quantized_gw_memory(
        cell_dms,
        None,
        num_processes,
        num_clusters,
        accuracy,
        nearest_neighbors,
        verbose,
        chunksize,
        exp_decay,
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
