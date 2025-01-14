from futhark_ffi import Futhark
from types import MethodType
from typing import Union, Optional
from cajal.run_gw import DistanceMatrix
from cajal.run_gw import Distribution
from cajal.run_gw import cell_iterator_csv
from cajal.utilities import n_c_2, uniform
import numpy.typing as npt
from math import exp, log
import cajal.gw_cython
import numpy as np
import itertools as it
from scipy.spatial.distance import squareform, pdist


def ugw_bound(gw_cost: float, rho1: float, rho2: float):
    """
    :param gw_cost: A known GW transport cost between two cells or metric spaces.
        Recall that we distinguish between GW cost and GW distance; the GW distance is
        defined as 0.5 * math.sqrt(gw_cost).
    :param rho1: The first marginal penalty
    :param rho2: The second marginal penalty
    :returns: An upper bound on the UGW cost that can be directly computed from
        the known GW cost and rho1, rho2.
    """
    sum = rho1 + rho2
    return sum * (1 - exp(-gw_cost / sum))


def mass_lower_bound(gw_cost: float, rho1: float, rho2: float):
    """
    :param gw_cost: A known GW transport cost between two cells or metric spaces.
        Recall that we distinguish between GW cost and GW distance;
        the GW distance is defined as 0.5 * math.sqrt(gw_cost).
    :param rho1: The first marginal penalty
    :param rho2: The second marginal penalty
    :returns: A lower bound on the mass preserved by the transport plan that can be
        directly computed from the known GW cost and rho1, rho2.
    """
    sum = rho1 + rho2
    return exp(-gw_cost / (2 * sum))


def estimate_distr(
    dmats: npt.NDArray[np.float64], sample_size: int = 100, quantile=0.15
):
    """
    Select `sample_size` many pairs of distance matrices from the given set
    of distance matrices, compute the Gromov-Wasserstein costs for each pair,
    and return the specified quantile of the observed distribution of GW costs.

    :param dmats: An array of shape (k, n, n) where `k` is the number of cells,
        and `n` is the number of points sampled from each cell.
    :param sample_size: How many cell pairs to sample.
    :param quantile: The quantile of the observed distribution to return.
    """
    num_dmats = dmats.shape[0]
    n_pairs = n_c_2(num_dmats)
    k = int(n_pairs / sample_size)
    u = uniform(dmats.shape[1])
    to_sample = it.islice(it.combinations(dmats, 2), 0, None, k)
    observations = np.array(
        [cajal.gw_cython.gw(cell1, u, cell2, u)[1] for cell1, cell2 in to_sample],
        dtype=float,
    )
    observations *= 2
    observations = observations * observations
    return np.quantile(observations, quantile)


def rho_of(gw_cost: float, mass_kept: float):
    """
    :param gw_cost: The GW cost of the optimal transport plan between two cells.
        Recall that we distinguish between GW cost and GW distance; the GW distance is
        defined as 0.5 * math.sqrt(gw_cost).
    :param mass_kept: A real number between 0 and 1 indicating a desired lower bound on
        the mass kept by the UGW transport plan.
    :returns: A value for rho such that, if unbalanced GW is run with this parameter, at
        least proportion `mass_kept` of the mass of the cells will be created. For simplicity
        this formula is written for the case where the cells and the GW transport plan have
        unit mass. This function is the inverse of mass_lower_bound.
    """
    return -gw_cost / (4*log(mass_kept))


_rho1_docstring = """:param rho1: The first marginal penalty coefficient, controls how much the
    first marginal for the transport plan (sum along rows) is allowed to deviate from mu.
    Higher is more strict and should give closer marginals."""
_rho2_docstring = """:param rho2: The second marginal penalty coefficient,
    controls how much the second marginal for the transport plan (sum along columns)
    is allowed to deviate from nu."""
_eps_docstring = """:param eps: The entropic regularization coefficient. Increasing this value
    makes the problem more convex and the algorithm will converge to an answer faster,
    but it may be inaccurate (far from the true minimum).
    If it's set too low there will be numerical instability issues and the function will
    return NaN."""
_exp_absorb_cutoff_docstring = """:param exp_absorb_cutoff: A numerical stability parameter.
    The inner loop, the Sinkhorn algorithm, tries to solve an optimization problem of solving
    for diagonal matrices A, B to minimize a cost function Cost(AKB) where the matrix K is given.
    The values of A, K, B are numerically extreme under normal conditions and so we represent
    A by a pair of vectors (a_bar, u_bar) where a_bar is between exp_absorb_cutoff and
    1/exp_absorb_cutoff, and diag(A) = a_bar * e^u_bar. When a_bar exceeds exp_absorb_cutoff,
    part of a_bar is "absorbed into the exponent" u_bar, thus the name.
    Set this to any value such that floating point arithmetic operations in the range
    (1/exp_absorb_cutoff, exp_absorb_cutoff) are reasonably accurate and not too
    close to overflow."""
_safe_for_exp_docstring = """:param safe_for_exp: A numerical stability parameter.
    This is used only once during the initialization of the algorithm, the user
    supplies a number R and the code tries to construct an initial starting matrix K with
    the property that e^R is an upper bound for all elements in K,
    and as many as possible of the values of K are "pushed up against" this upper bound.
    The main risk here is of *underflow* of values in K; by increasing the values in K
    we give ourselves more room to maneuver with the spectrum of values that floats can
    represent. Choose a number such that e^R is big but still is a comfortable
    distance from the max float value, a good region for addition and multiplication
    without risk of overflow."""
_tol_sinkhorn_docstring = """:param tol_sinkhorn: An accuracy parameter, controls
    the tolerance at which the inner loop exits. Suggest 10^-5 to 10^-9.
    The inner loop, the Sinkhorn algorithm, tries to solve an optimization problem of solving
    for diagonal matrices A, B to minimize a cost function Cost(AKB) where the matrix K is given.
    This gives a series of iterations A_n, B_n, A_n+1, B_n+1, ... The exit condition for
    the loop is when diag(A_n+1)/diag(A_n) is everywhere within 1 +/- tol_sinkhorn."""
_tol_outerloop_docstring = """:param tol_outerloop: An accuracy parameter, controls the
    tolerance at which the outer loop exits - when the ratio T_n+1/T_n is everywhere within
    1 +/- tol_outerloop. (T_n is the nth transport plan found in the main gradient descent)"""
_increasing_ratio_docstring = """:param increasing_ratio: The UGW algorithm is not
    numerically stable
    and on large datasets, NaNs in the output are likely. If increasing_ratio is not None, this
    function makes a second pass through the input and retries the UGW algorithm,
    each time multiplying the regularization parameter epsilon by a factor of
    increasing_ratio until it converges. Higher values of epsilon may make the results
    less accurate in some cases, but it may be more useful than discarding the data for which the
    algorithm did not converge."""
_mass_kept_docstring = """:param mass_kept: The user has the option to
    supply a real number between 0 and 1, "mass_kept", which takes precedence
    over the given rho1 and rho2 if it is not None. If this value is supplied,
    then the algorithm chooses rho1 and rho2 in such a way as to bound below the mass
    kept by the transport plan."""
_gw_cost_docstring = """:param gw_cost: This must be supplied
    if mass_kept is not None, as it is necessary to compute the
    appropriate values of rho1, rho2."""

_ugw_armijo_docstring = "\n".join(
    [
        """Given two metric measure spaces (A,mu) and (B, nu), compute the
unbalanced Gromov-Wasserstein distance between them,
using an algorithm based on that of Séjourné, Vialard and Peyré,
with some modifications for numerical stability and convergence.""",
        "",
        _rho1_docstring,
        _rho2_docstring,
        _eps_docstring,
        """:param A_dmat: A square pairwise distance matrix of shape (n, n),
    symmetric and with zeroes along the diagonal.""",
        """:param mu: A one-dimensional vector of length n with strictly
    positive entries. (The algorithm does not currently support zero entries.)""",
        """:param B_dmat: A square pairwise distance matrix of shape (m, m),
    symmetric and with zeroes along the diagonal.""",
        ":param nu: A one-dimensional vector of length m with strictly positive entries.",
        _exp_absorb_cutoff_docstring,
        _safe_for_exp_docstring,
        _tol_sinkhorn_docstring,
        _tol_outerloop_docstring,
        _mass_kept_docstring,
        _gw_cost_docstring,
        """:return: A Numpy array `ret` of floats of shape (5,),
    where `ret[0]` is the GW cost of the transport plan found,
    `ret[1]` is the first marginal penalty (not including the rho1 scaling factor),
    `ret[2]` is the second marginal penalty (not including the rho2 scaling factor),
    `ret[3]` is the entropic regularization cost (not including the scaling factor epsilon),
    `ret[4]` is the total UGW_eps cost (the appropriate weighted linear
    combination of the first four entries)""",
    ]
)

_ugw_pairwise_docstring = "\n".join(
    [
        """Given an array of squareform distance matrices of shape (k,n,n),
and an array of measures of shape (k,n), compute the pairwise unbalanced Gromov-Wasserstein
distance between all of them. Other than replacing two distance matrices with an array of
distance matrices, and two distributions with an array of distributions, parameters are as
in the function ugw_armijo.""",
        "",
        """Because the coefficient rho is difficult to interpret and choose, the default
behavior of the algorithm is to estimate an appropriate value of rho based
on the following heuristic. The user selects a lower bound "mass_kept" for the
fraction of the mass that they want to keep between two cells, and the algorithm
randomly computes many GW distances between
cells in the observed data to estimate the quantile of observed distances specified by
the `quantile` parameter.
Using this statistic, a value of rho is calculated which guarantees that
for cell pairs whose GW cost is below that threshold, the optimal UGW
transport plan will keep at least fraction "mass_kept" of the mass. This will
add some overhead to the algorithm because of the time to
compute the GW values.""",
        "",
        """:param mass_kept: A real number between 0 and 1, the minimum fraction of mass
    to be preserved by UGW transport plans between two cells in the same neighborhood.""",
        _eps_docstring,
        """:param dmats: An array of squareform distance matrices of shape (k,n,n),
    where k is the number of cells and n is the number of points sampled from each cell.
    Alternatively, dmats can be a string coding a filepath to a csv file containing
    icdms, in the file format established in cajal.sample_swc, cajal.sample_seg, etc.""",
        """:param distrs: An array of measures of shape (k,n),
    where k is the number of cells and n is the number of points per cell. If distrs is None,
    then the uniform distribution on all cells will be taken.""",
        """:param quantile: A real number between 0 and 1, the quantile in the
    distribution of distances which informs the notion of "same neighborhood"
    referred to in the parameter mass_kept.""",
        _increasing_ratio_docstring,
        """:param sample_size: In order to estimate the appropriate value of rho,
    before initiating the unbalanced GW computations, the function randomly
    computes sample_size many GW distances to estimate the quantile of the
    distribution specified by the parameter "quantile".""",
        """:param rho: If this is specified, the sampling routine and estimation
    of the appropriate rho for the target mass_kept is ignored, and the
    given value of rho is used for all cell pairs.""",
        """:param as_matrix: Return only the final distance matrix, as opposed to
    the full account of intermediary values.
    If as_matrix is False, return a pair (distances, rho) where rho was the value
    found for the given mass, and distances is an array of shape (k * (k-1)/2,5) where
    the rows correspond to pairs (i,j) of cells with i < j, and the columns are as in
    ugw_armijo.""",
        """:param pt_cloud: If `pt_cloud` is `True` then `dmats` will be interpreted,
    not as an array of distance matrices, but as an array of point clouds,
    of shape `(k, n, d)` where `d` is the dimension of the ambient Euclidean space
    the points are sampled from. This may improve performance for the core portion of
    the computation as computing distances between points may be faster than looking up the
    precomputed distances in memory.""",
        "",
        """All other values are as in ugw_armijo.""",
    ]
)

# _ugw_armijo_pairwise_unif_docstring = """This is the same as ugw_armijo_pairwise,
# but with all distributions hardcoded to the uniform distribution.
# May reduce memory usage and cache usage relative to storing the entire constant array."""

_ugw_armijo_pairwise_increasing_docstring = """This is a post-processing step for
ugw_armijo_pairwise. It loops through the output and identifies all pairs of matrices
where the algorithm failed to converge, and re-runs the computation for those inputs
repeatedly at exponentially increasing values of the parameter epsilon (scaled by
increasing_ratio) until the algorithm stabilizes. This is step is useful for situations
where you want a complete picture of the whole space of cells even if some of the
UGW values are slightly off due to the increased regularization parameter.

:param ugw_dmat: The output of the UGW algorithm, an array of shape
    (n * (n-1)/2, 5), where n is the number of cells in the data set;
    possibly containing NaN values.
:param increasing_ratio: The multiple to increase epsilon by each time the
    function fails to converge.

Other parameters are as in ugw_armijo_pairwise.
"""

_ugw_armijo_euclidean_docstring = """The core of this function
is similar to ugw_armijo_pairwise_unif,
but the user passes in the array of point clouds rather than distance matrices,
and the backend computes the distance matrices dynamically at the moment they
are needed. This may reduce memory consumption and cache usage. pt
(of shape `(k,n,d)`, where `k` is the
number of cells, `n` is the number of points per cell, and `d` is the dimension of the
ambient Euclidean space that the points live in).

Note that the `ugw_armijo_pairwise` function wraps this one while
also providing some additional preprocessing and postprocessing,
so that function may be more convenient.
"""


class UGW(Futhark):
    """The UGW class provides a wrapper around a C library with methods for
    efficiently computing the unbalanced Gromov-Wasserstein distance
    between two cells or the pairwise distance between all cells in an array.

    To use this module, call the UGW class constructor with the desired
    backend module as its argument; the object returned by the constructor
    carries the connection to the library, and the unbalanced GW functions are
    accessible as methods of this object. If the user wants to
    parallelize at the level of Python processes, then different Python processes
    should each instantiate the class separately,
    """

    # These boilerplate function definitions are the only solution I could come up with
    # to the problem that the C backend does not support named arguments/keyword arguments
    # at all, they have to be strictly positional.
    # Also, the C backend doesn't support default arguments.
    # IMO keyword arguments make it a lot easier to
    # read and understand code, so this is worth it just for the UI difference of being able
    # to specify what each parameter means by name, also helps users to read and follow the
    # tutorials.

    def ugw_armijo(
        self,
        rho1: float,
        rho2: float,
        eps: float,
        A_dmat: DistanceMatrix,
        mu: Distribution,
        B_dmat: DistanceMatrix,
        nu: Distribution,
        exp_absorb_cutoff: float = 1e100,
        safe_for_exp: float = 100.0,
        tol_sinkhorn: float = 1e-4,
        tol_outerloop: float = 0.4,
        mass_kept: Optional[float] = None,
        gw_cost: Optional[float] = None,
    ):
        _ugw_armijo_docstring

        if mass_kept is not None:
            if gw_cost is None:
                raise Exception("If mass_kept is not None, you must supply a GW cost.")
            rho1 = rho_of(gw_cost, mass_kept)
            rho2 = rho1

        self._ugw_armijo(
            self,
            rho1,
            rho2,
            eps,
            A_dmat,
            mu,
            B_dmat,
            nu,
            exp_absorb_cutoff,
            safe_for_exp,
            tol_sinkhorn,
            tol_outerloop,
        )

    ugw_armijo.__doc__ = _ugw_armijo_docstring

    def ugw_armijo_pairwise_increasing(
        self,
        ugw_dmat: npt.NDArray[np.float64],
        increasing_ratio: float,
        rho: float,
        eps: float,
        dmats: Union[str, npt.NDArray[np.float64]],
        distrs: npt.NDArray[np.float64],
        exp_absorb_cutoff: float = 1e100,
        safe_for_exp: float = 100.0,
        tol_sinkhorn: float = 1e-4,
        tol_outerloop: float = 0.4,
    ):
        if isinstance(dmats, str):
            _, icdms = zip(*cell_iterator_csv(dmats))
            dmats = np.stack(icdms, axis=0)
        return self._ugw_armijo_pairwise_increasing(
            ugw_dmat,
            increasing_ratio,
            rho,
            rho,
            eps,
            dmats,
            distrs,
            exp_absorb_cutoff,
            safe_for_exp,
            tol_sinkhorn,
            tol_outerloop,
        )

    ugw_armijo_pairwise_increasing.__doc__ = _ugw_armijo_pairwise_increasing_docstring

    def ugw_armijo_pairwise(
        self,
        mass_kept: float,
        eps: float,
        dmats: Union[str, npt.NDArray[np.float64]],
        distrs: Optional[npt.NDArray[np.float64]] = None,
        quantile: float = 0.15,
        increasing_ratio: Optional[float] = 1.1,
        sample_size: int = 100,
        exp_absorb_cutoff: float = 1e100,
        safe_for_exp: float = 100.0,
        tol_sinkhorn: float = 1e-4,
        tol_outerloop: float = 0.4,
        rho: Optional[float] = None,
        as_matrix=True,
        pt_cloud=False,
    ):

        if pt_cloud:
            pt_clouds = dmats
            dmats = np.stack(
                [squareform(pdist(a), force="tomatrix") for a in pt_clouds], axis=0
            )
        if isinstance(dmats, str):
            _, icdms = zip(*cell_iterator_csv(dmats))
            dmats = np.stack(list(icdms), axis=0)

        if rho is None:
            gw_cost = estimate_distr(dmats, sample_size, quantile)
            rho = rho_of(gw_cost, mass_kept)

        if distrs is None:
            if not pt_cloud:  # dmats is distance matrices
                ugw_dmat = self._ugw_armijo_pairwise_unif(
                    rho,
                    rho,
                    eps,
                    dmats,
                    exp_absorb_cutoff,
                    safe_for_exp,
                    tol_sinkhorn,
                    tol_outerloop,
                )
                print("Done first pass, cleaning up errors.")
            else:  # dmats is pt_clouds
                ugw_dmat = self._ugw_armijo_euclidean(
                    rho,
                    rho,
                    eps,
                    pt_clouds,
                    exp_absorb_cutoff,
                    safe_for_exp,
                    tol_sinkhorn,
                    tol_outerloop,
                )
        else:
            if pt_cloud:
                raise NotImplementedError(
                    "Currently, pt_cloud can only be true if distrs is None.\
                                     The only reason this feature is omitted is to avoid some\
                                     code bloat; this is easily fixable so contact us if you\
                                     need this feature."
                )
            ugw_dmat = self._ugw_armijo_pairwise(
                rho,
                rho,
                eps,
                dmats,
                distrs,
                exp_absorb_cutoff,
                safe_for_exp,
                tol_sinkhorn,
                tol_outerloop,
            )

        if increasing_ratio is not None:
            if distrs is None:
                u = uniform(dmats.shape[1])
                distrs = np.stack([u for _ in range(dmats.shape[0])])
            ugw_dmat = self._ugw_armijo_pairwise_increasing(
                ugw_dmat,
                increasing_ratio,
                rho,
                rho,
                eps,
                dmats,
                distrs,
                exp_absorb_cutoff,
                safe_for_exp,
                tol_sinkhorn,
                tol_outerloop,
            )

        if as_matrix:
            ugw_dmat = self.from_futhark(ugw_dmat)
            return squareform(ugw_dmat[:, 0] + rho * (ugw_dmat[:, 1] + ugw_dmat[:, 2]))
        else:
            return (ugw_dmat, rho)

    ugw_armijo_pairwise.__doc__ = _ugw_pairwise_docstring

    def ugw_armijo_euclidean(
        self,
        rho: float,
        eps: float,
        pt_clouds: npt.NDArray[np.float64],
        exp_absorb_cutoff: float = 1e100,
        safe_for_exp: float = 100.0,
        tol_sinkhorn: float = 1e-4,
        tol_outerloop: float = 0.4,
    ):
        return self._ugw_armijo_euclidean(
            rho,
            rho,
            eps,
            pt_clouds,
            exp_absorb_cutoff,
            safe_for_exp,
            tol_sinkhorn,
            tol_outerloop,
        )

    ugw_armijo_euclidean.__doc__ = _ugw_armijo_euclidean_docstring

    def __init__(self, backend):
        Futhark.__init__(self, backend)
        self._ugw_armijo = self.ugw_armijo
        self._ugw_armijo_pairwise = self.ugw_armijo_pairwise
        self._ugw_armijo_pairwise_unif = self.ugw_armijo_pairwise_unif
        self._ugw_armijo_pairwise_increasing = self.ugw_armijo_pairwise_increasing
        self._ugw_armijo_euclidean = self.ugw_armijo_euclidean

        self.ugw_armijo = MethodType(UGW.ugw_armijo, self)
        self.ugw_armijo_pairwise = MethodType(UGW.ugw_armijo_pairwise, self)
        # self.ugw_armijo_pairwise_unif = MethodType(UGW.ugw_armijo_pairwise_unif, self)
        self.ugw_armijo_pairwise_increasing = MethodType(
            UGW.ugw_armijo_pairwise_increasing, self
        )
        self.ugw_armijo_euclidean = MethodType(UGW.ugw_armijo_euclidean, self)
