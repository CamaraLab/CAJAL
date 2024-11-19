from futhark_ffi import Futhark
from types import MethodType
from cajal.run_gw import DistanceMatrix
from cajal.run_gw import Distribution
import numpy.typing as npt
import numpy as np

_rho1_docstring = """:param rho1: The first marginal penalty coefficient, controls how much the
    first marginal for the transport plan (sum along rows) is allowed to deviate from mu.
    Higher is more strict and should give closer marginals."""
_rho2_docstring = """:param rho2: The second marginal penalty coefficient,
    controls how much the second marginal for the transport plan (sum along columns)
    is allowed to deviate from nu."""
_eps_docstring = """:param eps: The entropic regularization coefficient. Increasing this value
    makes the problem more convex and the algorithm will converge to an answer faster,
    but it may be inaccurate (far from the true minimum).
    If it's set too low there will be numerical instability issues and the function should
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

_ugw_armijo_docstring = "\n".join(
    [
        """Given two metric measure spaces (A,mu) and (B, nu), compute the
        unbalanced Gromov-Wasserstein distance between them,
        using an algorithm based on that of Séjourné, Vialard and Peyré,
        with some modifications for numerical stability and convergence.""",
        _rho1_docstring,
        _rho2_docstring,
        _eps_docstring,
        """:param A: A square pairwise distance matrix of shape (n, n),
            symmetric and with zeroes along the diagonal.""",
        """:param mu: A one-dimensional vector of length n with strictly
            positive entries. (The algorithm does not currently support zero entries.)""",
        """:param B: A square pairwise distance matrix of shape (m, m),
            symmetric and with zeroes along the diagonal.""",
        ":param nu: A one-dimensional vector of length m with strictly positive entries.",
        _exp_absorb_cutoff_docstring,
        _safe_for_exp_docstring,
        _tol_sinkhorn_docstring,
        _tol_outerloop_docstring,
        """:return: A Numpy array ret of floats of shape (5,),
    where ret[0] is the GW cost of the transport plan found,
    ret[1] is the first marginal penalty (not including the rho1 scaling factor),
    ret[2] is the second marginal penalty (not including the rho2 scaling factor),
    ret[3] is the entropic regularization cost (not including the scaling factor epsilon),
    ret[4] is the total UGW_eps cost (the appropriate weighted linear
        combination of the first four entries)
""",
    ]
)

_ugw_pairwise_docstring = """Given an array of squareform distance matrices of shape (k,n,n),
    and an array of measures of shape (k,n), compute the pairwise unbalanced Gromov-Wasserstein
    distance between all of them. Other than replacing two distance matrices with an array of
    distance matrices, and two distributions with an array of distributions, parameters are as
    in the function ugw_armijo. Returns an array of shape (k * (k-1)/2, 5) where the rows
    correspond to cell pairs and the columns are as in ugw_armijo. One can get the matrix of
    UGW_eps costs by accessing column 4 and applying the scipy.spatial.distance.squareform
    function. The user should be aware that a few NaN's in the output matrix are likely, and they
    should try evaluating the UGW for those specific cell pairs at higher values of epsilon. A
    GW cost of zero in the first column is problematic and potentially indicates a numerical
    stability problem."""

_ugw_armijo_pairwise_unif_docstring = """This is the same as ugw_armijo_pairwise,
but with all distributions hardcoded to the uniform distribution.
May reduce memory usage and cache usage relative to storing the entire constant array."""

_ugw_armijo_euclidean_docstring = """This is the same as ugw_armijo_pairwise_unif,
    but the user passes in the array of point clouds rather than distance matrices,
    and the backend computes the distance matrices dynamically at the moment they
    are needed. This may reduce memory consumption and cache usage."""


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

    At the moment the only documented functions are ugw_armijo, ugw_armijo_pairwise,
    ugw_armijo_pairwise_unif, and ugw_armijo_euclidean.
    The module defines other algorithms
    which may be faster under some circumstances but we have counterexamples
    where these other algorithms fail to converge and so we don't encourage their use.
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
        tol_sinkhorn: float = 1e-5,
        tol_outerloop: float = 1.0,
    ):
        _ugw_armijo_docstring

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

    def ugw_armijo_pairwise(
        self,
        rho1: float,
        rho2: float,
        eps: float,
        dmats: npt.NDArray[np.float64],
        distrs: npt.NDArray[np.float64],
        exp_absorb_cutoff: float = 1e100,
        safe_for_exp: float = 100.0,
        tol_sinkhorn: float = 1e-5,
        tol_outerloop: float = 1e-3,
    ):
        return self._ugw_armijo_pairwise(
            self,
            rho1,
            rho2,
            eps,
            dmats,
            distrs,
            exp_absorb_cutoff,
            safe_for_exp,
            tol_sinkhorn,
            tol_outerloop,
        )

    ugw_armijo_pairwise.__doc__ = _ugw_pairwise_docstring

    def ugw_armijo_pairwise_unif(
        self,
        rho1: float,
        rho2: float,
        eps: float,
        dmats: npt.NDArray[np.float64],
        exp_absorb_cutoff: float = 1e100,
        safe_for_exp: float = 100.0,
        tol_sinkhorn: float = 1e-5,
        tol_outerloop: float = 1e-3,
    ):
        return self._ugw_armijo_pairwise_unif(
            rho1,
            rho2,
            eps,
            dmats,
            exp_absorb_cutoff,
            safe_for_exp,
            tol_sinkhorn,
            tol_outerloop,
        )

    ugw_armijo_pairwise_unif.__doc__ = _ugw_armijo_pairwise_unif_docstring

    def ugw_armijo_euclidean(
        self,
        rho1: float,
        rho2: float,
        eps: float,
        pt_clouds: npt.NDArray[np.float64],
        exp_absorb_cutoff: float = 1e100,
        safe_for_exp: float = 100.0,
        tol_sinkhorn: float = 1e-5,
        tol_outerloop: float = 1e-3,
    ):
        return self._ugw_armijo_euclidean(
            rho1,
            rho2,
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
        self._ugw_armijo_euclidean = self.ugw_armijo_euclidean

        self.ugw_armijo = MethodType(UGW.ugw_armijo, self)
        self.ugw_armijo_pairwise = MethodType(UGW.ugw_armijo_pairwise, self)
        self.ugw_armijo_pairwise_unif = MethodType(UGW.ugw_armijo_pairwise_unif, self)
        self.ugw_armijo_euclidean = MethodType(UGW.ugw_armijo_euclidean, self)
