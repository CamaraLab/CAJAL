from futhark_ffi import Futhark

from inspect import Parameter, Signature
from cajal.run_gw import DistanceMatrix
from cajal.run_gw import Distribution
import numpy.typing as npt
import numpy as np

rho1_parameter = Parameter(name='rho1',kind=Parameter.POSITIONAL_OR_KEYWORD,annotation=float)
rho2_parameter = Parameter(name='rho2',kind=Parameter.POSITIONAL_OR_KEYWORD,annotation=float)
eps_parameter = Parameter(name='eps',kind=Parameter.POSITIONAL_OR_KEYWORD,annotation=float)
A_parameter = Parameter(name='A',kind=Parameter.POSITIONAL_OR_KEYWORD,annotation=DistanceMatrix)
mu_parameter = Parameter(name='mu',kind=Parameter.POSITIONAL_OR_KEYWORD,annotation=Distribution)
B_parameter = Parameter(name='B',kind=Parameter.POSITIONAL_OR_KEYWORD,annotation=DistanceMatrix)
nu_parameter = Parameter(name='nu',kind=Parameter.POSITIONAL_OR_KEYWORD,annotation=Distribution)
exp_absorb_cutoff_parameter = Parameter(name='exp_absorb_cutoff',kind=Parameter.POSITIONAL_OR_KEYWORD,annotation=float)
safe_for_exp_parameter = Parameter(name='safe_for_exp',kind=Parameter.POSITIONAL_OR_KEYWORD,annotation=float)
tol_sinkhorn_parameter = Parameter(name='tol_sinkhorn',kind=Parameter.POSITIONAL_OR_KEYWORD,annotation=float)
tol_outerloop_parameter = Parameter(name='tol_outerloop',kind=Parameter.POSITIONAL_OR_KEYWORD,annotation=float)

ugw_armijo_sig = Signature(parameters=[
    Parameter(name='rho1',kind=Parameter.POSITIONAL_OR_KEYWORD,annotation=float),
    Parameter(name='rho2',kind=Parameter.POSITIONAL_OR_KEYWORD,annotation=float),
    Parameter(name='eps',kind=Parameter.POSITIONAL_OR_KEYWORD,annotation=float),
    Parameter(name='A',kind=Parameter.POSITIONAL_OR_KEYWORD,annotation=DistanceMatrix),
    Parameter(name='mu',kind=Parameter.POSITIONAL_OR_KEYWORD,annotation=Distribution),
    Parameter(name='B',kind=Parameter.POSITIONAL_OR_KEYWORD,annotation=DistanceMatrix),
    Parameter(name='nu',kind=Parameter.POSITIONAL_OR_KEYWORD,annotation=Distribution),
    Parameter(name='exp_absorb_cutoff',kind=Parameter.POSITIONAL_OR_KEYWORD,annotation=float),
    Parameter(name='safe_for_exp',kind=Parameter.POSITIONAL_OR_KEYWORD,annotation=float),
    Parameter(name='tol_sinkhorn',kind=Parameter.POSITIONAL_OR_KEYWORD,annotation=float),
    Parameter(name='tol_outerloop',kind=Parameter.POSITIONAL_OR_KEYWORD,annotation=float)
],
    return_annotation=npt.NDArray[np.float64])

rho1_docstring = ":param rho1: The first marginal penalty coefficient, controls how much the first marginal for the transport plan (sum along rows) is allowed to deviate from mu. Higher is more strict and should give closer marginals."
rho2_docstring = ":param rho2: The second marginal penalty coefficient, controls how much the second marginal for the transport plan (sum along columns) is allowed to deviate from nu."
eps_docstring = ":param eps: The entropic regularization coefficient. Increasing this value makes the problem more convex and the algorithm will converge to an answer faster, but it may be inaccurate (far from the true minimum).\
    If it's set too low there will be numerical instability issues and the function should return NaN."
exp_absorb_cutoff_docstring= """:param exp_absorb_cutoff: A numerical stability parameter. Suggest 1e100 by default. The inner loop, the Sinkhorn algorithm, tries to solve an optimization problem of solving for diagonal matrices A, B to minimize a cost function Cost(AKB) where the matrix K is given.
    The values of A, K, B are numerically extreme under normal conditions and so we represent
    A by a pair of vectors (a_bar, u_bar) where a_bar is between exp_absorb_cutoff and 1/exp_absorb_cutoff, and diag(A) = a_bar * e^u_bar. When a_bar exceeds exp_absorb_cutoff, part of a_bar is "absorbed into the exponent" u_bar, thus the name.
    Set this to any value such that floating point arithmetic operations in the range (1/exp_absorb_cutoff, exp_absorb_cutoff) are reasonably accurate and not too close to overflow."""
safe_for_exp_docstring=""":param safe_for_exp: A numerical stability parameter. Suggest 100 as a default. This is used only once during the initialization of the algorithm, the user supplies a number R and the code tries to construct an initial starting matrix K with the property that e^R is an upper bound for all elements in K,
     and as many as possible of the values of K are "pushed up against" this upper bound. The main risk here is of *underflow* of values in K; by increasing the values in K we give ourselves more room to maneuver with the spectrum of values that floats can represent. Choose a number such that e^R is big 
     but still is a comfortable distance from the max float value, a good region for addition and multiplication without risk of overflow."""
tol_sinkhorn_docstring=""":param tol_sinkhorn: An accuracy parameter, controls the tolerance at which the inner loop exits. Suggest 10^-5 to 10^-9.
    The inner loop, the Sinkhorn algorithm, tries to solve an optimization problem of solving for diagonal matrices A, B to minimize a cost function Cost(AKB) where the matrix K is given. This gives a series of iterations A_n, B_n, A_n+1, B_n+1, ...
    The exit condition for the loop is when diag(A_n+1)/diag(A_n) is everywhere within 1 +/- tol_sinkhorn."""
tol_outerloop_docstring=""":param tol_outerloop: An accuracy parameter, controls the tolerance at which the outer loop exits - when the ratio T_n+1/T_n is everywhere within 1 +/- tol_outerloop. (T_n is the nth transport plan found in the main gradient descent)"""
# UGW_Multicore.ugw_armijo.__signature__=ugw_armijo_sig

ugw_armijo_docstring =\
"""
Given two metric measure spaces (A, mu) and (B, nu), compute the unbalanced Gromov-Wasserstein distance between them.

:param rho1: The first marginal penalty coefficient, controls how much the first marginal for the transport plan (sum along rows) is allowed to deviate from mu. Higher is more strict and should give closer marginals.
:param rho2: The second marginal penalty coefficient, controls how much the second marginal for the transport plan (sum along columns) is allowed to deviate from nu.
:param eps: The entropic regularization coefficient. Increasing this value makes the problem more convex and the algorithm will converge to an answer faster, but it may be inaccurate (far from the true minimum).
    If it's set too low there will be numerical instability issues and the function should return NaN.
:param A: A square pairwise distance matrix of shape (n, n), symmetric and with zeroes along the diagonal.
:param mu: A one-dimensional vector of length n with strictly positive entries. (The algorithm does not currently support zero entries because this would be annoying to deal with.)
:param B: A square pairwise distance matrix of shape (m, m), symmetric and with zeroes along the diagonal.
:param nu: A one-dimensional vector of length m with strictly positive entries.
:param exp_absorb_cutoff: A numerical stability parameter. Suggest 1e100 by default. The inner loop, the Sinkhorn algorithm, tries to solve an optimization problem of solving for diagonal matrices A, B to minimize a cost function Cost(AKB) where the matrix K is given.
    The values of A, K, B are numerically extreme under normal conditions and so we represent
    A by a pair of vectors (a_bar, u_bar) where a_bar is between exp_absorb_cutoff and 1/exp_absorb_cutoff, and diag(A) = a_bar * e^u_bar. When a_bar exceeds exp_absorb_cutoff, part of a_bar is "absorbed into the exponent" u_bar, thus the name.
    Set this to any value such that floating point arithmetic operations in the range (1/exp_absorb_cutoff, exp_absorb_cutoff) are reasonably accurate and not too close to overflow.
:param safe_for_exp: A numerical stability parameter. Suggest 100 as a default. This is used only once during the initialization of the algorithm, the user supplies a number R and the code tries to construct an initial starting matrix K with the property that e^R is an upper bound for all elements in K,
     and as many as possible of the values of K are "pushed up against" this upper bound. The main risk here is of *underflow* of values in K; by increasing the values in K we give ourselves more room to maneuver with the spectrum of values that floats can represent. Choose a number such that e^R is big 
     but still is a comfortable distance from the max float value, a good region for addition and multiplication without risk of overflow.
:param tol_sinkhorn: An accuracy parameter, controls the tolerance at which the inner loop exits. Suggest 10^-5 to 10^-9.
    The inner loop, the Sinkhorn algorithm, tries to solve an optimization problem of solving for diagonal matrices A, B to minimize a cost function Cost(AKB) where the matrix K is given. This gives a series of iterations A_n, B_n, A_n+1, B_n+1, ...
    The exit condition for the loop is when diag(A_n+1)/diag(A_n) is everywhere within 1 +/- tol_sinkhorn.
:param tol_outerloop: An accuracy parameter, controls the tolerance at which the outer loop exits - when the ratio T_n+1/T_n is everywhere within 1 +/- tol_outerloop. (T_n is the nth transport plan found in the main gradient descent)
:return: A Numpy array ret of floats of shape (5,), 
    where ret[0] is the GW cost of the transport plan found, 
    ret[1] is the first marginal penalty (not including the rho1 scaling factor),
    ret[2] is the second marginal penalty (not including the rho2 scaling factor),
    ret[3] is the entropic regularization cost (not including the scaling factor epsilon),
    ret[4] is the total UGW_eps cost (the appropriate weighted linear combination of the first four entries)
"""
# UGW_Multicore.ugw_armijo.__doc__ = ugw_armijo_docstring


class UGW(Futhark):
    def __init__(backend_module):
        Futhark.__init__(backend_module)
        self.ugw_armijo