# cython: profile=True
# distutils: language=c++
# distutils: sources = src/cajal/EMD_wrapper.cpp

# This code is closely modelled on code from the Python Optimal Transport library, https://github.com/PythonOT/POT, and implements the gradient descent algorithm from Peyre et al. ICML 2016 "Gromov-Wasserstein averaging of Kernel and Distance matrices." Thanks very much to Rémi Flamary. 

"""
Cython implementations of the Gromov-Wasserstein distance between metric measure spaces by gradient descent.
"""

cimport cython
import numpy as np
cimport numpy as np
import scipy
import warnings
np.import_array()
from libc.stdlib cimport rand
from libc.stdint cimport uint64_t
# from ot.lp import emd_c, emd
from math import sqrt
from scipy.sparse import lil_matrix
from scipy import sparse

cdef extern from "EMD.h":
    int EMD_wrap(int n1, int n2, double *X, double *Y, double *D, double *G, double* alpha, double* beta, double *cost, uint64_t maxIter) nogil
    cdef enum ProblemType: INFEASIBLE, OPTIMAL, UNBOUNDED, MAX_ITER_REACHED
# cdef extern from "EMD.h":
#     int EMD_wrap(int n1,int n2, double *X, double *Y,double *D, double *G, double* alpha, double* beta, double *cost, uint64_t maxIter) nogil
#     int EMD_wrap_omp(int n1,int n2, double *X, double *Y,double *D, double *G, double* alpha, double* beta, double *cost, uint64_t maxIter, int numThreads) nogil
#     cdef enum ProblemType: INFEASIBLE, OPTIMAL, UNBOUNDED, MAX_ITER_REACHED


cdef extern from "stdlib.h":
    int RAND_MAX

DTYPE=np.float64
ctypedef np.float_t DTYPE_t


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def matrix_tensor(
        np.ndarray[DTYPE_t,ndim=2] c_C_Cbar,
        np.ndarray[DTYPE_t,ndim=2] C,
        np.ndarray[DTYPE_t,ndim=2] Cbar,
        np.ndarray[DTYPE_t,ndim=2] T,
        np.ndarray[DTYPE_t,ndim=2] LCCbar_otimes_T):

    np.matmul(C,T,out=LCCbar_otimes_T)
    # Cbar should be equal to Cbar.T so the transpose here is unnecessary.
    # assert np.all(Cbar == Cbar.T)
    np.matmul(LCCbar_otimes_T,Cbar,out=LCCbar_otimes_T)
    np.multiply(LCCbar_otimes_T,2.,out=LCCbar_otimes_T)
    np.subtract(c_C_Cbar,LCCbar_otimes_T,out=LCCbar_otimes_T)

# def frobenius(DTYPE_t[:,:] A, DTYPE_t[:,:] B) -> DTYPE_t:

#     cdef int n = A.shape[0]
#     cdef int m = A.shape[1]
#     assert n==B.shape[0]
#     assert m==B.shape[1]
#     cdef DTYPE_t sumval = 0.0
#     cdef int i, j
    
#     for i in range(A.shape[0]):
#         for j in range(A.shape[1]):
#             sumval+=A[i,j]*B[i,j]
#     return sumval


def n_c_2(int n):
    return <int>((n * (n-1))/2)


class GW_cell:
    dmat : npt.NDArray[np.float_] # Squareform distance matrix
    distribution : npt.NDArray[np.float_] # Probability distribution
    dmat_dot_dist : npt.NDArray[DTYPE] # Matrix-vector product of the distance
                                       # matrix with the probability
                                       # distribution
    cell_constant : float # ((A * A) @ a) @ a
    def __init__(self,dmat,distribution):
        self.dmat = dmat
        self.distribution = distribution
        self.dmat_dot_dist = dmat @ distribution
        self.cell_constant = ((dmat * dmat) @ distribution) @ distribution

cpdef gw_cython_init_cost(
    np.ndarray[DTYPE_t,ndim=2,mode='c'] A,
    np.ndarray[DTYPE_t,ndim=1,mode='c'] a,
    DTYPE_t c_A,
    np.ndarray[DTYPE_t,ndim=2,mode='c'] B,
    np.ndarray[DTYPE_t,ndim=1,mode='c'] b,
    DTYPE_t c_B,
    np.ndarray[DTYPE_t,ndim=2,mode='c'] C,
    int max_iters_descent =1000,
    uint64_t max_iters_ot = 200000,
):

    cdef int it = 0
    cdef int n = a.shape[0]
    cdef int m = b.shape[0]
    cdef int result_code
    cdef DTYPE_t cost=0.0
    cdef DTYPE_t newcost=0.0
    cdef np.ndarray[double, ndim=1, mode="c"] alpha=np.zeros(n)
    cdef np.ndarray[double, ndim=1, mode="c"] beta=np.zeros(m)
    cdef np.ndarray[DTYPE_t, ndim=2, mode="c"] neg2_PB
    cdef np.ndarray[double, ndim=2, mode="c"] AP=np.zeros((n,m),dtype=DTYPE,order='C')
    cdef np.ndarray[np.float64_t,ndim=2,mode='c'] P = np.zeros((n,m),dtype=DTYPE,order='C')
    cost=c_A+c_B
    cdef double temp=0.0
    cost+=float(np.tensordot(C,P))

    while it<max_iters_descent:
        result_code=EMD_wrap(n,m, <double*> a.data, <double*> b.data,
                             <double*> C.data, <double*>P.data,
                             <double*> alpha.data, <double*> beta.data,
                             <double*> &temp, max_iters_ot)

        if result_code != OPTIMAL:
            # cdef enum ProblemType: INFEASIBLE, OPTIMAL, UNBOUNDED, MAX_ITER_REACHED
            if result_code == INFEASIBLE:
                raise Exception("INFEASIBLE")
            if result_code == UNBOUNDED:
                raise Exception("UNBOUNDED")
            if result_code == MAX_ITER_REACHED:
                raise Warning("MAX_ITER_REACHED")
            
        # P_sparse = scipy.sparse.csc_matrix(P,shape=(n,m), dtype=DTYPE)
        np.dot(A,P,out=AP)
        np.multiply(AP,-2.0,out=AP)
        np.matmul(AP,B,out=C)
        newcost=c_A+c_B
        newcost+=float(np.tensordot(C,P))
        if newcost >= cost:
            cost = max(cost,0)
            return (P,sqrt(cost)/2.0)
        cost=newcost 
        it+=1
    

cpdef gw_cython_core(
        np.ndarray[DTYPE_t,ndim=2,mode='c'] A,
        np.ndarray[DTYPE_t,ndim=1,mode='c'] a,
        np.ndarray[DTYPE_t,ndim=1,mode='c'] Aa,
        DTYPE_t c_A,
        np.ndarray[DTYPE_t,ndim=2,mode='c'] B,
        np.ndarray[DTYPE_t,ndim=1,mode='c'] b,
        np.ndarray[DTYPE_t,ndim=1,mode='c'] Bb,
        DTYPE_t c_B,
        int max_iters_descent =1000,
        uint64_t max_iters_ot = 200000
):
    """
    :param A: A squareform distance matrix.
    :param a: A probability distribution on points of A.
    :param Aa: Should be equal to the matrix-vector product A@a.
    :param c_A: Should be equal to the scalar ((A * A)@a)@a.
    :param B: A squareform distance matrix.
    :param b: A probability distribution on points of B.
    :param Bb: Should be equal to the matrix-vector product B@b.
    :param c_B: Should be equal to the scalar ((B * B)@b)@b.
    :return: A pair (P, gw_dist) where P is a transport plan and gw_dist is the associated cost.
    """

    cdef np.ndarray[np.float64_t,ndim=2,mode='c'] C = np.multiply(Aa[:,np.newaxis],(-2.0*Bb)[np.newaxis,:],order='C')
    return gw_cython_init_cost(
        A,
        a,
        c_A,
        B,
        b,
        c_B,
        C,
        max_iters_descent,
        max_iters_ot)

def gw(
    A: DistanceMatrix,
    a: Distribution,
    B: DistanceMatrix,
    b: Distribution,
    max_iters_descent: int = 1000,
    max_iters_ot: int = 200000,
) -> tuple[Matrix, float]:
    """Compute the Gromov-Wasserstein distance between two metric measure spaces."""
    Aa = A @ a
    c_A = ((A * A) @ a) @ a
    Bb = B @ b
    c_B = ((B * B) @ b) @ b
    return gw_cython_core(A, a, Aa, c_A, B, b, Bb, c_B, max_iters_descent, max_iters_ot)

def gw_pairwise(
        list cell_dms           # A list of GW_cells.
):
    """
    :param cell_dms: A list of GW_cells.
    :return: a vectorform GW inter-cell (not intra-cell) distance matrix, as a numpy array
    """

    cdef Py_ssize_t i = 0
    cdef Py_ssize_t j = 0
    cdef Py_ssize_t k = 0
    cdef int N = len(cell_dms)
    
    # cdef list[double] gw_dists= ((N * (N-1))/2)*[0.0]
    cdef np.ndarray[DTYPE_t,ndim=1,mode='c'] gw_dists = np.zeros( (int((N * (N-1))/2),),dtype=DTYPE)
    cdef double gw_dist

    cdef np.ndarray[DTYPE_t,ndim=2,mode='c'] A, B
    cdef np.ndarray[DTYPE_t,ndim=1,mode='c'] a, b
    cdef np.ndarray[DTYPE_t,ndim=1,mode='c'] Aa, Bb
    cdef DTYPE_t c_A, c_B

    for i in range(N):
        cellA=cell_dms[i]
        A = cellA.dmat
        a = cellA.distribution
        Aa = cellA.dmat_dot_dist
        c_A = cellA.cell_constant
        for j in range(i+1,N):
            cellB=cell_dms[j]
            B = cellB.dmat
            b = cellB.distribution
            Bb = cellB.dmat_dot_dist
            c_B = cellB.cell_constant
            _,gw_dist=gw_cython_core(A,a,Aa,c_A,B,b,Bb,c_B)
            gw_dists[k]=gw_dist
            k+=1
    return gw_dists


def intersection(DTYPE_t a, DTYPE_t b, DTYPE_t c, DTYPE_t d) -> DTYPE_t:
    cdef DTYPE_t maxac= a if a >= c else c
    cdef DTYPE_t minbd= b if b <= d else d
    minbd=minbd-maxac
    minbd = minbd if minbd >= <DTYPE_t>0.0 else <DTYPE_t>0.0
    return minbd

def oneD_ot_CHECK(
        DTYPE_t[:,:] T):

    cdef DTYPE_t mysum=0.0
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            mysum+=T[i,j]
    return mysum

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def oneD_ot_gw(
        DTYPE_t[::1] a,
        int a_len,
        DTYPE_t[::1] b,
        int b_len,
        DTYPE_t[:,:] T,
        DTYPE_t scaling_factor
):

    cdef Py_ssize_t i = 0
    cdef Py_ssize_t j = 0
    cdef DTYPE_t cum_a_prob=0.0
    cdef DTYPE_t cum_b_prob=0.0

    # assert A.shape[0]==a_len
    # assert B.shape[0]==b_len
    # assert a.shape[0]==a_len
    # assert b.shape[0]==b_len
    # assert T.shape[0]==a_len
    # assert T.shape[1]==b_len
    # assert( abs(np.sum(a)-1.0)<1.0e-7 )
    # assert( abs(np.sum(b)-1.0)<1.0e-7 )
    while i+j < a_len+b_len-1:
        # Loop invariant:
        # [cum_a_prob,cum_a_prob+a[i]) intersects [cum_b_prob,cum_b_prob+b[j])
        # nontrivially.
        # assert i<a_len
        # assert j<b_len
        # assert cum_a_prob+a[i]>cum_b_prob and cum_b_prob+b[j]>cum_a_prob
        T[i,j]=intersection(cum_a_prob,
                            cum_a_prob+a[i],
                            cum_b_prob,
                            cum_b_prob+b[j])*scaling_factor
        if cum_a_prob+a[i]<cum_b_prob+b[j]:
            if i==a_len-1:
                # assert j==b_len-1
                break
            else:
                cum_a_prob+=a[i]
                i+=1
        elif cum_a_prob+a[i]>cum_b_prob+b[j]:
            if j==b_len-1:
                # assert i==a_len-1
                break
            else:
                cum_b_prob+=b[j]
                j+=1
        else:
            if i==a_len-1:
                # assert j==b_len-1
                break
            else:
                cum_a_prob+=a[i]
                i+=1
                cum_b_prob+=b[j]
                j+=1

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef sparse_oneD_OT_gw(
    int[::1] T_rows,
    int[::1] T_cols,
    DTYPE_t[::1] T_vals,
    int T_offset,
    int a_offset,
    int b_offset,
    DTYPE_t[::1] a, # 1dim
    int a_len, #int
    DTYPE_t[::1] b,#1dim
    int b_len, #int
    DTYPE_t scaling_factor #float
):
    # a, b are required to be probability distributions
    # The sparse matrix returned by this function may have triples of the form (0,0,0.0).
    # Code handling this should be aware of this.

    cdef Py_ssize_t i = 0
    cdef Py_ssize_t j = 0
    cdef DTYPE_t cum_a_prob=0.0
    cdef DTYPE_t cum_b_prob=0.0
    cdef int index

    # assert A.shape[0]==a_len
    # assert B.shape[0]==b_len
    # assert a.shape[0]==a_len
    # assert b.shape[0]==b_len
    # assert T.shape[0]==a_len
    # assert T.shape[1]==b_len
    # assert( abs(np.sum(a)-1.0)<1.0e-7 )
    # assert( abs(np.sum(b)-1.0)<1.0e-7 )
    while i+j < a_len+b_len-1:
        # print("inner i:" + str(i))
        # print("inner j:" + str(j))
        # Loop invariant:  [cum_a_prob,cum_a_prob+a[i]) intersects [cum_b_prob,cum_b_prob+b[j])
        # nontrivially.
        # assert i<a_len
        # assert j<b_len
        # assert cum_a_prob+a[i]>cum_b_prob and cum_b_prob+b[j]>cum_a_prob
        index=T_offset+i+j
        T_rows[index]= a_offset + i
        T_cols[index ]= b_offset + j
        assert T_vals[index]==0.0
        T_vals[index]=\
            intersection(cum_a_prob,cum_a_prob+a[i],cum_b_prob,cum_b_prob+b[j])*scaling_factor
        if cum_a_prob+a[i]<cum_b_prob+b[j]:
            if i==a_len-1:
                # assert j==b_len-1
                break
            else:
                cum_a_prob+=a[i]
                i+=1
        elif cum_a_prob+a[i]>cum_b_prob+b[j]:
            if j==b_len-1:
                # assert i==a_len-1
                break
            else:
                cum_b_prob+=b[j]
                j+=1
        else:
            if i==a_len-1:
                # assert j==b_len-1
                break
            else:
                cum_a_prob+=a[i]
                i+=1
                cum_b_prob+=b[j]
                j+=1


def qgw_init_cost(
        # a is the probability distribution on points of A
        np.ndarray[np.float64_t,ndim=1] a,
        # A_s=A_sample is a sub_matrix of A, of size ns x ns
        np.ndarray[np.float64_t,ndim=2] A_s,
        # A_si = A_sample_indices
        # Indices for sampled points of A, of length ns+1
        # Should satisfy A_s[x,y]=A[A_si[x],A_si[y]] for all x,y < ns
        # should satisfy A_si[ns]=n
        np.ndarray[Py_ssize_t,ndim=1] A_si,
        # Probability distribution on sample points of A_s; of length ns
        np.ndarray[np.float64_t,ndim=1] a_s,
        DTYPE_t c_As,
        # b is the probability distribution on points of B
        np.ndarray[np.float64_t,ndim=1] b,
        # B_sample, size ms x ms
        np.ndarray[np.float64_t,ndim=2] B_s,
        # B_sample_indices, size ms+1
        np.ndarray[Py_ssize_t,ndim=1] B_si,
        # Probability distribution on sample points of B_s; of length ms
        np.ndarray[np.float64_t,ndim=1] b_s,
        # np.dot(np.multiply(B_s,B_s),b_s)
        DTYPE_t c_Bs,
        # Initial cost matrix of tisze ns x ms
        np.ndarray[np.float64_t,ndim=2] C,
):

    cdef int n = a.shape[0]
    cdef int ns = A_s.shape[0]
    cdef int m = b.shape[0]
    cdef int ms = B_s.shape[0]
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t j = 0
    cdef int a_local_len
    cdef int b_local_len
    cdef np.ndarray[np.float64_t,ndim=2] quantized_coupling # size ns x ms
    
    quantized_coupling, _=gw_cython_init_cost(A_s,a_s,c_As,B_s,b_s,c_Bs,C)
    # We can count, roughly, how many elements we'll need in the coupling matrix.
    cdef int num_elts =0
    for i in range(ns):
        for j in range(ms):
            if quantized_coupling[i,j]!=0.0:
                num_elts += (A_si[i+1]-A_si[i]) + (B_si[j+1]-B_si[j]) - 1

    cdef np.ndarray[int ,ndim=1,mode="c"] T_rows = np.zeros((num_elts,),dtype=np.int32)
    cdef np.ndarray[int ,ndim=1,mode="c"] T_cols = np.zeros((num_elts,),dtype=np.int32)
    cdef np.ndarray[DTYPE_t ,ndim=1,mode="c"] T_vals = np.zeros((num_elts,),dtype=DTYPE)
    cdef int k = 0
    for i in range(ns):
        a_local=a[A_si[i]:A_si[i+1]]/a_s[i]
        # assert( abs(np.sum(a_local)-1.0)<1.0e-7 )
        a_local_len=A_si[i+1]-A_si[i]
        for j in range(ms):
            if quantized_coupling[i,j] != 0.0:
                b_local=b[B_si[j]:B_si[j+1]]/b_s[j]
                # assert( abs(np.sum(b_local)-1.0)<1.0e-7 )
                b_local_len=B_si[j+1]-B_si[j]
                # print("i:" + str(i))
                # print("j:" + str(j))
                # print("k:" + str(k))
                # print("num_elts:"+str(num_elts))
                # print("A_si[i]:" + str(A_si[i]))
                # print("B_si[j]:" + str(B_si[j]))                
                sparse_oneD_OT_gw(
                    T_rows,
                    T_cols,
                    T_vals,
                    k,
                    A_si[i],
                    B_si[j],
                    a_local, # 1dim
                    a_local_len, #int
                    b_local,#1dim
                    b_local_len, #int
                    quantized_coupling[i,j])#float
                k+= (A_si[i+1]-A_si[i])+(B_si[j+1]-B_si[j])-1
    return (T_rows,T_cols,T_vals)
    


# Turning off bounds checking doesn't improve performance on my end.
def quantized_gw_cython(
        # a is the probability distribution on points of A
        np.ndarray[np.float64_t,ndim=1] a,
        # A_s=A_sample is a sub_matrix of A, of size ns x ns
        np.ndarray[np.float64_t,ndim=2] A_s,
        # A_si = A_sample_indices
        # Indices for sampled points of A, of length ns+1
        # Should satisfy A_s[x,y]=A[A_si[x],A_si[y]] for all x,y < ns
        # should satisfy A_si[ns]=n
        np.ndarray[Py_ssize_t,ndim=1] A_si,
        # Probability distribution on sample points of A_s; of length ns
        np.ndarray[np.float64_t,ndim=1] a_s,
        np.ndarray[np.float64_t,ndim=1] A_s_a_s,
        DTYPE_t c_As,
        # b is the probability distribution on points of B
        np.ndarray[np.float64_t,ndim=1] b,
        # B_sample, size ms x ms
        np.ndarray[np.float64_t,ndim=2] B_s,
        # B_sample_indices, size ms+1
        np.ndarray[Py_ssize_t,ndim=1] B_si,
        # Probability distribution on sample points of B_s; of length ms
        np.ndarray[np.float64_t,ndim=1] b_s,
        # np.dot(np.multiply(B_s,B_s),b_s)
        np.ndarray[np.float64_t,ndim=1] B_s_b_s,
        DTYPE_t c_Bs,
):
    cdef np.ndarray[np.float64_t,ndim=2,mode='c'] C = np.multiply(A_s_a_s[:,np.newaxis],(-2.0*B_s_b_s)[np.newaxis,:],order='C')
    return qgw_init_cost(
        a,
        A_s,
        A_si,
        a_s,
        c_As,
        b,
        B_s,
        B_si,
        b_s,
        c_Bs,
        C)
