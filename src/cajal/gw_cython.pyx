# cython: profile=True
"""
GW
"""

import numpy as np
cimport numpy as np
cimport cython
np.import_array()
from libc.stdlib cimport rand
from ot.lp import emd_c, emd
from math import sqrt
from scipy.sparse import lil_matrix

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

def frobenius(DTYPE_t[:,:] A, DTYPE_t[:,:] B):

    cdef int n = A.shape[0]
    cdef int m = A.shape[1]
    assert n==B.shape[0]
    assert m==B.shape[1]
    cdef DTYPE_t sumval = 0.0
    cdef int i, j
    
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            sumval+=A[i,j]*B[i,j]
    return sumval

def gromov_linesearch(
        np.ndarray[np.float64_t,ndim=2] c_C_Cbar,
        np.ndarray[np.float64_t,ndim=2] C,
        np.ndarray[np.float64_t,ndim=2] Cbar,
        int C_length,
        int Cbar_length,
        np.ndarray[np.float64_t,ndim=2] T,
        np.ndarray[np.float64_t,ndim=2] deltaT,
        DTYPE_t cost_T
):
    # GW loss is
    # (-2 * < C * (\Delta T) * Cbar.T, \Delta T>_F)  * t^2 +
    # (<c_C_Cbar, \Delta T>_F  - 2*( <C * T * Cbar.T, \Delta T>_F + <C*(\Delta T)*Cbar.T,T>_F)) * t +
    # cost_T
    cdef np.ndarray[np.float64_t,ndim=2] C_deltaT_Cbar_T = np.matmul(C,deltaT)
    np.matmul(C_deltaT_Cbar_T,Cbar.T,out=C_deltaT_Cbar_T)
    # C_deltaT_Cbar_T =np.multiply(C_deltaT_Cbar_T, -2.)
    cdef DTYPE_t x
    cdef DTYPE_t y

    cdef DTYPE_t a=frobenius(C_deltaT_Cbar_T,deltaT)
    a *= -2.0                   # a is done
    cdef DTYPE_t b=frobenius(C_deltaT_Cbar_T,T)
    cdef np.ndarray[np.float64_t,ndim=2] C_T_Cbar_T = np.matmul(C,T)
    np.matmul(C_T_Cbar_T,Cbar.T,out=C_T_Cbar_T)
    b+=frobenius(C_T_Cbar_T,deltaT)
    b*= -2.0
    b+=frobenius(c_C_Cbar,deltaT) # b is done
 
    if a > 0:
        x = min(1.,max(0., -b/(2.0*a)))
    elif (a + b) < 0:
        x = 1.0
    else:
        x = 0.0
    y=(a * (x ** 2) + (b * x) + cost_T)
    return (x,y)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def gw_cython(
        np.ndarray[np.float64_t,ndim=2] C,
        np.ndarray[np.float64_t,ndim=2] Cbar,
        np.ndarray[np.float64_t,ndim=1] p,
        np.ndarray[np.float64_t,ndim=1] q,
        # assumed to be np.multiply(C,C) * p (not matmul!)
        # We will adopt the convention that this will be input as a matrix of shape (n,1)
        np.ndarray[np.float64_t,ndim=2] c_C, 
        # assumed to be q^T * np.multiply(Cbar.T,Cbar.T) (not matmul!)
        # We will adopt the convention that this will be input as a matrix of shape (1,n)
        np.ndarray[np.float64_t,ndim=2] c_Cbar,
        # Probably have to manually broadcast last two arguments.
        max_iters_OT : int =100000,
        max_iters_descent : int =1000
):
    # L(C,\overline{C}) \otimes T = (C_sq * p * 1^T) + ( - 2 * (C * T * Cbar)
    # L(C,\overline{C}) \otimes T = c_C_Cbar - 2 * (C * T * Cbar.T)

    cdef np.ndarray[np.float64_t,ndim=2] c_C_Cbar = c_C+c_Cbar
    cdef np.ndarray[np.float64_t,ndim=2] T = np.matmul(
        p[:,np.newaxis],
        q[np.newaxis,:])
    cdef int C_length=C.shape[0]
    cdef int Cbar_length=Cbar.shape[0]
    cdef int it = 0
    cdef DTYPE_t alpha
    cdef DTYPE_t gw_loss_T
    cdef DTYPE_t new_gw_loss_T
    log={ 'loss' : [], "alphas" : [] }
    cdef np.ndarray[np.float64_t,ndim=2] T_new =np.empty((C_length,Cbar_length),dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=2] deltaT = np.empty((C_length,Cbar_length),dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=2] LCCbar_otimes_T = np.empty((C_length,Cbar_length),dtype=DTYPE)

    matrix_tensor(c_C_Cbar,C,Cbar,T,LCCbar_otimes_T)
    gw_loss_T=frobenius(LCCbar_otimes_T,T)
    log['loss'].append(gw_loss_T)
    log['sparse']=[]

    while it<max_iters_descent:
        # T_new, inner_log = emd(p,q,2*LCCbar_otimes_T,max_iters_OT,log=True)
        # It's tempting to use the 'cost' from this function but remember that
        # it's a "mixed cost" between two different transport plans, T_r and T_r+1.
        # It's not meaningful! Don't use it
        T_new, _, _, _,_ = emd_c(p,q,2*LCCbar_otimes_T,max_iters_OT,numThreads=1)
        log['sparse'].append(np.count_nonzero(T_new))
        matrix_tensor(c_C_Cbar,C,Cbar,T_new,LCCbar_otimes_T)
        new_gw_loss_T =frobenius(LCCbar_otimes_T,T_new)
        if (new_gw_loss_T >= gw_loss_T):
            return (T, sqrt(gw_loss_T)/2,log)
        elif (new_gw_loss_T/gw_loss_T - 1.0 > -(1e-9)) or (new_gw_loss_T-gw_loss_T > -(1e-9)):
            gw_loss_T=new_gw_loss_T
            log['loss'].append(gw_loss_T)
            return (T_new, sqrt(gw_loss_T)/2,log)
        T=T_new
        gw_loss_T=new_gw_loss_T
        it +=1

    log['loss'].append(gw_loss_T)
    # assert np.allclose(np.sum(T,axis=0),p)
    # assert np.allclose(np.sum(T,axis=1),q)
    return (T, sqrt(gw_loss_T)/2,log)

def intersection(DTYPE_t a, DTYPE_t b, DTYPE_t c, DTYPE_t d):
    cdef maxac= a if a >= c else c
    cdef minbd= b if b <= d else d
    minbd=minbd-maxac
    return minbd if minbd >= 0.0 else 0.0

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

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def quantized_gw(
        # A is an n x n distance matrix
        # np.ndarray[np.float64_t,ndim=2] A,
        # a is the probability distribution on points of A
        np.ndarray[np.float64_t,ndim=1] a,
        # A_s=A_sample is a sub_matrix of A, of size ns x ns
        np.ndarray[np.float64_t,ndim=2] A_s,
        # A_si = A_sample_indices
        # Indices for sampled points of A, of length ns.
        # Should satisfy A_s[x,y]=A[A_si[x],A_si[y]] for all x,y < ns
        np.ndarray[Py_ssize_t,ndim=1] A_si,
        # Probability distribution on sample points of A_s; of length ns
        np.ndarray[np.float64_t,ndim=1] a_s,
        # np.dot(np.multiply(A_s,A_s),a_s)
        np.ndarray[np.float64_t,ndim=1] As_As_as,
        # B is an mxm distance matrix
        # np.ndarray[np.float64_t,ndim=2] B,
        # b is the probability distribution on points of B
        np.ndarray[np.float64_t,ndim=1] b,
        # B_sample, size ms x ms
        np.ndarray[np.float64_t,ndim=2] B_s,
        # B_sample_indices, size ms
        np.ndarray[Py_ssize_t,ndim=1] B_si,
        # Probability distribution on sample points of B_s; of length ms
        np.ndarray[np.float64_t,ndim=1] b_s,
        # np.dot(np.multiply(B_s,B_s),b_s)
        np.ndarray[np.float64_t,ndim=1] Bs_Bs_bs,
):

    # Assumptions: The points of A are arranged in such a way that:
    # for all k, i,  A_si[k] <= i < A_si[k+1],
    # the point i of A belongs to the Voronoi cell determined by A_si[k];
    # moreover, within the region A_si[k] <= i < A_si[k+1], the points
    # are in sorted order, i..e, for A_si[k] <= i < j < A_si[k+1],
    # we have A[A_si[k],i]<=A[A_si[k],j]
    # And B should also satisfy these assumptions.
    cdef int n = a.shape[0]
    cdef int ns = A_s.shape[0]
    cdef int m = b.shape[0]
    cdef int ms = B_s.shape[0]
    cdef DTYPE_t gw_cost=0.0
    cdef DTYPE_t local_gw_cost=0.0
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t j = 0
    cdef np.ndarray[np.float64_t,ndim=1] a_local
    cdef np.ndarray[np.float64_t,ndim=1] b_local
    cdef int a_local_len
    cdef int b_local_len
    cdef np.ndarray[np.float64_t,ndim=2] quantized_coupling # size ns x ms
    cdef np.ndarray[np.float64_t,ndim=2] T = np.zeros((n,m),dtype=DTYPE) # size n x m - the
    cdef DTYPE_t[:,:] T_local
    quantized_coupling, _,_ =gw_cython(A_s,B_s,a_s,b_s,As_As_as[:,np.newaxis],Bs_Bs_bs[np.newaxis,:])

    for i in range(ns):
        if (i+1<ns):
            a_local=a[A_si[i]:A_si[i+1]]/a_s[i]
            # assert( abs(np.sum(a_local)-1.0)<1.0e-7 )
            a_local_len=A_si[i+1]-A_si[i]
        else:
            # assert(i+1==ns)
            a_local=a[A_si[i]:]/a_s[i]
            # assert( abs(np.sum(a_local)-1.0)<1.0e-6 )
            a_local_len=n-A_si[i]
        for j in range(ms):
            if quantized_coupling[i,j] != 0.0:
                if (j+1<ms):
                    if(i+1<ns):
                        T_local= T[A_si[i]:A_si[i+1],:][:,B_si[j]:B_si[j+1]]
                    else:
                        # assert (i == ns-1)
                        T_local= T[A_si[i]:,:][:,B_si[j]:B_si[j+1]]
                    b_local=b[B_si[j]:B_si[j+1]]/b_s[j]
                    # assert( abs(np.sum(b_local)-1.0)<1.0e-7 )
                    b_local_len=B_si[j+1]-B_si[j]
                else:
                    # assert(j+1==ms)
                    if(i+1<ns):
                        T_local= T[A_si[i]:A_si[i+1],:][:,B_si[j]:]
                    else:
                        # assert (i == ns-1)
                        T_local= T[A_si[i]:,:][:,B_si[j]:]
                    b_local=b[B_si[j]:]/b_s[j]
                    # assert( abs(np.sum(b_local)-1.0)<1.0e-7 )
                    b_local_len=m-B_si[j]
                oneD_ot_gw(a_local, # 1dim
                           a_local_len, #int
                           b_local,#1dim
                           b_local_len, #int
                           T_local,# rectangle
                           quantized_coupling[i,j]) #float
    return T

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def quantized_gw_2(
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
        # np.dot(np.multiply(A_s,A_s),a_s)
        np.ndarray[np.float64_t,ndim=1] As_As_as,
        # b is the probability distribution on points of B
        np.ndarray[np.float64_t,ndim=1] b,
        # B_sample, size ms x ms
        np.ndarray[np.float64_t,ndim=2] B_s,
        # B_sample_indices, size ms+1
        np.ndarray[Py_ssize_t,ndim=1] B_si,
        # Probability distribution on sample points of B_s; of length ms
        np.ndarray[np.float64_t,ndim=1] b_s,
        # np.dot(np.multiply(B_s,B_s),b_s)
        np.ndarray[np.float64_t,ndim=1] Bs_Bs_bs,
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

    quantized_coupling, _,_ =gw_cython(A_s,B_s,a_s,b_s,As_As_as[:,np.newaxis],Bs_Bs_bs[np.newaxis,:])
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

