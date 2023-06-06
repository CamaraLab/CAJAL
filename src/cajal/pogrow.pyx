"""
POGROW
"""

import numpy as np
cimport numpy as np
np.import_array()
from libc.stdlib cimport rand

cdef extern from "stdlib.h":
    int RAND_MAX
DTYPE=float
ctypedef np.float_t DTYPE_t

def oneDtransport(
        np.ndarray[DTYPE_t,ndim=1] f,
        np.ndarray[DTYPE_t,ndim=1] g,
        np.ndarray[DTYPE_t,ndim=2] T,
        np.ndarray[np.int_t,ndim=1] fsort,
        np.ndarray[np.int_t,ndim=1] gsort
):
    cdef int fsize = f.shape[0]
    cdef int gsize = g.shape[0]
    # cdef np.ndarray[np.int_t,ndim=1] fsort = np.empty((fsize,),dtype=np.int_t)
    fsort=np.argsort(f)
    # cdef np.ndarray[np.int_t,ndim=1] gsort = np.empty((gsize,),dtype=np.int_t)
    gsort=np.argsort(g)
    cdef int i=0
    cdef int j=0


    T.fill(0.0)
    while (i < fsize) and (j < gsize):
        if (<float>i)/(<float>fsize) <= (<float>j)/(<float>gsize):
            while (<float>i+1)/(<float>fsize) <= (<float>j)/(<float>gsize):
                i += 1
            # assert i < fsize
            # assert (<float>i/<float>fsize)<=(<float>j/<float>gsize)
            T[fsort[i],gsort[j]]= min(<float>(i+1)/<float>fsize,<float>(j+1)/<float>gsize)-(<float>i/<float>fsize)
            # assert T[fsort[i],gsort[j]]>=0
            i += 1
        else:                       # i/fsize > j/gsize
            while (<float>j+1)/(<float>gsize) <= (<float>i)/(<float>fsize):
                j += 1
            # assert j < gsize
            # assert (<float>j/<float>gsize)<=(<float>i/<float>fsize)
            T[fsort[i],gsort[j]]= min(<float>(i+1)/<float>fsize,<float>(j+1)/<float>gsize)-(<float>j/<float>gsize)
            # assert T[fsort[i],gsort[j]]>=0
            j += 1

    # This code defines stuff for the sake of assertions to make sure the code works.
    # TODO: Move this to tests.
    # cdef np.ndarray[DTYPE_t,ndim=1] n_inv = np.full((fsize,),1.0/fsize,dtype=DTYPE)
    # cdef np.ndarray[DTYPE_t,ndim=1] m_inv = np.full((gsize,),1.0/gsize,dtype=DTYPE)
    # cdef np.ndarray[DTYPE_t,ndim=1] fsum = np.sum(T,axis=0)
    # cdef np.ndarray[DTYPE_t,ndim=1] gsum = np.sum(T,axis=1)
    # assert (np.all(np.isclose(fsum,n_inv)))
    # assert (np.all(np.isclose(gsum,m_inv)))

def search(np.ndarray[DTYPE_t,ndim=2] T, Py_ssize_t n, Py_ssize_t m, float p):

    cdef float cumprob = 0.0
    cdef Py_ssize_t x = 0
    cdef Py_ssize_t y = 0
    while (x < n) and (y < m) and (p < cumprob + T[x,y]):
        cumprob+=T[x,y]
        if x < n-1:
            x+=1
        elif y < m-1:                   # x == n-1
            y+=1
            x=0
        else:                   # x==n-1 and y==n-1
            return x,y
    return x, y


def pogrow(np.ndarray[DTYPE_t,ndim=2] f, np.ndarray[DTYPE_t,ndim=2] g, np.int it, np.float alpha):
    cdef Py_ssize_t fsize = f.shape[0]
    cdef np.ndarray[np.int_t,ndim=1] fsort=np.empty((fsize,),dtype=int)
    cdef Py_ssize_t gsize = g.shape[0]
    cdef np.ndarray[np.int_t,ndim=1] gsort=np.empty((gsize,),dtype=int)
    cdef np.ndarray[DTYPE_t,ndim=2] T = np.full((fsize,gsize),1.0/(<float>fsize*gsize),dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=2] T0 = np.empty((fsize,gsize),dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=2] hatL = np.zeros((fsize,gsize),dtype=DTYPE)
    cdef int i
    cdef int k
    cdef int s
    cdef float p

    for s in range(it):
        p= rand() / <float>RAND_MAX
        i,k=search(T,fsize,gsize,p)
        oneDtransport(f[i,:],g[k,:],T0,fsort,gsort)
        T0 *= alpha
        T *= (1.0-alpha)
        T += T0
    return T      

def medoid_coupling(
        np.ndarray[DTYPE_t,ndim=2] C,
        np.ndarray[DTYPE_t,ndim=2] Cbar,
        np.ndarray[DTYPE_t,ndim=2] T
):
    cdef int Csize=C.shape[0]
    cdef int Cbarsize=Cbar.shape[0]    
    cdef np.ndarray[DTYPE_t,ndim=2] res=np.matmul(C,C)
    cdef float a = np.sum(res)/(<float>Csize)
    np.matmul(Cbar,Cbar,out=res)


