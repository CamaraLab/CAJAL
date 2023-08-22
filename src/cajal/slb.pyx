# distutils: language=c++
"""
SLB2
"""

import numpy as np
cimport numpy as np
from .gw_cython import intersection
np.import_array()
from math import ceil,sqrt

DTYPE=np.float64
ctypedef np.float_t DTYPE_t


def slb(np.ndarray[DTYPE_t,ndim=1] f, np.ndarray[DTYPE_t,ndim=1] g):
    # Assume f, g are of shape (flen,) and (glen,), and are sorted.
    cdef int flen = f.shape[0]
    cdef int glen = g.shape[0]
    cdef float acc, progress, fnext, gnext, intval, delta
    cdef int i, j

    i = 0
    cdef int n = ceil(sqrt(2.0*flen))
    assert (n * (n-1))==flen*2
    j = 0
    cdef int m = ceil(sqrt(2.0*glen))
    assert (m * (m-1))==glen*2
    
    acc=0
    progress=0

    # the index i implicitly ranges across points on the unit interval where the value of d_X^{-1}
    # spikes, specifically, via the correspondence i \mapsto (1/n) + 2(i+1)/n^2
    
    # If X has n many points, the value of d_X^{-1} on the closed interval
    # [0,1/n] is 0.  If (x_i,x_j) is the closest pair of distinct points, and
    # their distance is d(x_i,x_j), then the value of d_X^{-1} at (1/n) + 2/n^2 is d(x_i,x_j).
    # Formally, if the k-th pair of points (k = 1, .... (n*(n-1))/2) is x_i,x_j, then
    # for t \in [1/n + 2k/n^2, 1/n + 2(k+1)/n^2), the value of f is d(x_i,x_j).

    # Here it is assumed that flen and glen are of the form n * (n-1)/2, m * (m-1)/2.
    while i < flen and j < glen:
        if (1.0/n + 2.0*(<float>i+1.0)/n**2) < (1.0/m + 2.0*(<float>j+1.0)/m**2):
            while(1.0/n + 2.0*(<float>i+2.0)/n**2) < (1.0/m + 2.0*(<float>j+1.0)/m**2):
                if j==0:
                    intval=(f[i]) ** 2
                else:
                    intval=(f[i]-g[j-1]) ** 2
                acc+= intval*(2.0)/(n**2)
                i+=1
            # Postcondition:  1/n + 2(i+1)/n^2 < 1/m + 2(j+1)/m^2,
            # but 1/n + 2(i+2)/n^2 >= 1/m + 2(j+1)/m^2.
            if j==0:
                intval=(f[i]) ** 2
            else:
                intval=(f[i]-g[j-1]) ** 2
            # This value of the function occurs until the next distinguished value,
            # at the current value of j.
            delta = (1.0/m + 2.0*(j+1.0)/(m**2))-(1.0/n + 2.0*(i+1.0)/(n**2))
            assert(delta>=0)
            acc+=delta*intval
            i+=1
        else:
            # j pointer is behind i.
            while(1.0/n + 2.0*(<float>i+1.0)/n**2) >= (1.0/m + 2.0*(<float>j+2.0)/m**2):
                if i == 0:
                    intval=(g[j])**2
                else:
                    intval=(g[j]-f[i-1])**2
                acc+=intval*(2.0)/(m**2)
                j+=1
            # Postcondition:  1/n + 2(i+1)/n^2 >= 1/m + 2(j+1)/m^2,
            # but 1/n + 2(i+1)/n^2 <= 1/m + 2(j+2)/m^2.
            if i==0:
                intval=(g[j]) ** 2
            else:
                intval=(f[i-1]-g[j]) ** 2
            delta = (1.0/n + 2.0*(i+1.0)/(n**2))-(1.0/m + 2.0*(j+1.0)/(m**2))
            assert(delta>=0)            
            acc+=delta*intval
            j+=1
    if j == glen-1:
        while i < flen-1:
            intval=(f[i]-g[j-1])**2
            acc+=intval*(2.0)/(n**2)
            i+=1
    if i==flen-1:
        while j < glen-1:
            intval=(g[j]-f[i-1])**2
            acc+=intval*(2.0)/(m**2)
            j+=1
    return sqrt(acc)/2.0

def slb2_block(np.ndarray[DTYPE_t,ndim=1] f, np.ndarray[DTYPE_t,ndim=1] g):
    # Assume f, g are of shape (flen,) and (glen,), and are sorted.
    cdef int flen = f.shape[0]
    cdef int glen = g.shape[0]
    cdef float acc, progress, fnext, gnext
    cdef int i, j

    i = 0
    j = 0
    acc=0
    progress=0

    while i < flen or j < glen:
        if i < (flen - 1):
            fnext = (<float>(i + 1))/(<float>flen)
            if j < (glen - 1):
                gnext = (<float>(j + 1))/ (<float>glen)
                if fnext < gnext:
                    acc+=((f[i]-g[j])**2) * (fnext-progress)
                    progress = fnext
                    i += 1
                else:
                    acc+=((f[i]-g[j])**2) * (gnext-progress)
                    progress = gnext
                    j += 1
            else:
                acc+=((f[i]-g[j])**2) * (fnext-progress)
                progress = fnext
                i += 1
        else:
            if j < (glen-1) and (<float>(glen-1)/<float>glen)>progress:
                acc +=((f[flen-1]-g[glen-1])**2) * ((<float>(glen-1)/<float>glen)-progress)
            return acc
    return acc

def l2(np.ndarray[DTYPE_t,ndim=1,mode='c'] f,
       np.ndarray[DTYPE_t,ndim=1,mode='c'] u,
       np.ndarray[DTYPE_t,ndim=1,mode='c'] cum_u,
       np.ndarray[DTYPE_t,ndim=1,mode='c'] g,
       np.ndarray[DTYPE_t,ndim=1,mode='c'] v,
       np.ndarray[DTYPE_t,ndim=1,mode='c'] cum_v,
       ):
    """
    Compute the (squared) L^2 distance between two functions, f and g, on the unit interval. The interpretation is that
    f[i] is the value of the function f on the interval [ sum(u[:i]), sum(u[:i+1])], and g[j] is the value of the function
    g on the interval [sum(v[:j]),sum(v[:j+1])].

    :param f: a piecewise constant function on the unit interval (the list of values of the function)
    :param u: a probability distribution on [0,1] (a vector of probabilities summing to one). Same length as f
    :param cum_u: cumulative distribution of u
    :param g: a piecewise constant function on the unit interval (the list of values of the function)
    :param u: a probability distribution on [0,1]. Same length as g.
    :param cum_v: cumulative distribution of v
    """

    cdef int flen = f.shape[0]
    cdef int glen = g.shape[0]
    cdef float acc, progress, fnext, gnext, f_begin, g_begin, region
    cdef int i, j

    i = 0
    j = 0
    f_begin = 0.0
    g_begin = 0.0
    acc=0
    progress=0

    while i < flen and j < glen:
        f_begin = cum_u[i-1] if i>0 else 0.0
        g_begin = cum_v[j-1] if j>0 else 0.0
            
        region = intersection(f_begin,cum_u[i],
                              g_begin,cum_v[j])
        acc+=((f[i]-g[j])**2) * region
        if cum_u[i] < cum_v[j] and i < flen - 1:
            i+=1
        elif cum_u[i] >= cum_v[j] and j < glen - 1:
            j+=1
        # Either i == flen - 1 or j == glen - 1.
        elif i == flen - 1:
            j+=1
        else:
            assert j == glen -1
            i+=1
        
    assert (i == flen and j == glen -1) or (i == flen - 1) and (j == glen)
    return acc



    
        
        


