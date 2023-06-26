from libc.stdint cimport uint64_t
cdef extern from "EMD.h":
     cdef int myCMessage(int a, int b)
     cdef int EMD_wrap(int n1, int n2, double *X, double *Y, double *D, double *G, double* alpha, double* beta, double *cost, uint64_t maxIter) nogil