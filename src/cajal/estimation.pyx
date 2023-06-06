
#include <stdbool.h>
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
import os
import numpy as np
from scipy.spatial.distance import squareform
cimport numpy as np
def nn_slb(
        np.ndarray[np.float64_t,ndim=2] slb_values,
        np.ndarray[np.float64_t,ndim=2] gw_known,
        int N
):

    cdef int num_cells = slb_values.shape[0]
    cdef np.ndarray[np.float64_t,ndim=2] new_values = np.copy(slb_values)
    cdef np.ndarray[np.npy_bool,ndim=2] known = np.full((num_cells,num_cells),False,dtype=bool)
    cdef np.ndarray[np.int_t,ndim=1] indices
    cdef np.ndarray[np.int_t,ndim=1] indices_temp
    cdef np.ndarray[np.int_t,ndim=1] true_indices
    cdef np.npy_bool[:] known_this_row
    cdef int known_out_to, i, j
    cdef int temp
    cdef int computations
    computations_count=[]
    overlap=[]
    for i in range(num_cells):
        known[i,i]=True

    for current_row in range(num_cells):
        computations=0
        known_out_to = 0
        indices=np.argsort(new_values[current_row,:])
        known_this_row=known[current_row,:]
        assert known_this_row[indices[known_out_to]] # Because indices[known_out_to] should be 0.
        # print("Initial: ")
        # print(indices[:N])
        while known_out_to <= N:
            # for k in range(N):
            #     print(f'{new_values[current_row,indices[k]]:.5f}',end=',')
            # print('\n')
            while known_this_row[indices[known_out_to]]:
                known_out_to +=1
            # Postcondition: known[current_row,indices[current_row,known_out_to]] is False.
            if known_out_to <= N:
                new_values[current_row,indices[known_out_to]]=gw_known[current_row,indices[known_out_to]]
                known_this_row[indices[known_out_to]]=True
                new_values[indices[known_out_to],current_row]=gw_known[indices[known_out_to],current_row]
                known[indices[known_out_to],current_row]=True
                computations+=1
                j=known_out_to
                while new_values[current_row,indices[j]]>new_values[current_row,indices[j+1]]:
                    temp=indices[j]
                    indices[j]=indices[j+1]
                    indices[j+1]=temp
                    j+=1
            # for k in range(N):
            #     print(f'{new_values[current_row,indices[k]]:.5f}',end=',')
            # print('\n')
            # print(indices[:N])
        computations_count.append(computations)
        true_indices=np.argsort(gw_known[current_row,:])[:N]
        overlap.append(np.intersect1d(indices[:N],true_indices).shape[0])
    
    return computations_count, overlap
