import numpy as np
import os 
import multiprocess
from multiprocess import shared_memory
import scipy.spatial.distance
from os import getpid
import time
import random
import string
import warnings
import itertools as it
import csv

"""
TO DO

- change the kNN array from floats to ints

-output - array where [i,j] is index of jth NN to ith point
        - and optional second array is distance between i and jth NN point
    -should save this to .npy file 
    - or to csv

"""


def _init(_N,_k, _dist , _lock, _mem_code, _exact_distances):
    # print('init called', str(getpid()))
    global N 
    global k
    global dist_counter
    global my_lock
    global exact_distances
    N=_N 
    k=_k 
    dist_counter = _dist 
    my_lock = _lock
    mem_code = _mem_code
    exact_distances = _exact_distances

    global process_data_shm 
    global process_min_shm
    global process_max_shm
    global process_computed_shm 
    global process_kNN_shm
    global data_array
    global min_array 
    global max_array 
    global computed_array 
    global kNN_array
    global fully_computed 



    with my_lock:
        #print('init ', getpid(), ' has lock')
        process_data_shm = shared_memory.SharedMemory(name='data_mem' + mem_code, create = False)
        process_min_shm = shared_memory.SharedMemory(name='min_mem' + mem_code, create = False)
        process_max_shm = shared_memory.SharedMemory(name='max_mem' + mem_code, create = False)
        process_computed_shm = shared_memory.SharedMemory(name='computed_mem' + mem_code, create = False)
        process_kNN_shm = shared_memory.SharedMemory(name='kNN_mem' + mem_code, create = False)

        data_array = np.ndarray((N,3), dtype=float, buffer=process_data_shm.buf)
        min_array = np.ndarray( (N,N), dtype = float , buffer = process_min_shm.buf )
        max_array = np.ndarray( (N,N), dtype = float , buffer = process_max_shm.buf )
        computed_array = np.ndarray( (N,N), dtype = float , buffer = process_computed_shm.buf )
        kNN_array = np.ndarray( (N,k), dtype = float , buffer = process_kNN_shm.buf )

        # for a in range(N):
        #     for b in range(a):
        #         assert max_array[a,b] >0 #end of init

    #print('init done', getpid()) #debugging
    #print(kNN_array) #debugging
    #check_bounds('end of init' )

def _full_update_arrays(i,j,d):
    global N 
    global data_array
    global min_array 
    global max_array 
    global computed_array 
    global dist_counter 
    global fully_computed 
    global my_lock
    # updates min,max arrays
    # uses a slightly different algorithm using numpy matrix trick
    # 1-2 orders of magnitude faster
    #check_bounds('in start of full_update_arrays ' + str(getpid()))

    with my_lock:
        old_min = np.copy(min_array)
        old_max = np.copy(max_array)
        
        
        min_array[i][j] = d
        min_array[j][i] = d
        max_array[i][j] = d
        max_array[j][i] = d     
        

        D = np.ones((N))*d
        

        # #checking updates:
        # for aa in range(N):
        #     for bb in range(N):
        #         if old_min[aa,bb] > min_array[aa,bb]:
        #             print('_full_update_arrays, min error',aa,bb,i,j)
        #         if old_max[aa,bb] < max_array[aa,bb]:
        #             print('_full_update_arrays, max error',aa,bb,i,j)

        # update max
        #first update max distances to i
        a = np.minimum(max_array[i,:], D + max_array[j,:]  )
        max_array[i,:] = a
        max_array[:,i] = a.T

        #update max distances to j
        b = np.minimum(max_array[j,:], D + max_array[i,:]  )
        max_array[j,:] = b
        max_array[:,j] = b.T


    



        #update distances k to k'
        for k in range(N):
            x = np.minimum.reduce([max_array[k,:k], max_array[k][i] +max_array[i,:k], max_array[k][j] +max_array[j,:k]   ])
            max_array[k,:k] = x
            max_array[:k,k] = x.T


        # update min 
        
        c = np.maximum.reduce([min_array[i,:], min_array[j,:] - D, D - max_array[j,:] ])
        min_array[i,:] = c
        min_array[:,i] = c.T
        
        d = np.maximum.reduce([min_array[j,:], min_array[i,:] - D, D - max_array[i,:] ])
        min_array[j,:] = d
        min_array[:,j] = d.T        
        
        for k in range(N): 
            y = np.maximum.reduce([min_array[k,:k] , min_array[k,i]- max_array[i,:k] ,min_array[i,:k]- max_array[k,i] ,min_array[k,j]- max_array[j,:k],min_array[j,:k]- max_array[k,j]   ])
            #y = np.copy(min_array[k,:k])
            min_array[k, :k] = y
            min_array[:k, k] = y.T
            min_array[k,k] = 0



def _test_nearest( i):
    global N 
    global k
    global data_array
    global min_array 
    global max_array 
    global computed_array 
    global dist_counter 
    global fully_computed    
    global my_lock 

    with my_lock: 

        nearest = [(min_array[i,j],max_array[i,j],j)    for j in range(N)]  

    nearest.sort()
    # list of the other points relative to i 

    assert nearest[0] == (0,0,i) 

    
    for n in range(1,k+2): #this used to be k+1, which was likely a bug
        if nearest[n][1] >   nearest[n+1][0]:
            return nearest[n][2]
        
    for n in range(k+2):
        assert nearest[n][1] <= nearest[n+1][0]
    
    return -1

def _find_nearest(i,N,k):
    # returns a list of the current k nearest indices to i, in order
    # WARNING does not guarantee that this is the final order
    global data_array
    global min_array 
    global max_array 
    global computed_array 
    global kNN_array
    global dist_counter 
    global fully_computed    
    global my_lock 

    with my_lock: 
        nearest = [(min_array[i,j],max_array[i,j],j)    for j in range(N)]  
    nearest.sort()

    #print('_find_nearest called ' ,i)

    return( [ a[2] for a in nearest[1:k+1]])


def _get_distance(i,j,dist_func):
    #print('get_distance called', i,j, getpid())

    ######
    global k
    global data_array
    global min_array 
    global max_array 
    global computed_array 
    #global dist_counter 
    global fully_computed 
    global my_lock
    #returns the distance between points i and j
    #doesn't update the min/max arrays using the triangle ineq

    d = dist_func(i,j) #THIS WILL BE THE REAL ONE, OTHERS ARE FOR TESTING

    with my_lock:

        if not computed_array[i][j]:
            dist_counter.value += 1
            computed_array[i][j] = 1
            computed_array[j][i] = 1
            
        min_array[i,j] = d 
        min_array[j,i] = d 
        max_array[i,j] = d 
        max_array[j,i] = d 

        
    



    #debugging 
    with my_lock:
        if min_array[i,j] != d :
            print('_get_distance error, min',i,j,d, min_array[i,j])
        if max_array[i,j] != d :
            print('_get_distance error, max',i,j,d, max_array[i,j])
        if not computed_array[i,j]:
            print('_get_distance error computed_array',i,j)

    if d ==0.0 and i != j:
        print("DISTANCE ERROR, ", i,j,d)
        assert False

    return d

def _compute_anchor(i, N,dist_func):
    global data_array
    global min_array 
    global max_array 
    global computed_array 
    global dist_counter 
    global fully_computed 
    global my_lock

    for j in range(N):
        _get_distance(i,j,dist_func)
     
    with my_lock:
        for k in range(N):
            x = np.minimum.reduce([max_array[k,:k], max_array[k][i] +max_array[i,:k]])
            max_array[k,:k] = x
            max_array[:k,k] = x.T    
            
        for k in range(N):
            y = np.maximum.reduce([min_array[k,:k], min_array[k,i]- max_array[i,:k] ,min_array[i,:k]- max_array[k,i] ])
            min_array[k, :k] = y
            min_array[:k, k] = y.T
            min_array[k,k] = 0



    #check_bounds('end of compute_anchor',i)

def _compute_nearest(p):
    i,dist_func = p

    #print('_compute_nearest started',i)
    #print('_compute_nearest ',i, type(dist_func)) #debugging


    global N 
    global k
    global exact_distances

    global process_data_shm 
    global process_min_shm
    global process_max_shm
    global process_computed_shm 
    global data_array
    global min_array 
    global max_array
    global computed_array 
    global kNN_array
    global dist_counter 
    global fully_computed 
    global my_lock



    j = _test_nearest(i) 
    while j !=  -1:
        d = _get_distance(i,j,dist_func)
        _full_update_arrays(i,j,d) 
        j = _test_nearest(i)
        #print('_compute_nearest cycle done', i,j, d)


    a = np.copy(_find_nearest(i,N,k))

    #print('_compute_nearest halfway', i)

    if exact_distances:
        for j in list(a):
            _get_distance(i,j,dist_func)
            with my_lock:
                if not computed_array[i,j]:
                    print('_compute_nearest error, not all exact_distances computed ', i,j)
                if not min_array[i,j] == max_array[i,j]:
                    print('_compute_nearest error, difference in min vs max ', i,j)



    try:
        with my_lock:
            kNN_array[i][:] = a 
    except Exception as error:
        print('ALERT - ', error)

    #print('_compute_nearest done', i)


def run_triangle_ineq(
    N :int, # total number of points
    k: int,  # number of nearest neighbors to each point to find 
    dist_func : callable, # distance function, takes in indices and returns the distance between those points
    num_processes  : int = 8,  #number of parallel processes to run 
    mem_code :str = None ,  # unique code to prevent shared memory overlap, randomly generated if None
    exact_distances : bool = False, #whether to calculate the exact distance of the kNN points or just return them in order
    chunksize: int = 20
    ):

    global data_shms
    global min_shm
    global max_shm
    global computed_shm
    global kNN_shm

    global dist_counter 
    global fully_computed 
    global my_lock


    global data_array
    global min_array
    global max_array
    global computed_array
    global kNN_array

    seed = 8
    np.random.seed(seed)
    anchor_num = 2 # 0-3 is often best


    my_lock = multiprocess.Lock()

    if not mem_code:
        mem_code = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
        warnings.warn(message ='Cannot guarantee no shared memory interference')


    blank_array = np.zeros(shape = (N,N), dtype = float)

    data_shm = shared_memory.SharedMemory(name = 'data_mem' + mem_code, create=True, size=np.random.rand(N,3).nbytes)
    data_array = np.ndarray((N,3), buffer=data_shm.buf)
    np.copyto(dst = data_array, src =  np.random.rand(N,3))


    min_shm = shared_memory.SharedMemory(name = 'min_mem' + mem_code, create=True, size=blank_array.nbytes)
    min_array = np.ndarray((N,N), buffer=min_shm.buf)
    np.copyto(dst = min_array, src =  np.zeros((N,N)) )  

    max_shm = shared_memory.SharedMemory(name = 'max_mem' + mem_code, create=True, size=blank_array.nbytes)
    max_array = np.ndarray((N,N), buffer=max_shm.buf)
    np.copyto(dst = max_array, src =  np.full((N,N), 10000.1) ) 

    computed_shm = shared_memory.SharedMemory(name = 'computed_mem' + mem_code, create=True, size=blank_array.nbytes)
    computed_array = np.ndarray((N,N), buffer=computed_shm.buf)
    np.copyto(dst = computed_array, src =  np.zeros((N,N)) )  

    kNN_shm = shared_memory.SharedMemory(name = 'kNN_mem' + mem_code, create=True, size=np.random.rand(N,k).nbytes)
    kNN_array = np.ndarray((N,k), buffer=kNN_shm.buf)
    np.copyto(dst = kNN_array, src =  np.zeros((N,k)) )  



    for i in range(N):
        max_array[i][i] = 0
        computed_array[i,i] =1
        
    fully_computed = set([]) # points which have all pairwise distance computed
    dist_counter = multiprocess.Value('d',0.0)  #counts the number of times we've called for a pairwise distance
    dist_counter.value = 0



    for i in range(anchor_num):
        _compute_anchor(i,N,dist_func)

        a = np.copy(_find_nearest(i,N,k))
        with my_lock:
            kNN_array[i][:] = a




    with multiprocess.Pool(processes=num_processes, initializer= _init, initargs = (N,k, dist_counter, my_lock, mem_code, exact_distances)) as pool:
        print('multiprocess started')
        results = pool.imap(_compute_nearest, zip(range( anchor_num, N), it.repeat(dist_func)) , chunksize = 5) 
        # https://stackoverflow.com/questions/26520781/multiprocess-pool-whats-the-difference-between-map-async-and-imap
        pool.close()
        pool.join() 





    with my_lock:
        output_kNN_array = np.copy(kNN_array)
    #print('output_kNN_array', output_kNN_array) #debugging

    if exact_distances:
        with my_lock:
            output_dist_array = np.zeros((N,k), dtype = float)
            for i in range(N):
                for j in range(k): 
                    a = int(output_kNN_array[i,j])
                    assert computed_array[i,a]
                    if min_array[i,a] != max_array[i,a]:
                        print('exact_distances error',i,j,a)
                        print('min i a', min_array[i,a] )
                        print('max i a', max_array[i,a] )
                        assert False
                    output_dist_array[i,j] = min_array[i,a] 


    data_shm.close()
    data_shm.unlink()
    min_shm.close()
    min_shm.unlink()
    max_shm.close()
    max_shm.unlink()
    computed_shm.close()
    computed_shm.unlink()
    kNN_shm.close()
    kNN_shm.unlink()
    #print("dist_counter = ", dist_counter.value, ' anchor_num = ', anchor_num)  
    #roughly N * k * 1.5 distance calculations


    if exact_distances:
        return output_kNN_array, output_dist_array
    return output_kNN_array




def triangle_ineq_exact_csv( 
    #Writes results to csv files
    N :int, # total number of points
    k: int,  # number of nearest neighbors to each point to find 
    dist_func : callable, # distance function, takes in indices and returns the distance between those points
    output_kNN_csv: str, # where the kNN array is stored
    output_dist_csv: str, # where the distances are stored
    num_processes  : int = 8,  #number of parallel processes to run 
    mem_code :str = None ,  # unique code to prevent shared memory overlap, randomly generated if None
    chunksize: int = 20
    ):

    input_kNN_array , input_dist_array = run_triangle_ineq(N =N, k=k, dist_func = dist_func, num_processes = num_processes, mem_code = mem_code, exact_distances = True, chunksize = chunksize)
    np.savetxt(output_kNN_csv, input_kNN_array, delimiter = ',') 
    np.savetxt(output_dist_csv, input_dist_array, delimiter = ',') 
    return 0


def triangle_ineq_csv(
    #writes result to csv file
    N :int, # total number of points
    k: int,  # number of nearest neighbors to each point to find 
    dist_func : callable, # distance function, takes in indices and returns the distance between those points
    output_kNN_csv: str, #where the kNN array is stored
    num_processes  : int = 8,  #number of parallel processes to run 
    mem_code :str = None ,  # unique code to prevent shared memory overlap, randomly generated if None
    chunksize: int = 20
    ):

    input_kNN_array = run_triangle_ineq(N =N, k=k, dist_func = dist_func, num_processes = num_processes, mem_code = mem_code, exact_distances = False, chunksize = chunksize)
    np.savetxt(output_kNN_csv, input_kNN_array, delimiter = ',') #changed to test
    return 0




def triangle_ineq_exact_npy( 
    #Writes results to npy files
    N :int, # total number of points
    k: int,  # number of nearest neighbors to each point to find 
    dist_func : callable, # distance function, takes in indices and returns the distance between those points
    output_kNN_npy: str, # where the kNN array is stored
    output_dist_npy: str, # where the distances are stored
    num_processes  : int = 8,  #number of parallel processes to run 
    mem_code :str = None ,  # unique code to prevent shared memory overlap, randomly generated if None
    chunksize: int = 20
    ):

    input_kNN_array , input_dist_array = run_triangle_ineq(N =N, k=k, dist_func = dist_func, num_processes = num_processes, mem_code = mem_code, exact_distances = True, chunksize = chunksize)
    input_kNN_array.tofile(output_kNN_npy, sep = ',')
    input_dist_array.tofile(output_dist_npy, sep = ',')
    with open('output_kNN_npy', 'wb') as f:
        np.save(f, input_kNN_array)
    with open('output_dist_npy', 'wb') as f:
        np.save(f, input_dist_array)
    return 0


def triangle_ineq_npy(
    #writes result to npy file
    N :int, # total number of points
    k: int,  # number of nearest neighbors to each point to find 
    dist_func : callable, # distance function, takes in indices and returns the distance between those points
    output_kNN_npy: str, # where the kNN array is stored
    num_processes  : int = 8,  #number of parallel processes to run 
    mem_code :str = None ,  # unique code to prevent shared memory overlap, randomly generated if None
    chunksize: int = 20
    ):

    input_kNN_array = run_triangle_ineq(N =N, k=k, dist_func = dist_func, num_processes = num_processes, mem_code = mem_code, exact_distances = False, chunksize = chunksize)
    with open('output_kNN_npy', 'wb') as f:
        np.save(f, input_kNN_array)
    return 0

def _test_dist2(i,j):
    #print('test_dist2 applied')
    np.random.seed(8)
    A = np.random.rand(20,3)
    
    return np.linalg.norm(A[i]-A[j])




# if __name__ == "__main__":
#     a,b =run_triangle_ineq(N =20, k =4, num_processes = 5, dist_func =  test_dist2 ,exact_distances = True, chunksize = 5)
#     print (a)
#     print(b)

