

import numpy as np
import os 
import multiprocess
from multiprocess import shared_memory, Value
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


-test more and make more effiecient
-test non-exact_distances version  
    - kNN is not guaranteed to agree with the base_dist_func kNN
    - no way to guarantee we find the "true" values
-formalize mathematical assumptions

DONE


- mathematical assumption 
    - the triangle inequality is satisfied within epsilon
    - the base_dist_func is an approximations from above

- make version where triangle inequality is satisfied within epsilon

- try out different numbers of anchors and choosing them in a good order

- reduce locking time to make everything faster
    - it's safe to read without locking
    - not sure how much improvement can be made without introducing race conditions




-add a functionality to output all the arrays for testing/debugging purposes


-deleted old data_array stuff
-changed kNN_array and computed_array to int arrays
- changed the initial max_array to np.inf

- modify update methods so that if a distance has been computed it cannot be changed
- or rather when dealing with distance computation inaccuracy, the computed one is an overestimate,
so we should always take the minimum
-and if minimum has been computed it cannot be changed

-fixed whatever bug is causing the crash on the desktops - likely related to the shm's
    -current lead is that it's from a limit in docker, can be changed using --shm-size=<size>



"""


def _init(_N,_k, _dist , _lock, _mem_code, _exact_distances, _dist_time, _update_time, _wait_time):
    # print('init called', str(getpid()))
    global N 
    global k
    global dist_counter
    global dist_time
    global update_time
    global wait_time
    
    global my_lock
    global exact_distances
    N=_N 
    k=_k 
    dist_counter = _dist 
    dist_time = _dist_time
    update_time = _update_time
    wait_time = _wait_time

    my_lock = _lock
    mem_code = _mem_code
    exact_distances = _exact_distances


    global process_min_shm
    global process_max_shm
    global process_uncomputed_shm 
    global process_kNN_shm
    global min_array 
    global max_array 
    global uncomputed_array 
    global kNN_array
    global fully_computed 



    with my_lock:
        #print('init ', getpid(), ' has lock')
        process_min_shm = shared_memory.SharedMemory(name='min_mem' + mem_code, create = False)
        process_max_shm = shared_memory.SharedMemory(name='max_mem' + mem_code, create = False)
        process_uncomputed_shm = shared_memory.SharedMemory(name='uncomputed_mem' + mem_code, create = False)
        process_kNN_shm = shared_memory.SharedMemory(name='kNN_mem' + mem_code, create = False)

        min_array = np.ndarray( (N,N), dtype = float , buffer = process_min_shm.buf )
        max_array = np.ndarray( (N,N), dtype = float , buffer = process_max_shm.buf )
        uncomputed_array = np.ndarray( (N,N), dtype = bool, buffer = process_uncomputed_shm.buf )
        kNN_array = np.ndarray( (N,k), dtype = int , buffer = process_kNN_shm.buf )



def _full_update_arrays(i,j,d):
    global N 
    global min_array 
    global max_array 
    global uncomputed_array 
    global my_lock
    global update_time
    global wait_time
    # updates min,max arrays
    # uses a slightly different algorithm using numpy matrix trick
    # 1-2 orders of magnitude faster
    wait_time_start = time.time()
    with my_lock:
        wait_time.value += time.time() - wait_time_start
        start_time = time.time()


        #d = min(d, max_array[i][j]) #as GW overestimates, so if max < d we should use max as it's more accurate
        
        min_array[i][j] = d
        min_array[j][i] = d

        max_array[i][j] = d
        max_array[j][i] = d 

        uncomputed_array[i][j] = 0
        uncomputed_array[j][i] = 0
        

        D = np.ones((N))*d
        

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
        # this version doesn't update entries which have been computed
        # has not been tested/debugged yet
        ones = np.ones((N))
        
        c = np.maximum.reduce([min_array[i,:], (min_array[j,:] - D)* (uncomputed_array[i,:]), (D - max_array[j,:])*(uncomputed_array[i,:]) ])

        c = np.minimum(min_array[i,:], max_array[i,:] )#prevents min > max

        min_array[i,:] = c
        min_array[:,i] = c.T
        
        d = np.maximum.reduce([min_array[j,:], (min_array[i,:] - D)*(uncomputed_array[j,:]), (D - max_array[j,:])*(uncomputed_array[j,:]) ])

        d = np.minimum(min_array[j,:], max_array[j,:]) #prevents min > max

        min_array[j,:] = d
        min_array[:,j] = d.T        
        
        for k in range(N): 
            y = np.maximum.reduce([min_array[k,:k] , (min_array[k,i]- max_array[i,:k])*( uncomputed_array[k,:k]) ,(min_array[i,:k]- max_array[k,i])*(uncomputed_array[k,:k]) ,(min_array[k,j]- max_array[k,:k])*(uncomputed_array[k,:k]),(min_array[j,:k]- max_array[k,j] )*(uncomputed_array[k,:k]) ])

            y = np.minimum(min_array[k,:k], max_array[k,:k]) #prevents min > max

            #y = np.copy(min_array[k,:k])
            min_array[k, :k] = y
            min_array[:k, k] = y.T
            min_array[k,k] = 0
    #print('_full_update_arrays done', i,j )
        update_time.value += time.time() - start_time

        # #debugging:
        # if (min_array > max_array).any():
        #     print('_full_update_arrays ERROR, minimum > max, params:', i,j)





def _test_nearest( i):
    global N 
    global k
    global min_array 
    global max_array 
    global my_lock 
    global wait_time
    #returns j where j is the closest point to i with overlapping distance bounds to i
    # or -1 if the nearest k are guaranteed ordered


    wait_time_start = time.time()
    with my_lock:
        wait_time.value += time.time() - wait_time_start 
        nearest = [(min_array[i,j],max_array[i,j],j)    for j in range(N)]  

    nearest.sort()
    # list of the other points relative to i 

    for t in nearest:
        if t[0] > t[1]:
            print('_test_nearest ALERT, min > max,', i, t)

    if nearest[0] != (0,0,i):
        print('_test_nearest ERROR (assertion)', i)

    
    # also ensure that the maxes and mins are ordered correctly    
    for n in range(1,k+3): 
        if nearest[n][0] > nearest[n][1]:
            print('_test_nearest min max error,' , i, nearest[n][2], nearest[n][0], nearest[n][1])
            #return nearest[n][2]

    for n in range(1,k+2): 
        if nearest[n][1] >   nearest[n+1][0]:
            return nearest[n][2]




    for n in range(k+2):
        if nearest[n][1] > nearest[n+1][0]:
            print('_test_nearest ERROR (assertion)', i)

    #print("_test_nearest returns -1",i)


    #debugging:
    # a = np.copy(_find_nearest(i,N,k))
    # with my_lock:
    #     if np.array([max_array[i,int(a[-1])] > min_array[i, l] for l in set(range(k)).difference(set(a))      ]).any():
    #         print('ERROR: _test_nearest ',i, ' failed') #This is failing
    #         #print(nearest[:k +3])

    


    return -1

def _find_nearest(i,N,k):
    # returns a list of the current k nearest indices to i, in order by min 
    # WARNING does not guarantee that this is the final order
    global min_array 
    global max_array 
    global my_lock 
    global wait_time

    #print('_find_nearest called ' ,i)
    
    wait_time_start = time.time()
    with my_lock:
        wait_time.value += time.time() - wait_time_start
 
        nearest = [(min_array[i,j],max_array[i,j],j)    for j in range(N)]  
    nearest.sort()


    #debugging
    #print(len([ int(a[2]) for a in nearest[1:k+1]]))

    return( [ int(a[2]) for a in nearest[1:k+1]])


def _get_distance(i,j,dist_func):
    #print('get_distance called', i,j, getpid())

    ######
    global min_array 
    global max_array 
    global uncomputed_array 
    global dist_counter 
    global my_lock
    global dist_time
    global wait_time
    #returns the distance between points i and j
    #only updates the computed_array

    start_time = time.time()
    #d = dist_func(i,j) 
    d = min(dist_func(i,j), max_array[i,j])
    
    dist_time.value += time.time() - start_time

    wait_time_start = time.time()
    with my_lock:
        wait_time.value += time.time() - wait_time_start
        if uncomputed_array[i,j]:
            dist_counter.value += 1


        uncomputed_array[i][j] = 0
        uncomputed_array[j][i] = 0

        #debuging:
    # if max_array[i][j] < d:
    #     print('max < d',i,j)
    d = min(d , max_array[i,j] )



    if d <=0.0 and i != j:
        print("ZERO DISTANCE ERROR, ", i,j,d)
        assert(False)

    return d

def _compute_anchor(i, N,k, dist_func):
    global min_array 
    global max_array 
    global uncomputed_array 
    global kNN_array
    global my_lock
    global update_time
    global wait_time

    #print('_compute_anchor called', i)

    for j in range(N):
        d = _get_distance(i,j,dist_func)
        #d = min(_get_distance(i,j,dist_func), max_array[i,j])
        wait_time_start = time.time()
        with my_lock:
            wait_time.value += time.time() - wait_time_start
            min_array[i,j] = d
            min_array[j,i] = d
            max_array[i,j] = d
            max_array[j,i] = d
     
    ones = np.ones((N))

    wait_time_start = time.time()

    #for debugging:
    # print('start', min_array[30,71] - max_array[71,10]  , min_array[10,30], dist_func(10,30))
    # print(min_array[30,71] == dist_func(30,71)) #false
    # print(min_array[30,71] == _get_distance(30,71,dist_func)) #true
    # print(max_array[10,71] == dist_func(10,71)) #true

    # print(dist_func(10,71), _get_distance(10,71,dist_func))
    # print(dist_func(30,71), _get_distance(30,71,dist_func))
    # print(dist_func(10,30), _get_distance(10,30,dist_func))


    with my_lock:
        wait_time.value += time.time() - wait_time_start

        # if (min_array > max_array).any():
        #     print('_compute_anchor  start of updates error min > max', i)


        start_time = time.time()
        for j in range(N):
            x = np.minimum.reduce([max_array[j,:j], max_array[j][i] +max_array[i,:j]])
            max_array[j,:j] = x
            max_array[:j,j] = x.T 
                    #debugging:
        # if (min_array> max_array).any():
        #     print('anchor max update error', i,j)
   
            
        for j in range(N):
            y = np.maximum.reduce([min_array[j,:j], (min_array[j,i]- max_array[i,:j])*(uncomputed_array[j,:j] ),(min_array[i,:j]- max_array[j,i])*(uncomputed_array[j,:j] )])

            min_array[j, :j] = y
            min_array[:j, j] = y.T
            min_array[j,j] = 0

            #debugging:
            # if min_array[10,30] > 1.28620611045:
            #     print('ALERT, min_array[10,30] too big!',i,j) # at 71, 30
            #     print(min_array[j,i]- max_array[i,10]) 
            #     print(min_array[30,71] - max_array[71,10]  , min_array[10,30])
            #     # min[30,71] - max[71,10]   ~?~ min[10,30] #Something is going wrong here !!!!!!
            # print(i,j)
            # if (y > max_array[j, :j]).any():               
            #     print('anchor min update error, computed ', i,j)  #THIS IS IT - the first place the error appears
            #     break
                

            # if ((y > max_array[j, :j]) * uncomputed_array[j,:j]).any():               
            #     print('anchor min update error, uncomputed ', i,j)  #THIS IS IT - the first place the error appears
            #     assert False

        update_time.value += time.time() - start_time

        #for debugging:
    #     if (min_array > max_array).any():
    #         print('_compute_anchor  end error min > max', i)

    #     #for debugging:

    # print('end', min_array[30,71] - max_array[71,10]  , min_array[10,30])    
    # for a in range(N):
    #     for b in range(N):
    #         if min_array[a,b] > max_array[a,b]:
    #             print(i,j)
    #             print(a,b)
    #             print(uncomputed_array[a,b])

    #             print(min_array[a,b] , max_array[a,b])
    #             print( _get_distance(a,b, dist_func), dist_func(a,b))


    #             assert False
    #end debugging block


    kNN_array[i][:] = np.copy(_find_nearest(i,N,k))





    #check_bounds('end of compute_anchor',i)

def _compute_nearest(p):
    try:
        i,dist_func = p

        #print('_compute_nearest started',i)
        #print('_compute_nearest ',i, type(dist_func)) #debugging


        global N 
        global k
        global exact_distances

        # global process_min_shm
        # global process_max_shm
        # global process_uncomputed_shm 
        global min_array 
        global max_array
        global computed_array 
        global kNN_array


        global my_lock
        global wait_time


        j = _test_nearest(i) 


        
        while j !=  -1:
            try:
                #print("_compute_nearest cycle     ",i,j)
                d = _get_distance(i,j,dist_func)
                _full_update_arrays(i,j,d) 
                j = _test_nearest(i)
                #print('_compute_nearest cycle done', i,j)

            except Exception as e:
                print("_compute_nearest cycle ERROR:", i,j, 'XXXXXXXXXXXXXX')
                print(e)
                j = -1

        #print('test',i)

        #print('_compute_nearest cycling done', i)  
        a = np.copy(_find_nearest(i,N,k))

        # with my_lock:
        #     if np.array([max_array[i,int(a[-1])] > min_array[i, l] for l in set(range(k)).difference(set(a))      ]).any():
        #         print('ERROR: _compute_nearest ',i, ' failed') #This is failing
        #         print(j) # even though j = -1
        #         print(len(a),k)


        if exact_distances:
            for j in list(a):
                if uncomputed_array[i,j]:
                    d = _get_distance(i,j,dist_func)
                    wait_time_start = time.time()
                    with my_lock:
                        wait_time.value += time.time() - wait_time_start
                        min_array[i,j] = d
                        min_array[j,i] = d
                        max_array[i,j] = d
                        max_array[j,i] = d
                        #if not computed_array[i,j]:
                         #   print('_compute_nearest error, not all exact_distances computed ', i,j)
                        #if not min_array[i,j] == max_array[i,j]:
                        #    print('_compute_nearest error, difference in min vs max ', i,j)
                        if min_array[i,j] > max_array[i,j]:
                            print('_compute_nearest error,  min > max ', i,j)

                        #debugging
                        #assert not uncomputed_array[i,j] # - passes


        #print('test',i)

        #for debugging:
        # if _test_nearest(i) != -1:
        #     print('_compute_nearest error ', i)





        #debugging:             ---- Always passes now
        # with my_lock:
        #     if ([uncomputed_array[i,j] for j in a]).any():
        #         print('********XXXXXXXXXXXXXX')
        #         print(i)
        #         for j in  a:
        #             print(uncomputed_array[i,j])
        #         #print(uncomputed_array[i,9]) 
        #         print('*******XXXXXXXXXXXXXX')

        #         assert False
        #debugging block end

        #print("_compute_nearest kNN_array",i,a) #debugging

        try:
            wait_time_start = time.time()
            with my_lock:
                wait_time.value += time.time() - wait_time_start
                kNN_array[i,:] = a 
        except Exception as error:
            print('ALERT 1 -  ', error)

        #debugging:
        # if i == 52 or i == 147 or i == 163:
        #     with my_lock:
        #         print('**********',i)
        #         print(a)
        #         # print([min_array[i,j] for j in a ])
        #         # print([max_array[i,j] for j in a ])
        #         if ([min_array[i,j] for j in a ] != [max_array[i,j] for j in a ]).any():
        #             print('_compute_nearest ERROR', i)
        # #debugging block end

    except Exception as error:
        print('ALERT - ', error,i)
    #print('_compute_nearest done', i)


def run_triangle_ineq(
    N :int, # total number of points
    k: int,  # number of nearest neighbors to each point to find 
    #dist_func : callable, # distance function, takes in indices and returns the distance between those points
    base_dist_func : callable,
    num_processes  : int = 8,  #number of parallel processes to run 
    mem_code :str = None ,  # unique code to prevent shared memory overlap, randomly generated if None
    exact_distances : bool = True, #whether to calculate the exact distance of the kNN points or just return them in order
    chunksize: int = 20,
    anchor_num = 5,
    epsilon = 0,
    verbose: bool = True):

    global min_shm
    global max_shm
    global uncomputed_shm
    global kNN_shm

    global dist_counter 
    global dist_time
    global update_time
    global wait_time

    global fully_computed 
    global my_lock



    global min_array
    global max_array
    global uncomputed_array
    global kNN_array

    dist_func = lambda i, j : base_dist_func(i,j) + epsilon #for testing  

    seed = 8
    np.random.seed(seed)
     # 0-3 is often best


    my_lock = multiprocess.Lock()

    if not mem_code:
        mem_code = ''.join(random.choices(string.ascii_letters + string.digits, k=20))
        warnings.warn(message ='Cannot guarantee no shared memory interference')


    blank_array = np.zeros(shape = (N,N), dtype = float)
    bool_blank_array = np.zeros(shape = (N,N), dtype = int)


    min_shm = shared_memory.SharedMemory(name = 'min_mem' + mem_code, create=True, size=blank_array.nbytes)
    min_array = np.ndarray((N,N), buffer=min_shm.buf)
    np.copyto(dst = min_array, src =  np.zeros((N,N)) )  

    max_shm = shared_memory.SharedMemory(name = 'max_mem' + mem_code, create=True, size=blank_array.nbytes)
    max_array = np.ndarray((N,N), buffer=max_shm.buf)
    np.copyto(dst = max_array, src =  np.full((N,N), np.inf) ) #this aught to work...

    # uncomputed pairs 1 = not computed, 0 = computed
    uncomputed_shm = shared_memory.SharedMemory(name = 'uncomputed_mem' + mem_code, create=True, size=bool_blank_array.nbytes)
    uncomputed_array = np.ndarray((N,N), buffer=uncomputed_shm.buf, dtype = bool)
    np.copyto(dst = uncomputed_array, src =  np.ones((N,N), dtype = bool) )  


    kNN_shm = shared_memory.SharedMemory(name = 'kNN_mem' + mem_code, create=True, size=np.zeros(shape = (N,k), dtype = int).nbytes) #changed size 
    kNN_array = np.ndarray((N,k), dtype = int, buffer=kNN_shm.buf)
    np.copyto(dst = kNN_array, src =  np.zeros((N,k), dtype = int) )  

    #debugging
    assert uncomputed_array.all()

    for i in range(N):
        max_array[i][i] = 0
        uncomputed_array[i,i] =0
        
    dist_counter = multiprocess.Value('i',0)  #counts the number of times we've called for a pairwise distance
    dist_counter.value = 0
    dist_time = multiprocess.Value('d',0.0)  #time spent calculating pairwise distances
    dist_time.value = 0
    update_time = multiprocess.Value('d',0.0)  #time spent updating the upper/lower bound matrices
    update_time.value = 0
    wait_time = multiprocess.Value('d',0.0)  #time spent waiting to get the lock to use shared memory
    wait_time.value = 0

    if verbose:
        print('shared memory set up')


    #	compute anchors

    def _next_anchor(): #not yet tested
        #print(computed_anchors) #debugging

        if len(computed_anchors) == 0:
            return 0


        with my_lock:
            min_dists = [min([max_array[a,l] for a in computed_anchors]) for l in range(N)] #previously was max 


        #print('next anchor ' , max(range(N), key = lambda x: min_dists[x])) #debugging
        #print(max_dists) #debugging
        return max(range(N), key = lambda x: min_dists[x])



    computed_anchors = []

    for j in range(anchor_num ):
        i = _next_anchor()
        _compute_anchor(i,N,k,dist_func)
        computed_anchors.append(i)

        #debugging:
        if (min_array > max_array).any():
            print('min > max ERROR after anchor:', i)



    if verbose:
        print('anchors done')
        print(computed_anchors)
    if (min_array > max_array).any():
        print('min > max ERROR after anchors')

    with multiprocess.Pool(processes=num_processes, initializer= _init, initargs = (N,k, dist_counter, my_lock, mem_code, exact_distances, dist_time, update_time, wait_time)) as pool:
        if verbose:
            print('multiprocess started')


        results = pool.imap(_compute_nearest, zip(set(range(N)).difference(set(computed_anchors)), it.repeat(dist_func)) , chunksize = 5) 
        # https://stackoverflow.com/questions/26520781/multiprocess-pool-whats-the-difference-between-map-async-and-imap
        pool.close()
        pool.join() 


    if verbose:
        print('distance calculations done')


    # option here to iterate through everything again to check and correct


    for i in range(N):
        kNN_array[i] = _find_nearest(i,N,k) 
    output_kNN_array = np.copy(kNN_array)
    #print('output_kNN_array', output_kNN_array) #debugging

    if exact_distances:
        with my_lock:
            output_dist_array = np.zeros((N,k), dtype = float)
            for i in range(N):
                for j in range(k): 
                    a = int(output_kNN_array[i,j])

                    #debugging:
                    # if i == 52 or i == 147 or i == 163:
                    #     print('**********',i)
                    #     print(a)
                    #debugging block end



                    #commented out for debugging:

                    #assert not uncomputed_array[i,a] #failing
                    # if uncomputed_array[i,a]:
                    #     print('uncomputed',i,a)

                    # if verbose and min_array[i,a] != max_array[i,a]:
                    #     print('exact_distances error',i,j,a)
                    #     print('min i a', min_array[i,a] )
                    #     print('max i a', max_array[i,a] )



                        #assert False
                    #output_dist_array[i,j] = max_array[i,a]  #should be max if they're a disagreement - why??
                    output_dist_array[i,j] = min_array[i,a]

                    # BUG ALERT - the output here is not matching the computed distance
                    # should be fixed now, but not yet tested

    #for debuggin:
    #print(0.5*(N**2 - np.count_nonzero(uncomputed_array) - N ))

    min_shm.close()
    min_shm.unlink()
    max_shm.close()
    max_shm.unlink()
    uncomputed_shm.close()
    uncomputed_shm.unlink()
    kNN_shm.close()
    kNN_shm.unlink()

    if verbose:
        print('dist_counter = ', dist_counter.value)
        print('dist_time = ', dist_time.value)
        print('update_time = ', update_time.value)
        print('wait_time = ', wait_time.value)




    #print("dist_counter = ", dist_counter.value, ' anchor_num = ', anchor_num)  
    #roughly N * k * 1.5 distance calculations


    if exact_distances:
        return output_kNN_array, np.vectorize(lambda x : x - epsilon)(output_dist_array)
        #return output_kNN_array, output_dist_array

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



