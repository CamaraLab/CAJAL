# Script to calculate Gromov-Wasserstein distances,
# using algorithms in Peyre et al. ICML 2016
import os
import ot
import argparse
import time
import pickle as pkl
import itertools as it
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.neighbors import kneighbors_graph
import networkx
from multiprocessing import Pool, RawArray
import random
import sys
import time
sys.path.insert(1,os.path.dirname(os.path.abspath(__file__)) + "/QuantizedGromovWasserstein")
from quantizedGW import compressed_gw_point_cloud

pj = lambda *paths: os.path.abspath(os.path.join(*paths))

'''
TODO:

- Deal with read csv header vs not cutting out points
- Option to return GW matching between points in cells as well
- Overall function that loads distances and calculates GW in one
- Choose using QuantizedGromovWasserstein's function or my own? Probably doesn't matter
'''

def read_mp_array(np_array):
    '''
    Convert a numpy array into an object which can be shared within multiprocessing.
    '''
    # https://research.wmz.ninja/articles/2018/03/on-sharing-large-arrays-when-using-pythons-multiprocessing.html
    mp_array = RawArray('d', np_array.shape[0] * np_array.shape[1])
    np_wrapper = np.frombuffer(mp_array, dtype=np.float64).reshape(np_array.shape)
    np.copyto(np_wrapper, np_array)
    return(mp_array)

def get_distances_all(data_dir, data_prefix = None, distances_dir=None, file_prefix="",
                      num_cells=None, num_pts=None, metric = "euclidean"):
    '''
    Compute the pairwise distances in the point cloud stored in each file.
    Return list of distance matrices, or save each to a file in distances_dir.
    
    
    Parameters:
    file_prefix (string): prefix each output distance file with this string
    data_dir (string): file path to directory containing all point cloud files
                      (currently assumes a header line)
    distances_dir (string): if None (default), return list of multiprocessing array.
                           if filepath string, save distance matrices in this directory
    file_prefix (string): if distances_dir is a file path, prefix each output distance
                         file with this string
    num_cells (int): evenly subsample this many cells from the list of files
                    None (default) uses all files
    num_cells (int): evenly subsample this many points from the each cell
                    None (default) uses all points
    data_prefix (string): only read files from data_dir starting with this string
                         None (default) uses all files
    metric (string): distance metric passed into pdist()
    
    Returns:
    None (creates path to distances_dir and saves files there)
    '''
    if distances_dir is not None and not os.path.exists(distances_dir):
        os.makedirs(distances_dir)
        
    files_list = os.listdir(data_dir)
    files_list = [data_file for data_file in files_list \
                  if data_prefix is None or data_file.startswith(data_prefix)]
    files_list.sort() # sort the list because sometimes os.listdir() result is not sorted
    if num_cells is not None:
        if len(files_list) < num_cells:
            raise ValueError("There are fewer than " + str(num_cells) + " cells in data dir")
        files_list = [files_list[i] for i in np.linspace(0,len(files_list)-1,num_cells).astype("uint32")]
    
    return_list = []
    for data_file in files_list:
        X = pd.read_csv(pj(data_dir,data_file))
        if num_pts is not None:
            if X.shape[0] < num_pts:
                raise ValueError("There are fewer than " + str(num_pts) + " points in data dir files")
            X = X.iloc[np.linspace(0,X.shape[0]-1,num_pts).astype("uint32"),:]
        D = pdist(X, metric=metric)

        if distances_dir is None:
            return_list.append(read_mp_array(squareform(D)))
        else:
            outfile =  file_prefix+"_"+data_file.replace(".csv","")+"_dist.txt" \
                        if file_prefix != "" else data_file.replace(".csv","")+"_dist.txt"
            np.savetxt(pj(distances_dir, outfile),
                   D, fmt='%.8f')
            return_list.append(outfile)
    return return_list
        
def load_distances_global(distances_dir, data_prefix = None):
    '''
    Load distance matrices from directory into list of arrays
    that can be shared with the multiprocessing pool.
    
    Parameters: 
    distances_dir (string): input directory where distance files are saved
    data_prefix (string): only read files from distances_dir starting with this string
    
    Returns: list of multiprocessing arrays containing distance matrix for each cell
    '''
    files_list = os.listdir(distances_dir)
    files_list = [data_file for data_file in files_list\
                  if data_prefix is None or data_file.startswith(data_prefix)]
    files_list.sort() # sort the list because sometimes os.listdir() result is not sorted
    
    dist_mat_list = [read_mp_array(squareform(np.loadtxt(pj(distances_dir,dist_file))))\
                     for dist_file in files_list]
    return dist_mat_list

def calculate_gw_preload_global(arguments):
    '''
    Compute GW distance between two distance matrices.
    Meant to be called within a multiprocessing pool where dist_mat_list exists globally
    
    Parameters:
    arguments (list): 
        i1 (int): index in the dist_mat_list for the first distance matrix
        i2 (int): index in the dist_mat_list for the second distance matrix
        return_mat (boolean): if True, returns transportation matrix between points
                            if False, only returns GW distance
    Returns:
    int: GW distance
    '''
    i1, i2, return_mat = arguments
    D1 = np.frombuffer(dist_mat_list[i1])
    numpts = int(np.sqrt(D1.shape))
    D1 = D1.reshape((numpts,numpts))
    D2 = np.frombuffer(dist_mat_list[i2]).reshape((numpts,numpts))
    
    gw, log = ot.gromov.gromov_wasserstein(
            D1, D2, ot.unif(D1.shape[0]), ot.unif(D2.shape[0]),
            'square_loss', log=True)
    if return_mat:
        return (log['gw_dist'],gw)
    else:
        return log['gw_dist']

def calculate_gw_quantized_preload_global(arguments):
    '''
    Compute quantized GW distance between two distance matrices using
    https://github.com/trneedham/QuantizedGromovWasserstein
    Meant to be called within a multiprocessing pool where dist_mat_list exists globally
    
    Parameters:
    arguments (list): 
        i1 (int): index in the dist_mat_list for the first distance matrix
        i2 (int): index in the dist_mat_list for the second distance matrix
        sample_size (int): subset of points for QGW to run GW on
        return_mat (boolean): if True, returns transportation matrix between points
                            if False, only returns GW distance
        
    Returns:
    int: GW distance
    '''
    i1, i2, sample_size, return_mat = arguments
    D1 = np.frombuffer(dist_mat_list[i1])
    numpts = int(np.sqrt(D1.shape))
    D1 = D1.reshape((numpts,numpts))
    D2 = np.frombuffer(dist_mat_list[i2]).reshape((numpts,numpts))
    
    node_subset1 = np.linspace(0,D1.shape[0]-1,sample_size).astype("uint32")
    node_subset2 = np.linspace(0,D2.shape[0]-1,sample_size).astype("uint32")
    
    p = ot.unif(D1.shape[0])
    q = ot.unif(D2.shape[0])
    
    res = compressed_gw_point_cloud(D1,D2,p,q,
                                    node_subset1,node_subset2, verbose = False, return_dense = True)
    constC, hC1, hC2 = ot.gromov.init_matrix(D1, D2, p, q,"square_loss")
    if return_mat:
        return (ot.gromov.gwloss(constC, hC1, hC2, res),res)
    else:
        return ot.gromov.gwloss(constC, hC1, hC2, res)

def init_fn(dist_mat_list_):
    '''
    Initialization function sets dist_mat_list to be global in multiprocessing pool.
    '''
    global dist_mat_list
    dist_mat_list = dist_mat_list_

def distance_matrix_preload_global(dist_mat_list, file_prefix, gw_results_dir, save_mat=False,
                    quantized=False, sample_size=100):
    '''
    Calculate the GW distance between every pair of distance matrices
    
    Parameters:
    dist_mat_list (list): list of multiprocessing arrays containing distance matrix for each cell
    file_prefix (string): name of output file to write GW distance matrix to
    gw_results_dir (string): path to directory to write output file to
    save_mat (boolean): if True, returns transportation matrix (matching) between points
                        if False, only returns GW distance
    quantized (boolean): if True, run quantized GW (QGW)
    sample_size (int): if quantized is True, subset of points for QGW to run GW on
    
    Results:
    None (writes distance matrix of GW distances to file)
    '''
    if not os.path.exists(gw_results_dir):
        os.makedirs(gw_results_dir)
            
    pool = Pool(processes=12, initializer = init_fn, initargs = (dist_mat_list,))
    arguments = list(it.combinations(range(len(dist_mat_list)),2))
    
    if quantized:
        arguments_quantized = [x + (sample_size,save_mat) for x in arguments]
        dist_results = pool.map(calculate_gw_quantized_preload_global, arguments_quantized)
    else:
        arguments = [x + (save_mat,) for x in arguments]
        dist_results = pool.map(calculate_gw_preload_global, arguments)
    if save_mat:
        np.savetxt(pj(gw_results_dir, file_prefix+"_gw_dist_mat.txt"),
                   np.array([res[0] for res in dist_results]), fmt='%.8f')
        np.savez_compressed(pj(gw_results_dir, file_prefix+"_gw_matching.npz"),*[res[1] for res in dist_results])
    else:
        np.savetxt(pj(gw_results_dir, file_prefix+"_gw_dist_mat.txt"), np.array(dist_results), fmt='%.8f')
