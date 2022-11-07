# Script to calculate Gromov-Wasserstein distances,
# using algorithms in Peyre et al. ICML 2016
import os
import ot
import itertools as it
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from multiprocessing import Pool

from CAJAL.lib.utilities import pj, load_dist_mat, list_sort_files, read_mp_array

'''
TODO:

- Deal with read csv header vs not cutting out points
- Option to return GW matching between points in cells as well
- Overall function that loads distances and calculates GW in one
- Tests
    - num_pts should be same in each input file and distance matrix
    - no distances should be 0 except diagonal (might not be requirement)
    - **distance matrix files should be in upper/lower triangle format
'''


def get_distances_one(data_file, metric="euclidean", return_mp=True, header=None):
    """
    Compute the pairwise distances in the point cloud stored in the file.
    Return distance matrix as numpy array or mp (multiprocessing) array

    Args:
        data_file (string): file path to point cloud file
                          (currently assumes a header line)
        metric (string): distance metric passed into pdist()
        return_mp (boolean): if True, return multiprocessing array, if False return numpy array
        header (boolean): passed into read_csv, whether data file has a header line

    Returns:
        Either a numpy array or a multiprocessing array, depending on the value of the flag return_mp.
    """
    
    coords = pd.read_csv(data_file, header=header)
    dist_mat = pdist(coords, metric=metric)

    # Return either as numpy array or mp (multiprocessing) array
    try: 
        return_dist = squareform(dist_mat)
    except Exception as err:
        print(err)
        print("Scipy raised an error while computing intracell distances in ", data_file)
        print("Check that this file is correctly formatted.")
        raise
        
    if return_mp:
        return_dist = read_mp_array(return_dist)
    return return_dist

# def save_distances_one(data_file, distances_dir=None, file_prefix="",
#                        metric="euclidean", header=None):
#     """
#     Not currently used, kept as legacy
#     Compute the pairwise distances in the point cloud stored in the file.
#     Save each to a file in distances_dir.


#     Args:
#         data_file (string): file path to point cloud file
#                           (currently assumes a header line)
#         distances_dir (string): if None (default), return list of multiprocessing array.
#                                if filepath string, save distance matrices in this directory
#         file_prefix (string): if distances_dir is a file path, prefix each output distance
#                              file with this string
#         metric (string): distance metric passed into pdist()
#         header (boolean): passed into read_csv, whether data file has a header line

#     Returns:
#         None (creates path to distances_dir and saves files there)
#     """
#     coords = pd.read_csv(data_file, header=header)
#     dist_mat = pdist(coords, metric=metric)

#     # Return distance matrices in list, except
#     # save distance matrices to the file path of distances_dir
#     outfile = file_prefix + "_" + data_file.replace(".csv", "") + "_dist.txt" \
#         if file_prefix != "" else data_file.replace(".csv", "") + "_dist.txt"
#     np.savetxt(pj(distances_dir, outfile),
#                dist_mat, fmt='%.8f')
#     return outfile


def get_distances_all(data_dir, data_prefix=None, data_suffix="csv",
                      #distances_dir=None,
                      metric="euclidean", return_mp=True, header=None):
    """
    Compute the pairwise distances in the point cloud stored in each file.
    Return list of distance matrices.
        (TODO : Add support for a flag "distances_dir" which will enable the user
         to write the list of distance matrices in addition to / rather than returning it.)
    
    
    Args:
        data_dir (string): file path to directory containing all point cloud files
                          (currently assumes a header line)
        data_prefix (string): only read files from data_dir starting with this string
                             None (default) uses all files
        distances_dir (string): if None (default), return list of multiprocessing array.
                               if filepath string, save distance matrices in this directory
        metric (string): distance metric passed into pdist()
        return_mp (boolean): only used of distances_dir is None.
                            if True, return multiprocessing array, if False return numpy array
        header (boolean): passed into read_csv, whether data file has a header line
    
    Returns:
        List of distance matrices.
        (In the future, will be a list of distance matrices or None,
         in the case where the distances_dir flag is enabled.)

    """
    # if distances_dir is not None and not os.path.exists(distances_dir):
    #     os.makedirs(distances_dir)

    files_list = list_sort_files(data_dir, data_prefix)
    
    # Compute pairwise distance between points in each file
    return_list = [get_distances_one(pj(data_dir, data_file), metric=metric, return_mp=return_mp, header=header)
                   for data_file in files_list]
    check_num_pts = all([len(x) == len(return_list[0]) for x in return_list])
    if not check_num_pts:
        raise Exception("Point cloud data files do not have same number of points")
    return return_list


def load_distances_global(distances_dir, data_prefix=None, return_mp=True):
    """
    Load distance matrices from directory into list of arrays
    that can be shared with the multiprocessing pool.
    
    Args: 
        distances_dir (string): input directory where distance files are saved
        data_prefix (string): only read files from distances_dir starting with this string
        return_mp (boolean): if True, return multiprocessing array, if False return numpy array
    
    Returns:
        list of multiprocessing arrays containing distance matrix for each cell
    """
    files_list = list_sort_files(distances_dir, data_prefix)
    return [load_dist_mat(pj(distances_dir, dist_file), return_mp=return_mp) for dist_file in files_list]


def calculate_gw_preload_global(arguments):
    """
    Compute GW distance between two distance matrices.
    Meant to be called within a multiprocessing pool where dist_mat_list exists globally
    
    Args:
        arguments (list):
            i1 (int): index in the dist_mat_list for the first distance matrix
            i2 (int): index in the dist_mat_list for the second distance matrix
            return_mat (boolean): if True, returns coupling matrix between points
                                if False, only returns GW distance
    Returns:
        int: GW distance
    """
    # Get distance matrices from global list (this saves memory so it's not copied in each process)
    i1, i2 = arguments
    d1 = np.frombuffer(dist_mat_list[i1])
    numpts = int(np.sqrt(d1.shape))  # mp_arrays are in vector form, we know this is a square matrix
    d1 = d1.reshape((numpts, numpts))
    d2 = np.frombuffer(dist_mat_list[i2]).reshape((numpts, numpts))
    # Compute Gromov-Wasserstein matching coupling matrix and distance
    gw, log = ot.gromov.gromov_wasserstein(
            d1, d2, ot.unif(d1.shape[0]), ot.unif(d2.shape[0]),
            'square_loss', log=True)
    if return_mat:
        return log['gw_dist'], gw
    else:
        return log['gw_dist']


def init_fn(dist_mat_list_, save_mat):
    """
    Initialization function sets dist_mat_list to be global in multiprocessing pool.
    Also sets other arguments because I couldn't figure out how to lazily modify an iterator
    """
    global dist_mat_list, return_mat
    dist_mat_list = dist_mat_list_
    return_mat = save_mat


def distance_matrix_preload_global(dist_mat_list_, save_mat=False, num_cores=12, chunk_size=100):
    """
    Calculate the GW distance between every pair of distance matrices
    
    Args:
        dist_mat_list_ (list): list of multiprocessing or numpy arrays containing distance matrix for each cell
        save_mat (boolean): if True, returns coupling matrix (matching) between points
                            if False, only returns GW distance
        num_cores (int): number of parallel processes to run GW in
        chunk_size (int): chunk size for the iterator of all pairs of cells
            larger size is faster but takes more memory, see multiprocessing pool.imap() for details
    
    Returns:
        None (writes distance matrix of GW distances to file)
    """
    arguments = it.combinations(range(len(dist_mat_list_)), 2)

    if num_cores > 1:
        # Start up multiprocessing w/ list of distance matrices in global environment
        with Pool(processes=num_cores, initializer=init_fn, initargs=(dist_mat_list_, save_mat)) as pool:
            dist_results = list(pool.imap(calculate_gw_preload_global, arguments, chunksize=chunk_size))
    else:
        # Set dist_mat_list in global environment so can call the same functions
        global dist_mat_list, return_mat
        dist_mat_list = dist_mat_list_
        return_mat = save_mat
        dist_results = list(map(calculate_gw_preload_global, arguments))
    return dist_results


def save_dist_mat_preload_global(dist_mat_list_, file_prefix, gw_results_dir,
                                 save_mat=False, num_cores=12, chunk_size=100):
    """
    Save the GW distance between each pair of distance matrices in vector form

    Args:
        dist_mat_list_ (list): list of multiprocessing or numpy arrays containing distance matrix for each cell
        file_prefix (string): name of output file to write GW distance matrix to
        gw_results_dir (string): path to directory to write output file to
        save_mat (boolean): if True, returns coupling matrix (matching) between points
                            if False, only returns GW distance
        num_cores (int): number of parallel processes to run GW in
        chunk_size (int): chunk size for the iterator of all pairs of cells
            larger size is faster but takes more memory, see multiprocessing pool.imap() for details

    Returns:
        None (writes distance matrix of GW distances to file)
    """
    # Create output directory
    if not os.path.exists(gw_results_dir):
        os.makedirs(gw_results_dir)

    dist_results = distance_matrix_preload_global(dist_mat_list_, save_mat=save_mat,
                                                  num_cores=num_cores, chunk_size=chunk_size)

    # Save results - suffix name of output files is currently hardcoded
    if save_mat:
        np.savetxt(pj(gw_results_dir, file_prefix + "_gw_dist_mat.txt"),
                   np.array([res[0] for res in dist_results]), fmt='%.8f')
        np.savez_compressed(pj(gw_results_dir, file_prefix + "_gw_matching.npz"), *[res[1] for res in dist_results])
    else:
        np.savetxt(pj(gw_results_dir, file_prefix + "_gw_dist_mat.txt"), np.array(dist_results), fmt='%.8f')
