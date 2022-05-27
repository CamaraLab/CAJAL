# Script to calculate Gromov-Wasserstein distances,
# using algorithms in Peyre et al. ICML 2016
import os
import ot
import itertools as it
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from multiprocessing import Pool, RawArray
# sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)) + "/QuantizedGromovWasserstein")
# from quantizedGW import compressed_gw_point_cloud

'''
TODO:

- Deal with read csv header vs not cutting out points
- Option to return GW matching between points in cells as well
- Overall function that loads distances and calculates GW in one
- Choose using QuantizedGromovWasserstein's function or my own? Probably doesn't matter
- Tests
    - num_pts should be same in each input file and distance matrix
    - no distances should be 0 except diagonal (might not be requirement)
    - **distance matrix files should be in upper/lower triangle format
'''


def pj(*paths):
    return os.path.abspath(os.path.join(*paths))


def read_mp_array(np_array):
    """
    Convert a numpy array into an object which can be shared within multiprocessing.
    """
    # https://research.wmz.ninja/articles/2018/03/on-sharing-large-arrays-when-using-pythons-multiprocessing.html
    mp_array = RawArray('d', np_array.shape[0] * np_array.shape[1])
    np_wrapper = np.frombuffer(mp_array, dtype=np.float64).reshape(np_array.shape)
    np.copyto(np_wrapper, np_array)
    return mp_array


def get_distances_one(data_file, num_pts=None, metric="euclidean", return_mp=True, header=None):
    """
    Compute the pairwise distances in the point cloud stored in the file.
    Return distance matrix as numpy array or mp (multiprocessing) array


    Args:
        data_file (string): file path to point cloud file
                          (currently assumes a header line)
        num_pts (int): evenly subsample this many points from each cell
                        None (default) uses all points
        metric (string): distance metric passed into pdist()
        return_mp (boolean): only used of distances_dir is None.
                            if True, return multiprocessing array, if False return numpy array
        header (boolean): passed into read_csv, whether data file has a header line

    Returns:
        None (creates path to distances_dir and saves files there)
    """
    coords = pd.read_csv(data_file, header=header)
    # Evenly sample a subset of points (optional)
    if num_pts is not None:
        if coords.shape[0] < num_pts:
            raise ValueError("There are fewer than " + str(num_pts) + " points in data dir files")
        coords = coords.iloc[np.linspace(0, coords.shape[0] - 1, num_pts).astype("uint32"), :]
    dist_mat = pdist(coords, metric=metric)

    # Return either as numpy array or mp (multiprocessing) array
    if return_mp:
        return_dist = read_mp_array(squareform(dist_mat))
    else:
        return_dist = squareform(dist_mat)
    return return_dist


def save_distances_one(data_file, num_pts=None, distances_dir=None, file_prefix="",
                       metric="euclidean", header=None):
    """
    Not currently used, kept as legacy
    Compute the pairwise distances in the point cloud stored in the file.
    Save each to a file in distances_dir.


    Args:
        data_file (string): file path to point cloud file
                          (currently assumes a header line)
        distances_dir (string): if None (default), return list of multiprocessing array.
                               if filepath string, save distance matrices in this directory
        file_prefix (string): if distances_dir is a file path, prefix each output distance
                             file with this string
        num_pts (int): evenly subsample this many points from each cell
                        None (default) uses all points
        metric (string): distance metric passed into pdist()
        header (boolean): passed into read_csv, whether data file has a header line

    Returns:
        None (creates path to distances_dir and saves files there)
    """
    coords = pd.read_csv(data_file, header=header)
    # Evenly sample a subset of points (optional)
    if num_pts is not None:
        if coords.shape[0] < num_pts:
            raise ValueError("There are fewer than " + str(num_pts) + " points in data dir files")
        coords = coords.iloc[np.linspace(0, coords.shape[0] - 1, num_pts).astype("uint32"), :]
    dist_mat = pdist(coords, metric=metric)

    # Return distance matrices in list, except
    # save distance matrices to the file path of distances_dir
    outfile = file_prefix + "_" + data_file.replace(".csv", "") + "_dist.txt" \
        if file_prefix != "" else data_file.replace(".csv", "") + "_dist.txt"
    np.savetxt(pj(distances_dir, outfile),
               dist_mat, fmt='%.8f')
    return outfile


def get_distances_all(data_dir, data_prefix=None, distances_dir=None,
                      num_cells=None, num_pts=None, metric="euclidean",
                      return_mp=True, header=None):
    """
    Compute the pairwise distances in the point cloud stored in each file.
    Return list of distance matrices, or save each to a file in distances_dir.
    
    
    Args:
        data_dir (string): file path to directory containing all point cloud files
                          (currently assumes a header line)
        data_prefix (string): only read files from data_dir starting with this string
                             None (default) uses all files
        distances_dir (string): if None (default), return list of multiprocessing array.
                               if filepath string, save distance matrices in this directory
        num_cells (int): evenly subsample this many cells from the list of files
                        None (default) uses all files
        num_pts (int): evenly subsample this many points from each cell
                        None (default) uses all points
        metric (string): distance metric passed into pdist()
        return_mp (boolean): only used of distances_dir is None.
                            if True, return multiprocessing array, if False return numpy array
        header (boolean): passed into read_csv, whether data file has a header line
    
    Returns:
        None (creates path to distances_dir and saves files there)
    """
    if distances_dir is not None and not os.path.exists(distances_dir):
        os.makedirs(distances_dir)
        
    # Get sorted list of files in the data directory, each containing the same number of points per cell
    files_list = os.listdir(data_dir)
    files_list = [data_file for data_file in files_list
                  if data_prefix is None or data_file.startswith(data_prefix)]
    files_list.sort()  # sort the list because sometimes os.listdir() result is not sorted
    
    # Evenly sample a subset of cells (optional)
    if num_cells is not None:
        if len(files_list) < num_cells:
            raise ValueError("There are fewer than " + str(num_cells) + " cells in data dir")
        files_list = [files_list[i] for i in np.linspace(0, len(files_list)-1, num_cells).astype("uint32")]
    
    # Compute pairwise distance between points in each file
    return_list = [get_distances_one(pj(data_dir, data_file), num_pts=num_pts,
                                     metric=metric, return_mp=return_mp, header=header)
                   for data_file in files_list]
    check_num_pts = all([len(x) == len(return_list[0]) for x in return_list])
    if not check_num_pts:
        raise Exception("Point cloud data files do not have same number of points")
    return return_list


def load_dist_mat(dist_file, return_mp=True):
    """
    Load distance matrix from a file, potentially as multiprocessing array
    Distances are assumed to be in vector form (upper or lower tri of symmetric matrix)
    as output by https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.squareform.html

    Args:
        dist_file (string): path to file with distance matrix saved in vector format
        return_mp (boolean): if True, return multiprocessing array, if False return numpy array

    Returns:
        distance matrix
    """
    try:
        dist_vec = np.loadtxt(dist_file)
    except ValueError:
        raise Exception("Distance files must be in vector form with one value per line")
    if len(dist_vec.shape) > 1 and dist_vec.shape[1] != 1:
        raise Exception("Distance files must be in vector form as output by squareform()")
    dist_mat = squareform(dist_vec)
    if return_mp:
        dist_mat = read_mp_array(dist_mat)
    return dist_mat


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
    # Get sorted list of files, each containing pairwise distances between points in a cell
    files_list = os.listdir(distances_dir)
    files_list = [data_file for data_file in files_list
                  if data_prefix is None or data_file.startswith(data_prefix)]
    files_list.sort()  # sort the list because sometimes os.listdir() result is not sorted

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


def calculate_gw_quantized_preload_global(arguments):
    """
    Compute quantized GW distance between two distance matrices using
    https://github.com/trneedham/QuantizedGromovWasserstein
    Meant to be called within a multiprocessing pool where dist_mat_list exists globally
    
    Args:
        arguments (list):
            i1 (int): index in the dist_mat_list for the first distance matrix
            i2 (int): index in the dist_mat_list for the second distance matrix
            sample_size (int): subset of points for QGW to run GW on
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
    
    # Get even subset of points for quantized GW
    node_subset1 = np.linspace(0, d1.shape[0]-1, sample_size).astype("uint32")
    node_subset2 = np.linspace(0, d2.shape[0]-1, sample_size).astype("uint32")
  
    # Compute quantized Gromov-Wasserstein matching coupling matrix
    res = compressed_gw_point_cloud(d1, d2, ot.unif(d1.shape[0]), ot.unif(d2.shape[0]),
                                    node_subset1, node_subset2, verbose=False, return_dense=True)
    # Intermediate step to getting GW distance from coupling matrix
    constc, hc1, hc2 = ot.gromov.init_matrix(d1, d2, ot.unif(d1.shape[0]), ot.unif(d2.shape[0]), "square_loss")
    # Compute GW distance and return distance and/or coupling matrix
    if return_mat:
        return ot.gromov.gwloss(constc, hc1, hc2, res), res
    else:
        return ot.gromov.gwloss(constc, hc1, hc2, res)


def init_fn(dist_mat_list_, save_mat, q_sample_size):
    """
    Initialization function sets dist_mat_list to be global in multiprocessing pool.
    Also sets other arguments because I couldn't figure out how to lazily modify an iterator
    """
    global dist_mat_list, return_mat, sample_size
    dist_mat_list = dist_mat_list_
    return_mat = save_mat
    sample_size = q_sample_size


def distance_matrix_preload_global(dist_mat_list_, save_mat=False,
                                   quantized=False, q_sample_size=100,
                                   num_cores=12, chunk_size=100):
    """
    Calculate the GW distance between every pair of distance matrices
    
    Args:
        dist_mat_list_ (list): list of multiprocessing or numpy arrays containing distance matrix for each cell
        save_mat (boolean): if True, returns coupling matrix (matching) between points
                            if False, only returns GW distance
        quantized (boolean): if True, run quantized GW (QGW)
        q_sample_size (int): if quantized is True, subset of points for QGW to run GW on
        num_cores (int): number of parallel processes to run GW in
        chunk_size (int): chunk size for the iterator of all pairs of cells
            larger size is faster but takes more memory, see multiprocessing pool.imap() for details
    
    Returns:
        None (writes distance matrix of GW distances to file)
    """
    if num_cores > 1:
        # Start up multiprocessing w/ list of distance matrices in global environment
        pool = Pool(processes=num_cores, initializer=init_fn, initargs=(dist_mat_list_, save_mat, q_sample_size))
    else:
        # Set dist_mat_list in global environment so can call the same functions
        global dist_mat_list, return_mat, sample_size
        dist_mat_list = dist_mat_list_
        return_mat = save_mat
        sample_size = q_sample_size
        
    # Get all pairs of entries in the list
    arguments = it.combinations(range(len(dist_mat_list_)), 2)
    # n = len(dist_mat_list_)
    # num_comb = n*n/2 - n/2
    # chunk_size = int(num_comb/num_cores/10000)
    # Compute Gromov-Wasserstein distance and/or coupling matrix
    if quantized:  # Run quantized Gromov-Wasserstein
        if num_cores > 1:
            dist_results = pool.map(calculate_gw_quantized_preload_global, arguments)
        else:
            dist_results = list(map(calculate_gw_quantized_preload_global, arguments))
    else:  # Run normal Gromov-Wasserstein
        if num_cores > 1:
            # dist_results = pool.map(calculate_gw_preload_global, arguments)
            dist_results = list(pool.imap(calculate_gw_preload_global, arguments, chunksize=chunk_size))
        else:
            dist_results = list(map(calculate_gw_preload_global, arguments))
            
    if num_cores > 1:  # stop Pool
        pool.close()
        pool.join()
        
    return dist_results


def save_dist_mat_preload_global(dist_mat_list_, file_prefix, gw_results_dir, save_mat=False,
                                 quantized=False, q_sample_size=100, num_cores=12, chunk_size=100):
    """
    Save the GW distance between each pair of distance matrices in vector form

    Args:
        dist_mat_list_ (list): list of multiprocessing or numpy arrays containing distance matrix for each cell
        file_prefix (string): name of output file to write GW distance matrix to
        gw_results_dir (string): path to directory to write output file to
        save_mat (boolean): if True, returns coupling matrix (matching) between points
                            if False, only returns GW distance
        quantized (boolean): if True, run quantized GW (QGW)
        q_sample_size (int): if quantized is True, subset of points for QGW to run GW on
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
                                                  quantized=quantized, q_sample_size=q_sample_size,
                                                  num_cores=num_cores, chunk_size=chunk_size)

    # Save results - suffix name of output files is currently hardcoded
    if save_mat:
        np.savetxt(pj(gw_results_dir, file_prefix + "_gw_dist_mat.txt"),
                   np.array([res[0] for res in dist_results]), fmt='%.8f')
        np.savez_compressed(pj(gw_results_dir, file_prefix + "_gw_matching.npz"), *[res[1] for res in dist_results])
    else:
        np.savetxt(pj(gw_results_dir, file_prefix + "_gw_dist_mat.txt"), np.array(dist_results), fmt='%.8f')
