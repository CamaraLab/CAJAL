# Helper functions
import os
from multiprocessing import RawArray
import numpy as np
from scipy.spatial.distance import squareform


def pj(*paths):
    return os.path.abspath(os.path.join(*paths))


def load_dist_mat(dist_file, return_mp=False):
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


def list_sort_files(data_dir, data_prefix=None):
    """
    Get sorted list of files in the data directory, each containing the same number of points per cell

    Args:
        data_dir (string): path to folder containing files to list
        data_prefix (string, or None): if not None, only list files starting with this prefix

    Returns:
        alphabetically sorted list of files from given folder
    """
    files_list = os.listdir(data_dir)
    files_list = [data_file for data_file in files_list
                  if data_prefix is None or data_file.startswith(data_prefix)]
    files_list.sort()  # sort the list because sometimes os.listdir() result is not sorted
    return files_list


def read_mp_array(np_array):
    """
    Convert a numpy array into an object which can be shared within multiprocessing.
    """
    # https://research.wmz.ninja/articles/2018/03/on-sharing-large-arrays-when-using-pythons-multiprocessing.html
    mp_array = RawArray('d', np_array.shape[0] * np_array.shape[1])
    np_wrapper = np.frombuffer(mp_array, dtype=np.float64).reshape(np_array.shape)
    np.copyto(np_wrapper, np_array)
    return mp_array
