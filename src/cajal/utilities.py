# Helper functions
import os
from multiprocessing import RawArray
import numpy as np
import numpy.typing as npt
import csv
from scipy.spatial.distance import squareform
from tinydb import TinyDB
import itertools as it
import math
from typing import Tuple, List, Iterator, Optional

def pj(*paths):
    return os.path.abspath(os.path.join(*paths))


def load_dist_mat(dist_file: str ) -> npt.NDArray[np.float_]:
    """
    Load distance matrix from a file.
    Distances in the file are assumed to be in vector form (upper or lower tri of symmetric matrix)
    as output by \
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.squareform.html

    Args:
        dist_file (string): path to file with distance matrix saved in vector format
    Returns:
        distance matrix as numpy array
    """
    try:
        dist_vec = np.loadtxt(dist_file)
    except ValueError:
        raise Exception("Distance files must be in vector form with one value per line")
    if len(dist_vec.shape) > 1 and dist_vec.shape[1] != 1:
        raise Exception("Distance files must be in vector form as output by squareform()")
    return squareform(dist_vec)

def list_sort_files(data_dir, data_prefix=None, data_suffix=None):
    """
    Get sorted list of files in the data_dir directory.

    Args:
        data_dir (string): path to folder containing files to list
        data_prefix (string, or None): if not None, only list files starting with this prefix
        data_suffix (string, or None): if not None, only list files starting with this prefix
    Returns:
        alphabetically sorted list of files from given folder
    """
    files_list = os.listdir(data_dir)
    if data_prefix is not None:
        if data_suffix is not None:
            files_list = [data_file for data_file in files_list if data_file.startswith(data_prefix) and data_file.endswith(data_suffix)]
        else:
            files_list = [data_file for data_file in files_list if data_file.startswith(data_prefix)]
    elif data_suffix is not None:
        files_list = [data_file for data_file in files_list if data_file.endswith(data_suffix)]

    files_list.sort()
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

def write_csv_block(
        out_csv : str,
        dist_mats : Iterator[Tuple[str, Optional[npt.NDArray[np.float_]]]],
        batch_size : int = 1000
) -> List[str]:
    failed_cells : List[str] = []
    with open(out_csv, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        name, arr = next(dist_mats)
        length=arr.shape[0]
        sidelength=math.ceil(math.sqrt(2*length))
        assert(sidelength*(sidelength-1)==length*2)
        firstline = [ "cell_id" ] + [ "d_%d_%d" % (i , j) for i,j in it.combinations(range(sidelength),2)]
        csvwriter.writerow(firstline)
        csvwriter.writerow([name]+arr.tolist())
        while(next_batch := list(it.islice(dist_mats, batch_size))):
            good_cells : List[[List[str | float]]] = []
            for name, cell in next_batch:
                if cell is None:
                    failed_cells.append(name)
                else:
                    good_cells.append( [ name ] + cell.tolist())
        csvwriter.writerows(good_cells)
    return failed_cells

def write_tinydb_block(
        output_db : TinyDB,
        dist_mats : Iterator[Tuple[str, Optional[npt.NDArray[np.float_]]]],
        batch_size : int = 1000
) -> List[str]:

    failed_cells : List[str] = []
    while(next_batch := list(it.islice(dist_mats, batch_size))):
        good_cells : List[Tuple[str,List[float]]] = []
        for name, cell in next_batch:
            if cell is None:
                failed_cells.append(name)
            else:
                good_cells.append((name, cell.tolist()))
        output_db.insert_multiple({ 'name' : name, 'cell' : cell} for name, cell in good_cells)
    return failed_cells
