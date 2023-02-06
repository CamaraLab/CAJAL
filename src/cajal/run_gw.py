"""
Functionality to compute Gromov-Wasserstein distances\
using algorithms in Peyre et al. ICML 2016
"""
# import os
# std lib dependencies
import itertools as it
import time
import csv
# import ctypes
from typing import List, Optional, Tuple, Iterable, Iterator, Dict, TypedDict, TypeVar, Callable
# from multiprocessing import Pool
from pathos.pools import ProcessPool
from multiprocess.shared_memory import SharedMemory
from multiprocess import set_start_method
from math import sqrt, ceil
import statistics
#external dependencies
import ot
import numpy as np
import numpy.typing as npt
# import pandas as pds
# from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.sparse import coo_matrix
from tinydb import TinyDB

# cajal dependencies
# from .utilities import pj load_dist_mat, list_sort_files, read_mp_array

# TODO
# - Deal with read csv header vs not cutting out points
# - Option to return GW matching between points in cells as well
# - Overall function that loads distances and calculates GW in one
# - Tests
#     - num_pts should be same in each input file and distance matrix
#     - no distances should be 0 except diagonal (might not be requirement)
#     - **distance matrix files should be in upper/lower triangle format

# def compute_intracell_distances_one(
#         data_file: str,g
#         metric: str ="euclidean",
#         return_mp: bool =True,
#         header: Optional[int | List[int]] = None) ->  npt.NDArray[np.float_] :
#     """
#     Compute the pairwise distances in the point cloud stored in a \*.csv file.
#     Return distance matrix as numpy array
#     :param data_file: file path to point cloud file (currently assumes a header line)
#         * metric (string): distance metric passed into pdist()
#         * return_mp (boolean): if True, return multiprocessing array, if False return numpy array
#         * header: If the \*.csv file has a row header labelling the columns,
#              use this field to label it, see :func:`pandas.read_csv` for details.
#     Returns:
#         A multiprocessing array, if return_mp == True; else a numpy array.
#     """
#     coords = pd.read_csv(data_file, header=header)
#     dist_mat = pdist(coords, metric=metric)
#     # Return either as numpy array or mp (multiprocessing) array
#     try:
#         return_dist = squareform(dist_mat)
#     except Exception as err:
#         print(err)
#         print("Scipy raised an error while computing intracell distances in ", data_file)
#         print("Check that this file is correctly formatted.")
#         raise

#     if return_mp:
#         return_dist = read_mp_array(return_dist)
#     return return_dist

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


# def compute_intracell_distances_all(
#         data_dir: str,
#         data_prefix: Optional[str] =None,
#         data_suffix: str ="csv",
#         #distances_dir=None,
#         metric: str ="euclidean",
#         return_mp: bool =True,
#         header: Optional[int | List[int]] = None):
#     """
#     Compute the pairwise distances in the point cloud stored in each file.
#     Return list of distance matrices.

#     Args:
#         * data_dir (string): file path to directory containing all \
#                 point cloud files (currently assumes a header line)
#         * data_prefix (string): only read files from data_dir \
#               starting with this string. None (default) uses \
#               all files
#         * metric (string): distance metric passed into pdist()
#         * return_mp (boolean): only used of distances_dir is None.\
#               If True, return multiprocessing array,\
#               if False return numpy array
#         * header: If the \*.csv file has a row header labelling the\
#               columns, use this field to label it, see\
#               :func:`pandas.read_csv` for details.

#     Returns:
#         List of distance matrices. (In the future, will be a list \
#             of distance matrices or None, in the case where \
#             the distances_dir flag is enabled.)

#     """

#     # if distances_dir is not None and not os.path.exists(distances_dir):
#     #     os.makedirs(distances_dir)

#     # (TODO : Add support for a flag "distances_dir" which will enable the user
#     #  to write the list of distance matrices in addition to / rather than returning it.)

#     files_list = list_sort_files(data_dir, data_prefix, data_suffix=data_suffix)

#     # Compute pairwise distance between points in each file
#     return_list = [compute_intracell_distances_one(pj(data_dir, data_file),
#                                                metric=metric, return_mp=return_mp, header=header)
#                    for data_file in files_list]
#     check_num_pts = all([len(x) == len(return_list[0]) for x in return_list])
#     if not check_num_pts:
#         raise Exception("Point cloud data files do not have same number of points")
#     return return_list


# def load_intracell_distances(distances_dir : str,
#                                 data_prefix: str =None):
#     """
#     Load distance matrices from directory into list of numpy arrays.

#     Args:
#         distances_dir (string): input directory where distance files are saved
#         data_prefix (string): only read files from distances_dir starting with this string
#     Returns:
#         list of numpy arrays containing distance matrix for each cell
#     """

#     files_list = list_sort_files(distances_dir, data_prefix)
#     return [load_dist_mat(pj(distances_dir, dist_file))
#             for dist_file in files_list]

# def load_intracell_distances_mp(distances_dir, data_prefix=None):
#     """
#     Load distance matrices from directory into list of arrays.

#     Args:
#         distances_dir (string): input directory where distance files are saved
#         data_prefix (string): only read files from distances_dir starting with this string
#         return_mp (boolean): if True, return multiprocessing array, if False return numpy array

#     Returns:
#         list of multiprocessing arrays containing distance matrix for each cell
#     """
#     files_list = list_sort_files(distances_dir, data_prefix)
#     return [load_dist_mat(pj(distances_dir, dist_file), return_mp=return_mp)
#             for dist_file in files_list]

# def _calculate_gw_preload_global(
#         index_pair : Tuple[int,int]
# ) -> Tuple[float, Optional[npt.NDArray[np.float_]]]:
#     """
#     Compute GW distance and the coupling matrix between two distance matrices.
#     Meant to be called within a multiprocessing pool where _DIST_MAT_LIST and RETURN_MAT\
#     exist globally.

#     :param index_pair: indices in the global list _DIST_MAT_LIST for the first\
#          and second distance matrix, respectively
#     :param return_mat: if True, returns the coupling matrix between points
#             if False, only returns GW distance

#         i1 (int): index in the _DIST_MAT_LIST for the first distance matrix
#             i2 (int): index in the _DIST_MAT_LIST for the second distance matrix
#             return_mat (boolean):
#     Returns:
#         float: GW distance
#     """
#     # Get distance matrices from global list (this saves memory so it's not copied in each process)
#     fst, snd = index_pair
#     dist_mat1_unformatted = np.frombuffer(_DIST_MAT_LIST[fst])
#     # mp_arrays are in vector form, we know this is a square matrix
#     numpts = int(np.sqrt(dist_mat1_unformatted.shape[0]))
#     dist_mat1 = dist_mat1_unformatted.reshape((numpts, numpts))
#     dist_mat2 = np.frombuffer(_DIST_MAT_LIST[snd]).reshape((numpts, numpts))
#     # Compute Gromov-Wasserstein matching coupling matrix and distance
#     coupling_mat, log = ot.gromov.gromov_wasserstein(
#         dist_mat1,
#         dist_mat2,
#         ot.unif(dist_mat1.shape[0]),
#         ot.unif(dist_mat2.shape[0]),
#         'square_loss',
#         log=True)
#     if RETURN_MAT:
#         return log['gw_dist'], coupling_mat
#     # RETURN_MAT is false
#     return log['gw_dist'], None

# def _init_fn(dist_mat_list_arg, save_mat):
#     """
#     Initialization function sets _DIST_MAT_LIST and RETURN_MAT to be global in multiprocessing pool.
#     Also sets other arguments because I couldn't figure out how to lazily modify an iterator
#     """
#     global _DIST_MAT_LIST, RETURN_MAT
#     _DIST_MAT_LIST = dist_mat_list_arg
#     RETURN_MAT = save_mat

T = TypeVar('T')
def _batched(itera : Iterator[T], n: int) -> Iterator[List[T]]:
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    while (batch := list(it.islice(itera, n))):
        yield batch

def _is_sorted(int_list : List[int]) -> bool:
    if (len(int_list) <= 1): return True
    return all(map(lambda tup : tup[0] <= tup[1],zip(int_list[:-1],int_list[1:])))

def _convert_document(doc) -> Tuple[int,str, npt.NDArray[np.float64]]:
    return(doc.doc_id,doc['name'],squareform(np.array(doc['cell'],dtype=np.float64)))

def _compute_gw_pool_batched_indices(
        shared_mem_name_1 : str,
        shared_mem_name_2 : str,
        block_length_1 : int,
        block_length_2 : int,
        side_length : int,
        index_list : List[Tuple[int,int]],
        return_coupling_mat : bool
) -> List[Tuple[float, Optional[List[List[float]]]]]:
    time0=time.time()
    
    shared_mem_1 = SharedMemory(name=shared_mem_name_1)
    shared_mem_2 = SharedMemory(name=shared_mem_name_2)
    np_array_1 : npt.NDArray[np.float64] = np.ndarray(
        (block_length_1,side_length,side_length),
        dtype=np.float64,
        buffer=shared_mem_1.buf)
    np_array_2 : npt.NDArray[np.float64] = np.ndarray(
        (block_length_2,side_length,side_length),
        dtype=np.float64,
        buffer=shared_mem_2.buf)
    retlist : List[Tuple[float,Optional[List[List[float]]]]] = []
    for index_1, index_2 in index_list:
        fst_matrix = np_array_1[index_1]
        snd_matrix = np_array_2[index_2]
        # Compute Gromov-Wasserstein matching coupling matrix and distance
        coupling_mat, log = ot.gromov.gromov_wasserstein(
            fst_matrix,
            snd_matrix,
            ot.unif(side_length),
            ot.unif(side_length),
            'square_loss',
            log=True)
        if return_coupling_mat:
            # coo_mat = coo_matrix(coupling_mat)
            retlist.append((log['gw_dist'], coupling_mat.tolist()))
        else:
            retlist.append((log['gw_dist'], None))
    shared_mem_1.close()
    shared_mem_2.close()
    time1=time.time()
    print(time1-time0)
    return(retlist)
    
# def _compute_gw(
#         block_length_1 : int,
#         block_length_2 : int,
#         side_length : int,
#         index_1 : int,
#         index_2 : int,
#         shared_mem_name_1 : str,
#         shared_mem_name_2 : str,
#         return_coupling_mat : bool
# ) -> Tuple[float, Optional[List[List[float]]]]:
#     time0=time.time()

#     shared_mem_1 = SharedMemory(name=shared_mem_name_1)
#     shared_mem_2 = SharedMemory(name=shared_mem_name_2)

#     np_array_1 : npt.NDArray[np.float64] = np.ndarray(
#         (block_length_1,side_length,side_length),
#         dtype=np.float64,
#         buffer=shared_mem_1.buf)
#     np_array_2 : npt.NDArray[np.float64] = np.ndarray(
#         (block_length_2,side_length,side_length),
#         dtype=np.float64,
#         buffer=shared_mem_2.buf)
#     fst_matrix = np_array_1[index_1]
#     snd_matrix = np_array_2[index_2]
#     # Compute Gromov-Wasserstein matching coupling matrix and distance
#     coupling_mat, log = ot.gromov.gromov_wasserstein(
#         fst_matrix,
#         snd_matrix,
#         ot.unif(side_length),
#         ot.unif(side_length),
#         'square_loss',
#         log=True)
#     shared_mem_1.close()
#     shared_mem_2.close()
#     time1=time.time()
#     if return_coupling_mat:
#         coo_mat = coo_matrix(coupling_mat)
#         return log['gw_dist'], { "data" : coo_mat.data.tolist(),
#                                  "row" : coo_mat.row.tolist(),
#                                  "col" : coo_mat.col.tolist() }
#     return log['gw_dist'], None

GW_Record_W_CouplingMat = TypedDict('GW_Record_W_CouplingMat',
                                    { "name_1" : str,
                                      "name_2" : str,
                                      "coupling_mat" : List[List[float]],
                                      "gw_dist" : float
                                     })

GW_Record_WO_CouplingMat = TypedDict('GW_Record_WO_CouplingMat',
                                    { "name_1" : str,
                                      "name_2" : str,
                                      "gw_dist" : float})

def _batched_cell_list_iterator(
        intracell_db_loc : str,
        chunk_size : int
) -> Iterator[Tuple[
        List[Tuple[int,str, npt.NDArray[np.float64]]],
        List[Tuple[int,str, npt.NDArray[np.float64]]]]]:
    """
    :param intracell_db_loc: A full file path to a TinyDB json database.
    :param chunk_size: A size parameter.

    :return: An iterator over pairs (list1, list2), where each element in list1 and list2 is a triple
    (cell_id, cell_name, icdm), where cell_id is the unique identifier of a cell in the tinydb database,
    cell_name is a string, and icdm is a square n x n distance matrix.
    """

    # intracell_db is an existing \*.json file
    intracell_db = TinyDB(intracell_db_loc)
    # We assume that the table is called "_default", which is the TinyDB default name for a database.
    intracell_table = intracell_db.table('_default', cache_size=2 * chunk_size)
    cell_id_list : List[int] = []
    for cell in iter(intracell_table):
        cell_id_list.append(cell.doc_id)
    assert(_is_sorted(cell_id_list))

    # Construct an iterator over all cells in the table.
    # Batch the iterator into blocks of size chunk_size.
    outer_doc_iter = iter(intracell_table)
    batched_outer = _batched(outer_doc_iter, chunk_size)
    for outer_batch in batched_outer:
        # Convert cells to a simple form (key, name, intracell_distance_matrix of shape (n,n))
        outer_batch_tuples = list(map(_convert_document,outer_batch))
        side_length : int = outer_batch_tuples[0][2].shape[0]
        inner_doc_iter = iter(intracell_table)
        # This loop discards the initial segment of the iterator containing all
        # those cells whose key is less than the first cell in outer_batch. It
        # also discards the first cell for which this test fails, which should
        # be outer_batch[0] itself if the iterator returns the cells in increasing
        # order of their keys.
        while(next(inner_doc_iter).doc_id < outer_batch[0].doc_id):
            pass
        batched_inner = _batched(inner_doc_iter, chunk_size)
        for inner_batch in batched_inner:
            inner_batch_tuples = list(map(_convert_document,inner_batch))
            yield outer_batch_tuples, inner_batch_tuples

def _batched_cell_list_iterator_csv(
        intracell_csv_loc : str,
        chunk_size : int
) -> Iterator[Tuple[
        List[Tuple[int,str, npt.NDArray[np.float64]]],
        List[Tuple[int,str, npt.NDArray[np.float64]]]]]:
    """
    :param intracell_csv_loc: A full file path to a csv file.
    :param chunk_size: A size parameter.

    :return: An iterator over pairs (list1, list2), where each element in list1 and list2 is a triple
    (cell_id, cell_name, icdm), where cell_id is a natural number,
    cell_name is a string, and icdm is a square n x n distance matrix.
    cell_id is guaranteed to be unique.
    """

    with open(intracell_csv_loc, newline='') as icdm_csvfile_outer:
        csv_outer_reader = enumerate(csv.reader(icdm_csvfile_outer,delimiter=','))
        _, header = next(csv_outer_reader)
        assert header[0]=="cell_id"
        line_length = len(header[1:])
        side_length = ceil(sqrt(2*line_length))
        assert(side_length * (side_length -1)== 2*line_length)
        batched_outer = _batched(csv_outer_reader, chunk_size)
        for outer_batch in batched_outer:
            outer_list =\
                [(cell_id, ell[0],
                  squareform(np.array(
                      [ float(x) for x in ell[1:]],dtype=np.float64)))
                  for (cell_id, ell) in outer_batch]
            first_outer_id = outer_list[0][0]
            with open(intracell_csv_loc, newline='') as icdm_csvfile_inner:
                csv_inner_reader = enumerate(csv.reader(icdm_csvfile_inner,delimiter=','))
                while (next(csv_inner_reader)[0] < first_outer_id):
                    pass
                batched_inner = _batched(csv_inner_reader, chunk_size)
                for inner_batch in batched_inner:
                    inner_list =\
                        [(cell_id, ell[0], squareform(np.array(
                            [float(x) for x in ell[1:]],dtype=np.float64)))
                          for (cell_id, ell) in inner_batch]
                    yield outer_list, inner_list

def compute_gw_distance_matrix(
        intracell_csv_loc : str,
        gw_csv_loc : str,
        save_mat : bool =False
) -> None:
    with open(gw_csv_loc, 'a', newline='') as gw_csv_file:
        csvwriter = csv.writer(gw_csv_file, delimiter=',')
        chunk_size=100
        with open(intracell_csv_loc, 'r', newline='') as icdm_csv:
            header=next(csv.reader(icdm_csv, delimiter=','))
            assert header[0]=='cell_id'
            list_length = len(header[1:])
        side_length : int = ceil(sqrt(2 * list_length))
        assert (side_length * (side_length - 1) == list_length * 2)
        if save_mat:
            header = ["first_cell","second_cell","gw_dist"] + [*range(side_length ** 2)]
        else:
            header = ["first_cell","second_cell","gw_dist"]
        csvwriter.writerow(header)
        time_start=time.time()
        cell_comparisons=0
        for cell_list_1, cell_list_2 in _batched_cell_list_iterator_csv(intracell_csv_loc, chunk_size):
            name_mat_pairs =\
                _batched(((t[1], t[2], s[1], s[2])
                          for (t, s) in it.product(cell_list_1, cell_list_2)
                          if t[0]<s[0]),
                         chunk_size)
            for name_mat_pair_batch in name_mat_pairs:
                rowlist : list[ list[str | float] ] = []
                for name1, icdm_1, name2, icdm_2 in name_mat_pair_batch:
                    coupling_mat, log = ot.gromov.gromov_wasserstein(
                        icdm_1,
                        icdm_2,
                        ot.unif(side_length),
                        ot.unif(side_length),
                        'square_loss',
                        log=True)
                    if save_mat:
                        flatlist = [x for l in coupling_mat.tolist() for x in l]
                        rowlist.append([name1,name2,log['gw_dist']]+flatlist)
                    else:
                        rowlist.append([name1,name2,log['gw_dist']])
                csvwriter.writerows(rowlist)
                cell_comparisons+=len(name_mat_pair_batch)
                print("Total time elapsed:" + str(time.time()-time_start))
                print("Cell comparisons so far:" + str(cell_comparisons))

# def compute_gw_distance_matrix_json(
#         intracell_db_loc : str,
#         gw_csv : str,
#         save_mat : bool =False
# ) -> None:
#     with open(gw_csv, 'a', newline='') as csvfile:
#         csvwriter = csv.writer(csvfile, delimiter=',')
#         chunk_size=100
#         list_length : int = len(next(iter(TinyDB(intracell_db_loc)))['cell'])
#         side_length : int = ceil(sqrt(2 * list_length))
#         assert (side_length * (side_length - 1) == list_length * 2)
#         if save_mat:
#             header = ["first_cell","second_cell","gw_dist"] + list(range(side_length ** 2))
#         else:
#             header = ["first_cell","second_cell","gw_dist"]
#         csvwriter.writerow(header)
#         time_start=time.time()
#         cell_comparisons=0
#         for cell_list_1, cell_list_2 in _batched_cell_list_iterator(intracell_db_loc, chunk_size):
#             name_mat_pairs =\
#                 _batched(((t[1], t[2], s[1], s[2])
#                           for (t, s) in it.product(cell_list_1, cell_list_2)
#                           if t[0]<s[0]),
#                          chunk_size)
#             for name_mat_pair_batch in name_mat_pairs:
#                 rowlist : list[ list[str | float] ] = []
#                 for name1, icdm_1, name2, icdm_2 in name_mat_pair_batch:
#                     coupling_mat, log = ot.gromov.gromov_wasserstein(
#                         icdm_1,
#                         icdm_2,
#                         ot.unif(side_length),
#                         ot.unif(side_length),
#                         'square_loss',
#                         log=True)
#                     if save_mat:
#                         flatlist = [x for l in coupling_mat.tolist() for x in l]
#                         rowlist.append([name1,name2,log['gw_dist']]+flatlist)
#                     else:
#                         rowlist.append([name1,name2,log['gw_dist']])
#                 csvwriter.writerows(rowlist)
#                 cell_comparisons+=len(name_mat_pair_batch)
#                 print("Total time elapsed:" + str(time.time()-time_start))
#                 print("Cell comparisons so far:" + str(cell_comparisons))
                
# def compute_gw_distance_matrix_one_process(
#         intracell_db_loc : str,
#         gw_db_loc : str,
#         save_mat : bool =False
#         chunk_size : int = 1000
#         )-> None:
#     r"""
#     Compute the GW distance between every pair of cells in the database intracell_db_loc.

#     :param intracell_db_loc: A \*.json database of intracell distance matrices, structured \
#         according to the format used by TinyDB. \
#         The database should have a single top-level name-value pair, where the name is "_default" \
#         and the value is the table of the database. The table should be a dictionary whose \
#         keys are strings representing integers (in ascending order through the file) and \
#         whose entries are TinyDB documents. A document is a dictionary with two entries, "name" \
#         and "cell". "name" is a string, and "cell" is a list of n \* (n - 1) / 2 floating point \
#         real numbers, where the distance between objects :math:`x_i` and :math:`x_j` \
#         (for :math:`i<j` ) occurs in the list `cell` at index \
#         :math:`{n \choose 2} - {n - i \choose 2} + (j - i - 1)` (see \
#         :func:`scipy.spatial.distance.pdist` and footnote 2 of \
#         :func:`scipy.spatial.distance.squareform`), corresponding to the \
#         entries lying above the diagonal of the \
#         intracell distance matrix.
#     :param gw_db_loc: The path to a (not already existing) \*.json file where the computation \
#         results will be written. Documents in the output database will contain fields "name_1" \
#         (the name of the first cell), "name_2" (the name of the second cell), "gw_dist" \
#         (the Gromov-Wasserstein distance between the two cells as a floating point number), \
#         and (optionally) "coupling_mat", the coupling matrix recording the best fit between the \
#         two cells.
#     :param save_mat: If save_mat is true, for each pair of input cells, the output database \
#         will contain not just the GW distance between the two cells \
#         (a single floating-point real number) but an additional field "coupling_mat" \
#         giving the best-fit coupling matrix relating the two matrices. The user is warned that \
#         for two cells represented by 50 sample points, the coupling matrix will be ~28kb, \
#         and that the number of coupling matrices grows with n * (n-1)/2, where n is the number \
#         of cells to compare. On a test run of 150 cells this yields a 170MB output file.
#     :param chunk_size: Controls the size of cell batches passed to subprocesses, adjust \
#         as appropriate for desired memory usage.
    
#     """
#     # intracell_db is an existing \*.json file
#     intracell_db = TinyDB(intracell_db_loc)
#     # We assume that the table is called "_default", which is the TinyDB default name for a database.
#     intracell_table = intracell_db.table('_default', cache_size=2 * chunk_size)
#     # The output matrices will be written to gw_db.
#     gw_db = TinyDB(gw_db_loc)
#     # We will assume the id's in the database are in sorted order in order to
#     # have a reasonably efficient algorithm design and also not have to come up
#     # with anything too clever. The following loop validates this.
#     cell_id_list : List[int] = []
#     for cell in iter(intracell_table):
#         cell_id_list.append(cell.doc_id)
#     assert(_is_sorted(cell_id_list))
#     assert chunk_size > 0
#     # Main outer loop:
#     # Construct an iterator over all cells in the table.
#     # Batch the iterator into blocks of size chunk_size.
#     outer_doc_iter = iter(intracell_table)
#     batched_outer = _batched(outer_doc_iter, chunk_size)
#     for outer_batch in batched_outer:
#         # Convert cells to a simple form (key, name, intracell_distance_matrix of shape (n,n))
#         outer_batch_tuples = list(map(_convert_document,outer_batch))
#         side_length : int = outer_batch_tuples[0][2].shape[0]
#         inner_doc_iter = iter(intracell_table)
#         batched_inner = _batched(inner_doc_iter, chunk_size)
#         # This loop discards the initial segment of the iterator containing all
#         # those cells whose key is less than the first cell in outer_batch. It
#         # also discards the first cell for which this test fails, which should
#         # be outer_batch[0] itself if the iterator returns the cells in increasing
#         # order of their keys.
#         while(next(inner_doc_iter).doc_id < outer_batch[0].doc_id):
#             pass
#         for inner_batch in batched_inner:
#             inner_batch_tuples = list(map(_convert_document,inner_batch))
#             filter_fun : Callable[[Tuple[int,int]],bool]
#             # If i, j are array indices, filter_fun (i, j) is true iff the
#             # key of the cell at outer_batch_tuples[i] is lower than the one at inner_batch_tuples[j].
#             filter_fun = lambda tup : (outer_batch_tuples[tup[0]][0] < inner_batch_tuples[tup[1]][0])
#             array_index_pairs : Iterator[Tuple[int,int]]
#             array_index_pairs = filter(
#                     filter_fun,
#                     it.product(range(len(outer_batch_tuples)),range(len(inner_batch_tuples))))
#             compute_gw_pool_local : Callable [[List[Tuple[int,int]]],
#                                                 Tuple[List[Tuple[int,int]],
#                                                       List[Tuple[float,
#                                                                  Optional[List[List[float]]]]]]]
#             compute_gw_pool_local =\
#                 lambda index : (index,_compute_gw(
#                     side_length,
#                     index,
#                     save_mat))
#             out = map(
#                 compute_gw_pool_local,
#                 array_index_pairs)

#             counter = 0
#             time0 = time.time()
#             for index, batch_output in out:
#                 time1 = time.time()
#                 print("Total time of this iteration:" + str(1000*(time1 - time0)))
#                 print("Currently processing batch " + str(counter))
#                 time0 = time.time()
#                 if save_mat:
#                     insert_dict_list_w_coupling_mat : List[GW_Record_W_CouplingMat]
#                     insert_dict_list_w_coupling_mat =\
#                         [ { "name_1" : outer_batch_tuples[t[0][0]][1],
#                             "name_2" : inner_batch_tuples[t[0][1]][1],
#                             "coupling_mat" : t[1][1],
#                             "gw_dist" : t[1][0] }
#                           for t in zip(index, batch_output) ]
#                     gw_db.insert_multiple(insert_dict_list_w_coupling_mat)
#                     time2 = time.time()
#                     print("Pairs in batch: " + str(len(insert_dict_list_w_coupling_mat)))
#                     print("Time spent writing to file: " + str(1000*(time2 - time0)))
#                 else:           # save_mat is false
#                     insert_dict_list_wo_coupling_mat : List[GW_Record_WO_CouplingMat]
#                     insert_dict_list_wo_coupling_mat =\
#                         [ { "name_1" : outer_batch_tuples[t[0][0]][1],
#                             "name_2" : inner_batch_tuples[t[0][1]][1],
#                              "gw_dist" : t[1][0] }
#                           for t in zip(index,batch_output) ]
#                     gw_db.insert_multiple(insert_dict_list_wo_coupling_mat)
#                     time2 = time.time()
#                     print("Pairs in batch: " + str(len(insert_dict_list_wo_coupling_mat)))
#                     print("Time spent writing to file: " +str(1000*(time2-time0)))
#                 counter += 1

# This works but does not appear to offer any real performance boosts.
def compute_gw_distance_matrix_parallel(
        intracell_db_loc : str,
        gw_db_loc : str,
        save_mat : bool =False,
        num_cores : int=8,
        chunk_size : int = 50
        )-> None:
    r"""
    Compute the GW distance between every pair of cells in the database intracell_db_loc.

    :param intracell_db_loc: A \*.json database of intracell distance matrices, structured \
        according to the format used by TinyDB. \
        The database should have a single top-level name-value pair, where the name is "_default" \
        and the value is the table of the database. The table should be a dictionary whose \
        keys are strings representing integers (in ascending order through the file) and \
        whose entries are TinyDB documents. A document is a dictionary with two entries, "name" \
        and "cell". "name" is a string, and "cell" is a list of n \* (n - 1) / 2 floating point \
        real numbers, where the distance between objects :math:`x_i` and :math:`x_j` \
        (for :math:`i<j` ) occurs in the list `cell` at index \
        :math:`{n \choose 2} - {n - i \choose 2} + (j - i - 1)` (see \
        :func:`scipy.spatial.distance.pdist` and footnote 2 of \
        :func:`scipy.spatial.distance.squareform`), corresponding to the \
        entries lying above the diagonal of the \
        intracell distance matrix.
    :param gw_db_loc: The path to a (not already existing) \*.json file where the computation \
        results will be written. Documents in the output database will contain fields "name_1" \
        (the name of the first cell), "name_2" (the name of the second cell), "gw_dist" \
        (the Gromov-Wasserstein distance between the two cells as a floating point number), \
        and (optionally) "coupling_mat", the coupling matrix recording the best fit between the \
        two cells.
    :param save_mat: If save_mat is true, for each pair of input cells, the output database \
        will contain not just the GW distance between the two cells \
        (a single floating-point real number) but an additional field "coupling_mat" \
        giving the best-fit coupling matrix relating the two matrices. The user is warned that \
        for two cells represented by 50 sample points, the coupling matrix will be ~28kb, \
        and that the number of coupling matrices grows with n * (n-1)/2, where n is the number \
        of cells to compare. On a test run of 150 cells this yields a 170MB output file.
    :param num_cores: The number of independent parallel processes that will be launched. \
        Recommended to set equal to the number of cores on your machine. \
    :param chunk_size: Controls the size of cell batches passed to subprocesses, adjust \
        as appropriate for desired memory usage.
    
    """
    # intracell_db is an existing \*.json file
    intracell_db = TinyDB(intracell_db_loc)
    # We assume that the table is called "_default", which is the TinyDB default name for a database.
    intracell_table = intracell_db.table('_default', cache_size=2 * chunk_size)
    # The output matrices will be written to gw_db.
    gw_db = TinyDB(gw_db_loc)
    # We will assume the id's in the database are in sorted order in order to
    # have a reasonably efficient algorithm design and also not have to come up
    # with anything too clever. The following loop validates this.
    cell_id_list : List[int] = []
    for cell in iter(intracell_table):
        cell_id_list.append(cell.doc_id)
    assert(_is_sorted(cell_id_list))
    assert chunk_size > 0
    set_start_method("spawn", force=True)
    pool = ProcessPool(nodes=num_cores)
    pool.restart()

    # Main outer loop:
    # Construct an iterator over all cells in the table.
    # Batch the iterator into blocks of size chunk_size.
    outer_doc_iter = iter(intracell_table)
    batched_outer = _batched(outer_doc_iter, chunk_size)
    time_start= time.time()
    total_num_pairs =0 
    for outer_batch in batched_outer:
        # outer_batch is a list of chunk_size many documents.
        # Convert cells to a simple form: (key, name, intracell_distance_matrix of shape (n,n))
        outer_batch_tuples = list(map(_convert_document,outer_batch))
        
        assert(len(outer_batch_tuples) <= 1000)
        side_length : int = outer_batch_tuples[0][2].shape[0]
        assert(side_length == 100)
        outer_local_array = np.empty(
            shape=(len(outer_batch_tuples),side_length, side_length),
            dtype=np.float64)
        shm_icdm_1 = SharedMemory(create=True,size=outer_local_array.nbytes)
        outer_shared_dists : npt.NDArray[np.float64]=np.ndarray(
            outer_local_array.shape,dtype=np.float64, buffer=shm_icdm_1.buf)

        for i in range(len(outer_batch_tuples)):
            outer_shared_dists[i][:]=outer_batch_tuples[i][2][:]
        for i in range(len(outer_batch_tuples)):
            assert(outer_shared_dists[i].shape == (100,100))
            assert(outer_batch_tuples[i][2].shape == (100,100))
            assert(np.array_equal(outer_shared_dists[i],outer_batch_tuples[i][2]))

        inner_doc_iter = iter(intracell_table)
        batched_inner = _batched(inner_doc_iter, chunk_size)
        # This loop discards the initial segment of the iterator containing all
        # those cells whose key is less than the first cell in outer_batch. It
        # also discards the first cell for which this test fails, which should
        # be outer_batch[0] itself if the iterator returns the cells in increasing
        # order of their keys.
        while(next(inner_doc_iter).doc_id < outer_batch[0].doc_id):
            pass
        for inner_batch in batched_inner:
            inner_batch_tuples = list(map(_convert_document,inner_batch))
            inner_local_array = np.empty(
                shape=(len(inner_batch_tuples), side_length, side_length),
                dtype=np.float64)
            shm_icdm_2 = SharedMemory(create=True,size=inner_local_array.nbytes)
            inner_shared_dists : npt.NDArray[np.float64] =np.ndarray(
                inner_local_array.shape,dtype=np.float64, buffer=shm_icdm_2.buf)
            for j in range(len(inner_batch_tuples)):
                inner_shared_dists[j][:]=inner_batch_tuples[j][2][:]
            # filter_fun : Callable[[Tuple[int,int]],bool]
            # If i, j are array indices, filter_fun (i, j) is true iff the
            # key of the cell at outer_batch_tuples[i] is lower than the one at inner_batch_tuples[j].
            # filter_fun = lambda tup : (outer_batch_tuples[tup[0]][0] < inner_batch_tuples[tup[1]][0])
            all_index_pairs = it.product(
                range(len(outer_batch_tuples)),
                range(len(inner_batch_tuples)))
            array_index_pairs : Iterator[Tuple[int,int]]            
            array_index_pairs = ( (i, j) for i, j in all_index_pairs if
                                  outer_batch_tuples[i][0] < inner_batch_tuples[j][0])
            # if save_mat:
            #     array_index_pairs_batched = _batched(array_index_pairs,int(num_cores*chunk_size/10))
            # else:
            #     array_index_pairs_batched = _batched(array_index_pairs,num_cores*chunk_size)

            # array_index_pairs = filter(
            #         filter_fun,
            #     it.product(range(len(outer_batch_tuples)),range(len(inner_batch_tuples))))
            if save_mat:
                array_index_pairs_batched = _batched(array_index_pairs,int(num_cores*chunk_size/10))
            else:
                array_index_pairs_batched = _batched(array_index_pairs,num_cores*chunk_size)
            compute_gw_pool_local : Callable [[List[Tuple[int,int]]],
                                                Tuple[List[Tuple[int,int]],
                                                      List[Tuple[float,
                                                                 Optional[List[List[float]]]]]]]
            compute_gw_pool_local =\
                lambda index_list : (index_list,_compute_gw_pool_batched_indices(
                    shm_icdm_1.name,
                    shm_icdm_2.name,
                    len(outer_batch_tuples),
                    len(inner_batch_tuples),
                    side_length,
                    index_list,
                    save_mat))
            out = pool.uimap(
                compute_gw_pool_local,
                array_index_pairs_batched,chunksize=1)
            for index_list, batch_output in out:
                assert(len(index_list)==len(batch_output))
                total_num_pairs += len(index_list)
                time_now = time.time()
                print("Total pairs: " + str(total_num_pairs))
                print("Total time elapsed: "+ str(time_now-time_start))
                if save_mat:
                    insert_dict_list_w_coupling_mat : List[GW_Record_W_CouplingMat]
                    insert_dict_list_w_coupling_mat =\
                        [ { "name_1" : outer_batch_tuples[t[0][0]][1],
                            "name_2" : inner_batch_tuples[t[0][1]][1],
                            "coupling_mat" : t[1][1],
                            "gw_dist" : t[1][0] }
                          for t in zip(index_list, batch_output) ]
                    gw_db.insert_multiple(insert_dict_list_w_coupling_mat)
                else:           # save_mat is false
                    insert_dict_list_wo_coupling_mat : List[GW_Record_WO_CouplingMat]
                    insert_dict_list_wo_coupling_mat =\
                        [ { "name_1" : outer_batch_tuples[t[0][0]][1],
                            "name_2" : inner_batch_tuples[t[0][1]][1],
                             "gw_dist" : t[1][0] }
                          for t in zip(index_list,batch_output) ]
                    gw_db.insert_multiple(insert_dict_list_wo_coupling_mat)
            shm_icdm_2.close()
            shm_icdm_2.unlink()
        shm_icdm_1.close()
        shm_icdm_1.unlink()
    pool.close()
    pool.join()

# def compute_gw_distance_matrix_draft(
#         intracell_db_loc : str,
#         gw_db_loc : str,
#         save_mat : bool =False,
#         num_cores : int=8,
#         chunk_size : int = 1000
#         )-> None:
#     r"""
#     Compute the GW distance between every pair of cells in the database intracell_db_loc.
#     :param intracell_db_loc: A \*.json file which codes a document in the format associated with\
#     TinyDB. The \*.json file is assumed to have one table, called "_default". Each document in \
#     the table has two keys, called "name" and "cell", where name is a string and cell is a list of\
#     floats of length n * (n-1)/2 (the entries lying above the diagonal of the \
#     intracell distance matrix).
#     """
#     # intracell_db is an existing \*.json file
#     intracell_db = TinyDB(intracell_db_loc)
#     # We assume that the table is called "_default", which is the TinyDB default name for a database.
#     intracell_table = intracell_db.table('_default', cache_size=2 * chunk_size)
#     # The output matrices will be written to gw_db.
#     gw_db = TinyDB(gw_db_loc)
#     # We will assume the id's in the database are in sorted order in order to
#     # have a reasonably efficient algorithm design and also not have to come up
#     # with anything too clever. The following loop validates this.
#     cell_id_list : List[int] = []
#     for cell in iter(intracell_table):
#         cell_id_list.append(cell.doc_id)
#     assert(_is_sorted(cell_id_list))
#     assert chunk_size > 0
#     set_start_method("spawn")
#     pool = ProcessPool(nodes=1)

#     # Main outer loop:
#     # Construct an iterator over all cells in the table.
#     # Batch the iterator into blocks of size chunk_size.
#     outer_doc_iter = iter(intracell_table)
#     batched_outer = _batched(outer_doc_iter, chunk_size)
#     for outer_batch in batched_outer:
#         # Convert cells to a simple form (key, name, intracell_distance_matrix of shape (n,n))
#         outer_batch_tuples = list(map(_convert_document,outer_batch))
#         side_length : int = outer_batch_tuples[0][2].shape[0]
#         outer_local_array = np.empty(
#             shape=(len(outer_batch_tuples),side_length, side_length),
#             dtype=np.float64)
#         shm_icdm_1 = SharedMemory(create=True,size=outer_local_array.nbytes)
#         outer_shared_dists : npt.NDArray[np.float64]=np.ndarray(
#             outer_local_array.shape,dtype=np.float64, buffer=shm_icdm_1.buf)
#         for i in range(len(outer_batch_tuples)):
#             outer_shared_dists[i][:]=outer_batch_tuples[i][2][:]
#         inner_doc_iter = iter(intracell_table)
#         batched_inner = _batched(inner_doc_iter, chunk_size)
#         # This loop discards the initial segment of the iterator containing all
#         # those cells whose key is less than the first cell in outer_batch. It
#         # also discards the first cell for which this test fails, which should
#         # be outer_batch[0] itself if the iterator returns the cells in increasing
#         # order of their keys.
#         while(next(inner_doc_iter).doc_id < outer_batch[0].doc_id):
#             pass
#         for inner_batch in batched_inner:
#             inner_batch_tuples = list(map(_convert_document,inner_batch))
#             print(len(inner_batch_tuples))
#             inner_local_array = np.empty(
#                 shape=(len(inner_batch_tuples), side_length, side_length),
#                 dtype=np.float64)
#             shm_icdm_2 = SharedMemory(create=True,size=inner_local_array.nbytes)
#             inner_shared_dists : npt.NDArray[np.float64] =np.ndarray(
#                 inner_local_array.shape,dtype=np.float64, buffer=shm_icdm_2.buf)
#             filter_fun : Callable[[Tuple[int,int]],bool]
#             # If i, j are array indices, filter_fun (i, j) is true iff the
#             # key of the cell at outer_batch_tuples[i] is lower than the one at inner_batch_tuples[j].
#             filter_fun = lambda tup : (outer_batch_tuples[tup[0]][0] < inner_batch_tuples[tup[1]][0])
#             array_index_pairs : Iterator[Tuple[int,int]]
#             array_index_pairs = filter(
#                     filter_fun,
#                     it.product(range(len(outer_batch_tuples)),range(len(inner_batch_tuples))))
#             # array_index_pairs_batched = _batched(array_index_pairs,100)
#             compute_gw_pool_local : Callable [[Tuple[int,int]],
#                                               Tuple[Tuple[int,int],
#                                                     Tuple[float,Optional[List[List[float]]]]]]
#             compute_gw_pool_local =\
#                 lambda tup : (tup,_compute_gw_pool(
#                     shm_icdm_1.name,
#                     shm_icdm_2.name,
#                     len(outer_batch_tuples),
#                     len(inner_batch_tuples),
#                     side_length,
#                     tup[0],
#                     tup[1],
#                     save_mat))
#             out = pool.uimap(
#                 compute_gw_pool_local,
#                 array_index_pairs,chunksize=1000)
#             test_list = []
#             n = 1
#             start = time.time()
#             for a in out:
#                 stop = time.time()
#                 test_list.append(stop-start)
#                 n+=1
#                 start = time.time()
#                 if n > 1000:
#                     pool.terminate()
#                     pool.join()
#                     shm_icdm_2.close()
#                     shm_icdm_1.close()
#                     shm_icdm_2.unlink()
#                     shm_icdm_1.unlink()
#                     return test_list
            
                
                
                      
#             batched_output = _batched(out,100)
#             counter = 0
#             time2=time.time()
#             for batch_output in batched_output:
#                 print("Currently processing batch " + str(counter))
#                 if save_mat:
#                     time0 = time.time()
#                     print("Time to compute batch_output =" +str(1000*(time0-time2)))
#                     insert_dict_list_w_coupling_mat : List[GW_Record_W_CouplingMat]
#                     insert_dict_list_w_coupling_mat =\
#                         [ { "name_1" : outer_batch_tuples[index_gw_dist_pair[0][0]][1],
#                             "name_2" : inner_batch_tuples[index_gw_dist_pair[0][1]][1],
#                             "coupling_mat" : index_gw_dist_pair[1][1],
#                             "gw_dist" :  index_gw_dist_pair[1][0] }
#                           for index_gw_dist_pair in batch_output]
#                     time1 = time.time()
#                     print("Time to form insertion dictionary=" +str(1000*(time1-time0)))
#                     gw_db.insert_multiple(insert_dict_list_w_coupling_mat)
#                     time2 = time.time()
#                     print("Time to write dictionary to db=" +str(1000*(time2-time1)))
#                 else:           # save_mat is false
#                     time0 = time.time()
#                     print("Time to compute batch_output =" +str(1000*(time0-time2)))
#                     insert_dict_list_wo_coupling_mat : List[GW_Record_WO_CouplingMat]
#                     insert_dict_list_wo_coupling_mat =\
#                         [ { "name_1" : outer_batch_tuples[index_gw_dist_pair[0][0]][1],
#                             "name_2" : inner_batch_tuples[index_gw_dist_pair[0][1]][1],
#                             "gw_dist" :  index_gw_dist_pair[1][0] }
#                           for index_gw_dist_pair in batch_output]
#                     time1 = time.time()
#                     print("Time to form insertion dictionary=" +str(1000*(time1-time0)))
#                     gw_db.insert_multiple(insert_dict_list_wo_coupling_mat)
#                     time2 = time.time()
#                     print("Time to write dictionary to db=" +str(1000*(time2-time1)))
#                 counter += 1
#             shm_icdm_2.close()
#             shm_icdm_2.unlink()
            
            
#         shm_icdm_1.close()
#         shm_icdm_1.unlink()
        
#     pool.close()
#     pool.join()

        # # Group them into batches of size chunk_size.
    # batched_ids = _batched(cell_id_list,chunk_size)
    # batch_pairs = it.combinations_with_replacement(batched_ids,2)

    # pool = ProcessPool(nodes=num_cores)

    # # The outer loop iterates over pairs of tuples of cell_ids of length (at most) chunk_size.
    # for cell_id_block_1, cell_id_block_2 in batch_pairs:
    #     doc_list_1 : Dict[int, dict] = {}
    #     doc_list_2 : Dict[int, dict] = {}
    #     for item in intracell_table:
    #         if item.doc_id >= cell_id_block_1[0] and item.doc_id <= cell_id_block_1[-1]:
    #             doc_list_1[item.doc_id] = item
    #         if item.doc_id >= cell_id_block_2[0] and item.doc_id <= cell_id_block_2[-1]:
    #             doc_list_2[item.doc_id] = item
    #         item['cell'] = squareform(np.array(item['cell'],dtype=np.float64))
    #         if item.doc_id > max(fst[-1],snd[-1]):
    #             break
        
    #     # Shared memory space for the two lists of intracell distance matrices.
    #     side_length : int = doc_list_1[doc_list_1.keys()[0]]['cell'].shape[0]

    #     indexed_keys_1 : List[int] = doc_list_1.keys().sort()
    #     num_cells_1 = len(doc_list_1)
    #     local_array_1 = np.ones(shape=(len(doc_list_1),side_length, side_length),dtype=np.float64)
    #     shm_icdm_1 = shared_memory.SharedMemory(create=True,size=local_array_1.nbytes)
    #     shared_dists_1=np.ndarray(local_array_1.shape,dtype=np.float64, buffer=shm_icdm_1.buf)

    #     indexed_keys_2 : List[int] = doc_list_2.keys().sort()
    #     num_cells_2 = len(doc_list_2)
    #     local_array_2 = np.ones(shape=(len(doc_list_2),side_length, side_length),dtype=np.float64)
    #     shm_icdm_2 = shared_memory.SharedMemory(create=True,size=local_array_2.nbytes)
    #     shared_dists_2=np.ndarray(local_array_2.shape,dtype=np.float64, buffer=shm_icdm_2.buf)
        
    #     for i in range(len(indexed_keys_1)):
    #         shared_dists_1[i][:]=doc_list_1[indexed_keys_1[i]]['cell']
    #     for j in range(len(indexed_keys_2)):
    #         shared_dists_2[j][:]=doc_list_2[indexed_keys_2[j]]['cell']
        # index_pairs = filter(lambda tup : tup[0]<tup[1],it.product(fst,snd))
        # dist_mats : Iterator[Tuple[str,str,float,Optional[npt.NDArray[np.float_]]]]
        # dist_mats = pool.imap(_calculate_gw_new, index_pairs, doc_list_1, doc_list_2)
        
        # indexed_pairs_batched = _batched(index_pairs, chunk_size)

# def compute_gw_distance_matrix(
#         dist_mat_list_arg : List[npt.NDArray[np.float_]],
#         save_mat : bool =False,
#         num_cores : int=8,
#         chunk_size : int =100
# )-> Tuple[npt.NDArray[np.float_], Optional[List[npt.NDArray[np.float_]]]]:
#     """
    
#     Compute the GW distance between each pair of matrices in a given list of intracell \
#     distance matrices

#     :param dist_mat_list_arg: list of numpy arrays of shape (n,) containing distance\
#               matrix for each cell, in "vector form"
#     :param save_mat: if True, returns coupling matrix (matching) between points. \
#                             if False, only returns GW distance
#     :param num_cores: number of parallel processes to run GW in
#     :param chunk_size: chunk size for the iterator of all pairs of cells. \
#             Larger size is faster but takes more memory, see \
#             :meth:`multiprocessing.pool.Pool.imap` for details.

#     :return:
#         A matrix of the GW distances between all the intracell distance matrices in \
#         dist_mat_list_arg.
#     """

#     num_mats = len(dist_mat_list_arg)
#     if num_mats == 0:
#         return (np.empty((0,)), None)

#     assert len(dist_mat_list_arg[0].shape)==1

#     dist_mat_list_squareform = [squareform(matrix) for matrix in dist_mat_list_arg]

#     arguments = it.combinations(range(num_mats), 2)


#     # if num_cores > 1:
#         # Start up multiprocessing w/ list of distance matrices in global environment
#     with Pool(processes=num_cores,
#               initializer=_init_fn,
#               initargs=(dist_mat_list_squareform, save_mat)) as pool:
#         dist_results = list(pool.map(
#             _calculate_gw_preload_global,
#             arguments,
#             chunksize=chunk_size))

#     results_stream = zip(*dist_results)
#     gw_dist_mat = list(next(results_stream))
#     if save_mat:
#         gw_coupling_mats = list(next(results_stream))
#     else:
#         gw_coupling_mats = None
#     return (gw_dist_mat, gw_coupling_mats)

    # else:
    #     # Set dist_mat_list in global environment so can call the same functions
    #     global dist_mat_list, return_mat
    #     dist_mat_list = dist_mat_list_arg
    #     return_mat = save_mat
    #     dist_results = list(map(_calculate_gw_preload_global, arguments))
# def compute_and_save_GW_dist_mat(
#         dist_mat_list_local : List[npt.NDArray| ctypes.Array],
#         file_prefix : str,
#         gw_results_dir : str,
#         save_mat:bool =False,
#         num_cores: int =12,
#         chunk_size: int =100):
#     """
#     Compute the GW distance between each pair of distance matrices in vector form,
#     and write the resulting matrix of GW distances to a file.

#     Args:
#         * dist_mat_list_local (list): list of multiprocessing or numpy arrays containing\
#               intracell distance matrix for each cell
#         * file_prefix (string): name of output file to write GW distance matrix to
#         * gw_results_dir (string): path to directory to write output file to
#         * save_mat (boolean): if True, returns coupling matrix (matching) between points.\
#                             if False, only returns GW distance
#         * num_cores (int): number of parallel processes to run GW in
#         * chunk_size (int): chunk size for the iterator of all pairs of cells. \
#                  Larger size is faster but takes more memory, see \
#                  :meth:`multiprocessing.pool.Pool.imap` for details

#     Returns:
#         None (writes distance matrix of GW distances to file)
#     """
#     # Create output directory
#     if not os.path.exists(gw_results_dir):
#         os.makedirs(gw_results_dir)

#     dist_results = compute_GW_distance_matrix_preload_global(
#                       dist_mat_list_local,
#                       save_mat=save_mat,
#                       num_cores=num_cores,
#                       chunk_size=chunk_size)

#     # Save results - suffix name of output files is currently hardcoded
#     if save_mat:
#         np.savetxt(pj(gw_results_dir, file_prefix + "_gw_dist_mat.txt"),
#                    np.array([res[0] for res in dist_results]), fmt='%.8f')
#         np.savez_compressed(pj(gw_results_dir, file_prefix +
#             "_gw_matching.npz"), *[res[1] for res in dist_results])
#     else:
#         np.savetxt(pj(gw_results_dir, file_prefix + "_gw_dist_mat.txt"),
#          np.array(dist_results), fmt='%.8f')
