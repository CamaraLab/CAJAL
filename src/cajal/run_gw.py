"""
Functionality to compute Gromov-Wasserstein distances\
using algorithms in Peyre et al. ICML 2016
"""
# import os
# std lib dependencies
import itertools as it
import time
import csv

from typing import List, Optional, Tuple, Iterable, Iterator, Dict, TypedDict, TypeVar, Callable

from pathos.pools import ProcessPool
from multiprocess.shared_memory import SharedMemory
from multiprocess import set_start_method
from math import sqrt, ceil
import statistics

#external dependencies
import ot
import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import squareform
from scipy.sparse import coo_matrix


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
