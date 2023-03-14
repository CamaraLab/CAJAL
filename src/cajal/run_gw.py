"""
Functionality to compute Gromov-Wasserstein distances\
using algorithms in Peyre et al. ICML 2016
"""
# std lib dependencies
import itertools as it
import time
import csv
from typing import (
    List,
    Optional,
    Tuple,
    Iterable,
    Iterator,
    Dict,
    TypedDict,
    TypeVar,
    Callable,
)
from math import sqrt, ceil


# external dependencies
import ot
import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import squareform

T = TypeVar("T")


def _batched(itera: Iterator[T], n: int) -> Iterator[List[T]]:
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    while batch := list(it.islice(itera, n)):
        yield batch


def _is_sorted(int_list: List[int]) -> bool:
    if len(int_list) <= 1:
        return True
    return all(map(lambda tup: tup[0] <= tup[1], zip(int_list[:-1], int_list[1:])))


def _batched_cell_list_iterator_csv(
    intracell_csv_loc: str, chunk_size: int
) -> Iterator[
    tuple[
        list[tuple[int, str, npt.NDArray[np.float_]]],
        list[tuple[int, str, npt.NDArray[np.float_]]],
    ]
]:
    """
    :param intracell_csv_loc: A full file path to a csv file.
    :param chunk_size: A size parameter.

    :return: An iterator over pairs (list1, list2), where each element in list1 and list2 is a triple
    (cell_id, cell_name, icdm), where cell_id is a natural number,
    cell_name is a string, and icdm is a square n x n distance matrix.
    cell_id is guaranteed to be unique.

    Increasing chunk_size increases memory usage but reduces the frequency of file reads.

    Note that for parallelization concerns it is best to communicate large batches of work \
    to a child process at one time. However, numpy is already parallelizing the GW computations \
    under the hood so this is probably an irrelevant concern.
    """

    with open(intracell_csv_loc, newline="") as icdm_csvfile_outer:
        csv_outer_reader = enumerate(csv.reader(icdm_csvfile_outer, delimiter=","))
        _, header = next(csv_outer_reader)
        assert header[0] == "cell_id"
        line_length = len(header[1:])
        side_length = ceil(sqrt(2 * line_length))
        assert side_length * (side_length - 1) == 2 * line_length
        batched_outer = _batched(csv_outer_reader, chunk_size)
        for outer_batch in batched_outer:
            outer_list = [
                (
                    cell_id,
                    ell[0],
                    squareform(np.array([float(x) for x in ell[1:]], dtype=np.float_)),
                )
                for (cell_id, ell) in outer_batch
            ]
            first_outer_id = outer_list[0][0]
            with open(intracell_csv_loc, newline="") as icdm_csvfile_inner:
                csv_inner_reader = enumerate(
                    csv.reader(icdm_csvfile_inner, delimiter=",")
                )
                while next(csv_inner_reader)[0] < first_outer_id:
                    pass
                batched_inner = _batched(csv_inner_reader, chunk_size)
                for inner_batch in batched_inner:
                    inner_list = [
                        (
                            cell_id,
                            ell[0],
                            squareform(
                                np.array([float(x) for x in ell[1:]], dtype=np.float64)
                            ),
                        )
                        for (cell_id, ell) in inner_batch
                    ]
                    yield outer_list, inner_list


def cell_pair_iterator_csv(
    intracell_csv_loc: str, chunk_size: int
) -> Iterator[
    tuple[
        tuple[int, str, npt.NDArray[np.float_]], tuple[int, str, npt.NDArray[np.float_]]
    ]
]:
    batched_it = _batched_cell_list_iterator_csv(intracell_csv_loc, chunk_size)
    return it.chain.from_iterable((zip(t1, t2) for t1, t2 in batched_it))


def gw(A: npt.NDArray, B: npt.NDArray) -> float:
    """
    Readability/convenience wrapper for ot.gromov.gromov_wasserstein.
    
    :param A: Squareform distance matrix.
    :param B: Squareform distance matrix.
    :return: GW distance between them with square_loss optimization and \
    uniform distribution on points.
    """
    _, log = ot.gromov.gromov_wasserstein(
        A, B, ot.unif(A.shape[0]), ot.unif(B.shape[0]), "square_loss", log=True
    )
    return log["gw_dist"]


def gw_with_coupling_mat(
    A: npt.NDArray, B: npt.NDArray
) -> tuple[float, npt.NDArray[np.float_]]:
    """
    Readability/convenience wrapper for ot.gromov.gromov_wasserstein.
    
    :param A: Vectorform distance matrix.
    :param B: Vectorform distance matrix.
    :return: Pair (gw_dist, coupling_mat), where gw_dist is the GW distance between A and B \
       with square_loss optimization and \
       uniform distribution on points, and coupling_mat is the best-fit coupling matrix.
    """
    A = squareform(A)
    B = squareform(B)
    coupling_mat, log = ot.gromov.gromov_wasserstein(
        A, B, ot.unif(A.shape[0]), ot.unif(B.shape[0]), "square_loss", log=True
    )
    return log["gw_dict"], coupling_mat


def write_gw(
    gw_csv_loc: str,
    name_name_dist: Iterator[tuple[str, str, float]],
) -> None:
    chunk_size = 100
    counter = 0
    start = time.time()
    batched = _batched(name_name_dist, chunk_size)
    with open(gw_csv_loc, "a", newline="") as gw_csv_file:
        csvwriter = csv.writer(gw_csv_file, delimiter=",")
        header = ["first_object", "second_object", "gw_dist"]
        for batch in batched:
            counter += len(batch)
            csvwriter.writerows(batch)
            now = time.time()
            print("Time elapsed: " + str(now - start))
            print("Cell pairs computed: " + counter)
    stop = time.time()
    print(
        "Computation finished. Computed "
        + str(counter)
        + " many cell pairs."
        + " Time elapsed: "
        + str(stop - start)
    )


def compute_and_save_gw_distance_matrix(
    intracell_csv_loc: str, gw_csv_loc: str
) -> None:
    chunk_size = 100
    cell_pairs = cell_pair_iterator_csv(intracell_csv_loc, chunk_size)
    gw_dists = (
        (name1, name2, gw(icdm1, icdm2))
        for (_, name1, icdm1), (_, name2, icdm2) in cell_pairs
    )
    write_gw(gw_csv_loc, gw_dists)


# def compute_and_save_gw_distance_matrix_w_coupling_mats(
#         intracell_csv_loc : str,
#         gw_csv_loc : str
# ) -> None:
#     chunk_size=100
#     with open(gw_csv_loc, 'a', newline='') as gw_csv_file:
#         csvwriter = csv.writer(gw_csv_file, delimiter=',')
#         with open(intracell_csv_loc, 'r', newline='') as icdm_csv:
#             header=next(csv.reader(icdm_csv, delimiter=','))
#             assert header[0]=='cell_id'
#             list_length = len(header[1:])
#         side_length : int = ceil(sqrt(2 * list_length))
#         assert (side_length * (side_length - 1) == list_length * 2)
#         if save_mat:
#             header = ["first_cell","second_cell","gw_dist"] + [*range(side_length ** 2)]
#         else:
#             header = ["first_cell","second_cell","gw_dist"]
#         csvwriter.writerow(header)
#         time_start=time.time()
#         cell_comparisons=0
#         for cell_list_1, cell_list_2 in _batched_cell_list_iterator_csv(intracell_csv_loc, chunk_size):
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
