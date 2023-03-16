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
    Iterator,
    TypeVar,
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


def icdm_csv_validate(intracell_csv_loc: str) -> None:
    with open(intracell_csv_loc, "r", newline="") as icdm_infile:
        csv_reader = csv.reader(icdm_infile, delimiter=",")
        header = next(csv_reader)
        if header[0] != "cell_id":
            raise ValueError("Expects header on first line starting with 'cell_id' ")
        linenum = 1
        for line in csv_reader:
            try:
                float(line[1])
            except:
                raise ValueError(
                    "Unexpected value at file line " + str(linenum) + ", token 2"
                )
            line_length = len(header[1:])
            side_length = ceil(sqrt(2 * line_length))
            if side_length * (side_length - 1) != 2 * line_length:
                raise ValueError(
                    "Line " + str(linenum) + " is not in upper triangular form."
                )
            linenum += 1


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

    :return: An iterator over pairs (list1, list2), where each element \
    in list1 and list2 is a triple
    (cell_id, cell_name, icdm), where cell_id is a natural number,
    cell_name is a string, and icdm is a square n x n distance matrix.
    cell_id is guaranteed to be unique.

    Increasing chunk_size increases memory usage but reduces the frequency of file reads.

    Note that for parallelization concerns it is best to communicate large batches of work \
    to a child process at one time. However, numpy is already parallelizing the GW computations \
    under the hood so this is probably an irrelevant concern.
    """

    # Validate input
    icdm_csv_validate(intracell_csv_loc)

    with open(intracell_csv_loc, "r", newline="") as icdm_csvfile_outer:
        csv_outer_reader = enumerate(csv.reader(icdm_csvfile_outer, delimiter=","))
        next(csv_outer_reader)
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
            print(first_outer_id)
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


def gw(fst_mat: npt.NDArray, snd_mat: npt.NDArray) -> float:
    """
    Readability/convenience wrapper for ot.gromov.gromov_wasserstein.
    
    :param A: Squareform distance matrix.
    :param B: Squareform distance matrix.
    :return: GW distance between them with square_loss optimization and \
    uniform distribution on points.
    """
    _, log = ot.gromov.gromov_wasserstein(
        fst_mat,
        snd_mat,
        ot.unif(fst_mat.shape[0]),
        ot.unif(snd_mat.shape[0]),
        "square_loss",
        log=True,
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
    gw_csv_loc: str, name_name_dist_coupling: Iterator[List[str | float | int]]
) -> None:
    chunk_size = 100
    counter = 0
    start = time.time()
    batched = _batched(name_name_dist_coupling, chunk_size)
    with open(gw_csv_loc, "w", newline="") as gw_csv_file:
        csvwriter = csv.writer(gw_csv_file, delimiter=",")
        header = ["first_object", "second_object", "gw_dist"]
        csvwriter.writerow(header)
        for batch in batched:
            counter += len(batch)
            csvwriter.writerows(batch)
            now = time.time()
            print("Time elapsed: " + str(now - start))
            print("Cell pairs computed: " + str(counter))
    stop = time.time()
    print(
        "Computation finished. Computed "
        + str(counter)
        + " many cell pairs."
        + " Time elapsed: "
        + str(stop - start)
    )


def gw_linear(
    cellA_name: str,
    cellA_icdm: npt.NDArray[np.float_],
    cellB_name: str,
    cellB_icdm: npt.NDArray[np.float_],
    save_mat: bool,
) -> List[int | float | str]:
    """
    Compute the Gromov-Wasserstein distance between cells A and B.
    Return all relevant information in a single list of strings which can be written
    to file directly.

    :param cellA_icdm: expected to be in vector form.
    :param cellB_icdm: expected to be in vector form.
    
    :param save_mat: if True, the coupling matrix will be included in the return list.\
    Otherwise, false.
    :return: a list
    [cellA_name, cellA_sidelength, cellB_name, cellB_sidelength, \
    gw_dist(, gw coupling matrix entries)]
    where the gw coupling matrix entries are optional.
    """
    cellA_square = squareform(cellA_icdm)
    cellA_sidelength = cellA_square.shape[0]
    cellB_square = squareform(cellB_icdm)
    cellB_sidelength = cellB_square.shape[0]
    coupling_mat, log = ot.gromov.gromov_wasserstein(
        cellA_square,
        cellB_square,
        ot.unif(cellA_sidelength),
        ot.unif(cellB_sidelength),
        "square_loss",
        log=True,
    )

    retlist = [
        cellA_name,
        cellA_sidelength,
        cellB_name,
        cellB_sidelength,
        log["gw_dist"],
    ]
    if save_mat:
        coupling_mat_list: list[float] = [x for ell in list(coupling_mat) for x in ell]
        retlist += coupling_mat_list
    return retlist


def compute_and_save_gw_distance_matrix(
    intracell_csv_loc: str, gw_csv_loc: str, save_mat: bool
) -> None:
    chunk_size = 100
    cell_pairs = cell_pair_iterator_csv(intracell_csv_loc, chunk_size)
    gw_dists: Iterator[List[str | float | int]]
    gw_dists = (
        gw_linear(cellA_name, cellA_icdm, cellB_name, cellB_icdm, save_mat)
        for (_, cellA_name, cellA_icdm), (_, cellB_name, cellB_icdm) in cell_pairs
    )
    write_gw(gw_csv_loc, gw_dists)
