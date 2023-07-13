"""
Functionality to compute Gromov-Wasserstein distances\
using algorithms in Peyre et al. ICML 2016
"""
from __future__ import annotations
from dataclasses import dataclass

# std lib dependencies
import itertools as it
import time
import csv
from typing import List, Iterator, TypeVar, Optional
from math import sqrt, ceil

import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# external dependencies
import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402
from scipy.sparse import coo_array  # noqa: E402
from multiprocessing import Pool  # noqa: E402

from .gw_cython import (  # noqa: E402
    GW_cell,
    gw_cython_core,
)

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


def n_c_2(n: int):
    return (n * (n - 1)) // 2


def icdm_csv_validate(intracell_csv_loc: str) -> None:
    """
    Raise an exception if the file in intracell_csv_loc fails to pass formatting tests.
    Else return None.
    :param intracell_csv_loc: The (full) file path for the CSV file containing the intracell
    distance matrix.

    The file format is as follows:
    * A line whose first character is '#' is discarded as a comment.
    * The first line which is not a comment is discarded as a "header" - this line may
      contain the column titles for each of the columns.
    * Values separated by commas. Whitespace is not a separator.
    * The first value in the first non-comment line should be the string 'cell_id', and
      all values in the first column after that should be the cell id for that line.
    * All values after the first column should be floats.
    * Not including the cell id in the first column, each row except the header should contain
      the entries of an intracell distance matrix lying strictly above the diagonal,
      as in the footnotes of
      https://docs.scipy.org/doc/scipy/reference/\
      generated/scipy.spatial.distance.squareform.html
    """
    with open(intracell_csv_loc, "r", newline="") as icdm_infile:
        csv_reader = csv.reader(icdm_infile, delimiter=",")
        header = next(csv_reader)
        while header[0] == "#":
            header = next(csv_reader)
        if header[0] != "cell_id":
            raise ValueError("Expects header on first line starting with 'cell_id' ")
        linenum = 1
        for line in csv_reader:
            if line[0] == "#":
                continue
            for value in line[1:]:
                try:
                    float(value)
                except ValueError:
                    print(
                        "Unexpected value at file line "
                        + str(linenum)
                        + ", could not convert value"
                        + str(value)
                        + " to a float"
                    )
                    raise

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
        _, first_line = next(csv_outer_reader)
        while first_line[0] == "#":
            _, first_line = next(csv_outer_reader)
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


def cell_iterator_csv(
    intracell_csv_loc: str,
) -> Iterator[tuple[str, npt.NDArray[np.float_]]]:
    """
    :param intracell_csv_loc: A full file path to a csv file.
     Return an iterator over cells in a directory. Intracell distance matrices are in squareform.
    """
    icdm_csv_validate(intracell_csv_loc)
    with open(intracell_csv_loc, "r", newline="") as icdm_csvfile:
        csv_reader = csv.reader(icdm_csvfile, delimiter=",")
        # Assume a header
        next(csv_reader)
        while ell := next(csv_reader, None):
            cell_name = ell[0]
            arr = squareform(
                np.array([float(x) for x in ell[1:]], dtype=np.float64),
                force="tomatrix",
            )
            yield cell_name, arr


def cell_pair_iterator_csv(
    intracell_csv_loc: str, chunk_size: int
) -> Iterator[
    tuple[
        tuple[int, str, npt.NDArray[np.float_]], tuple[int, str, npt.NDArray[np.float_]]
    ]
]:
    """
    :param intracell_csv_loc: A full file path to a csv file.

    Return an iterator over pairs of cells in a directory, of the form
    ((indexA, nameA, distance_matrixA),(indexB, nameB, distance_matrixB)).
    Intracell distance matrices are in squareform.
    """
    batched_it = _batched_cell_list_iterator_csv(intracell_csv_loc, chunk_size)
    return it.chain.from_iterable(
        (
            filter(lambda tup: tup[0][0] < tup[1][0], it.product(t1, t2))
            for t1, t2 in batched_it
        )
    )


# This is one of two versions of this function; this one is simpler and only expects
# GW distances. The other is more complicated,
# and expects the coupling matrices and the GW distances.
def write_gw_dists(
    gw_dist_csv_loc: str,
    name_name_dist: Iterator[tuple[str, str, float]],
    verbose: Optional[bool] = False,
) -> None:
    """
    :param gw_dist_csv_loc: A file path to a (not necessarily existing) file.
    :param name_name_dist: entries are of the form (cellA_name,cellB_name, gw_dist).
    :param verbose: Print timing information.

    Given an iterator name_name_dist containing pairwise GW distances between cells,
    writes these entries to a csv file.
    If the file exists it will be deleted and overwritten.
    """
    chunk_size = 100
    counter = 0
    start = time.time()
    batched = _batched(name_name_dist, chunk_size)
    with open(gw_dist_csv_loc, "w", newline="") as gw_csv_file:
        csvwriter = csv.writer(gw_csv_file, delimiter=",")
        header = ["first_object", "second_object", "dist"]
        csvwriter.writerow(header)
        for batch in batched:
            counter += len(batch)
            csvwriter.writerows(batch)
            now = time.time()
            if verbose:
                print("Time elapsed: " + str(now - start))
                print("Cell pairs computed: " + str(counter))
    stop = time.time()
    print(
        "Computation finished. Computed "
        + str(counter)
        + " cell pairs."
        + " Time elapsed: "
        + str(stop - start)
    )


@dataclass
class GW_cell_pair_data:
    cellA_name: str
    cellA_sidelength: int
    cellB_name: str
    cellB_sidelength: int
    coupling_matrix: npt.NDArray[
        np.float_
    ]  # of shape (cellA_sidelength,cellB_sidelength)
    gw_dist: float


def write_dists_and_coupling_mats(
    gw_dist_csv_loc: str,
    gw_coupling_mat_csv_loc: str,
    gw_cell_pair_data: Iterator[GW_cell_pair_data],
    chunk_size: int = 500,
    verbose: Optional[bool] = False,
) -> None:
    """Given an iterator over cell pairs with computed intracell distance matrices,
    writes these entries to a pair of CSV files. One will contain the pairwise
    Gromov-Wasserstein distances, and the other will contain the coupling matrices.

    :param gw_dist_csv_loc: A file path to a (not necessarily existing) file.
    :param gw_coupling_mat_csv_loc: A file path to a (not necessarily existing) file.
    :param gw_cell_pair_data: Iterator over GW cell pair data. The cell pairs *should*
    have coupling matrices.
    :param verbose: Print timing information.
    """

    counter = 0
    start = time.time()
    batched = _batched(gw_cell_pair_data, chunk_size)
    with open(gw_dist_csv_loc, "w", newline="") as gw_dist_csv_file, open(
        gw_coupling_mat_csv_loc, "w", newline=""
    ) as gw_coupling_mat_csv_file:
        # Create the files and write headers to them.
        dist_writer = csv.writer(gw_dist_csv_file, delimiter=",")
        coupling_writer = csv.writer(gw_coupling_mat_csv_file, delimiter=",")
        dist_header = ["first_object", "second_object", "gw_dist"]
        dist_writer.writerow(dist_header)
        coupling_header = [
            "first_object",
            "first_object_sidelength",
            "second_object",
            "second_object_sidelength",
            "num_non_zero",
            "coupling",
        ]
        coupling_writer.writerow(coupling_header)
        # Go through all cells in the iterator and write them to files
        for batch in batched:
            for cell_pair in batch:
                if cell_pair.coupling_matrix is None:
                    raise ValueError("Coupling matrix is none.")
            couplings = [
                [
                    cell_pair.cellA_name,
                    cell_pair.cellA_sidelength,
                    cell_pair.cellB_name,
                    cell_pair.cellB_sidelength,
                ]
                + list(map(str, cell_pair.coupling_matrix.flatten()))
                for cell_pair in batch
            ]
            dists = [
                (cell_pair.cellA_name, cell_pair.cellB_name, cell_pair.gw_dist)
                for cell_pair in batch
            ]
            counter += len(batch)
            dist_writer.writerows(dists)
            coupling_writer.writerows(couplings)
            now = time.time()
            if verbose:
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


def _coupling_mat_reformat(coupling_mat: npt.NDArray[np.float_]) -> list[float | int]:
    """
    Convert a sparse coupling matrix to something that can be written to a csv file.
    """
    return [x for ell in coupling_mat for x in ell]
    coo = coo_array(coupling_mat)
    ell = [coo.nnz]
    ell += list(coo.data)
    ell += list(coo.row)
    ell += list(coo.col)
    return ell


def _gw_generator_couplings(
    GW_cells: list[GW_cell],
) -> Iterator[tuple[int, int, npt.NDArray[np.float_], float]]:
    N = len(GW_cells)
    for i in range(N):
        A = GW_cells[i]
        for j in range(i + 1, N):
            B = GW_cells[j]
            coupling_mat, gw_dist = gw_cython_core(
                A.dmat,
                A.distribution,
                A.dmat_dot_dist,
                A.cell_constant,
                B.dmat,
                B.distribution,
                B.dmat_dot_dist,
                B.cell_constant,
            )
            yield (i, j, coupling_mat, gw_dist)


def _init_gw_pool(GW_cells: list[GW_cell]):
    global _GW_CELLS
    _GW_CELLS = GW_cells


def _gw_index(p: tuple[int, int]):
    i, j = p
    A: GW_cell
    B: GW_cell
    A = _GW_CELLS[i]
    B = _GW_CELLS[j]
    coupling_mat, gw_dist = gw_cython_core(
        A.dmat,
        A.distribution,
        A.dmat_dot_dist,
        A.cell_constant,
        B.dmat,
        B.distribution,
        B.dmat_dot_dist,
        B.cell_constant,
    )
    return (i, j, coupling_mat, gw_dist)


def gw_pairwise(
    cells: list[
        tuple[
            npt.NDArray[np.float_],  # Squareform distance matrix
            npt.NDArray[np.float_],  # Probability distribution on cells
        ]
    ],
    num_processes: int,
    names: Optional[list[str]] = None,
    gw_dist_csv: Optional[str] = None,
    gw_coupling_mat_csv: Optional[str] = None,
    return_coupling_mats: bool = False,
) -> tuple[
    npt.NDArray[np.float_],  # Pairwise GW distance matrix (Squareform)
    Optional[list[tuple[int, int, npt.NDArray[np.float_]]]],
]:
    """Compute the pairwise Gromov-Wasserstein distances between cells,
    possibly along with their coupling matrices.

    If appropriate file names are supplied, the output is also written to file.
    If computing a large number of coupling matrices, it is suggested not to
    return the coupling matrices, and instead write them to file.

    :param cells: A list of pairs (A,a) where `A` is a squareform intracell
        distance matrix and `a` is a probability distribution on the cells of
        `A`.
    :param num_processes: How many Python processes to run in parallel for the computation.
    :param names: A list of unique cell identifiers, where names[i] is the identifier
        for cell i. This argument is required if gw_dist_csv is not None, or if
        gw_coupling_mat_csv is not None, and is ignored otherwise.
    :param gw_dist_csv: If this field is a string giving a file path, the GW distances
        will be written to this file. A list of cell names must be supplied.
    :param gw_coupling_mat_csv: If this field is a string giving a file path, the GW coupling
        matrices will be written to this file. A list of cell names must be supplied.
    :param return_coupling_mats: Whether the function should return the coupling matrices.
        Please be warned that for a large
        number of cells, `couplings` will be large, and memory consumption will be high.
        If `return_coupling_mats` is False, returns `(gw_dmat, None)`.
        This argument is independent of whether the coupling matrices are written to a file;
        one may return the coupling matrices, write them to file, both, or neither.

    :return: If `return_coupling_mats` is True,
        returns `( gw_dmat, couplings )`,
        where gw_dmat is a square matrix whose (i,j) entry is the GW distance
        between two cells, and `couplings` is a list of tuples (i,j,
        coupling_mat) where `i,j` are indices corresponding to positions in the list `cells`
        and `coupling_mat` is a coupling matrix between the two cells.
        If `return_coupling_mats` is False, returns `(gw_dmat, None)`.
    """

    GW_cells = []
    for A, a in cells:
        GW_cells.append(GW_cell(A, a, A @ a, ((A * A) @ a) @ a))
    num_cells = len(cells)
    gw_dmat = np.zeros((num_cells, num_cells))
    if return_coupling_mats is not None:
        gw_coupling_mats = []
    if gw_dist_csv is not None:
        gw_dist_file = open(gw_dist_csv, "w", newline="")
        gw_dist_writer = csv.writer(gw_dist_file)
        gw_dist_writer.writerow(["first_object,second_object,gw_distance"])
    if gw_coupling_mat_csv is not None:
        gw_coupling_mat_file = open(gw_coupling_mat_csv, "w", newline="")
        gw_coupling_mat_writer = csv.writer(gw_coupling_mat_file)
    ij = it.combinations(range(num_cells), 2)
    with Pool(
        initializer=_init_gw_pool, initargs=(GW_cells,), processes=num_processes
    ) as pool:
        gw_data = pool.imap_unordered(_gw_index, ij)
        gw_data_batched = _batched(gw_data, 2000)
        for batch in gw_data_batched:
            for i, j, coupling_mat, gw_dist in batch:
                gw_dmat[i, j] = gw_dist
                gw_dmat[j, i] = gw_dist
                if return_coupling_mats:
                    gw_coupling_mats.append((i, j, coupling_mat))
            if gw_dist_csv is not None:
                if names is None:
                    raise Exception(
                        "Must supply list of cell identifiers for writing to file."
                    )
                writelist = [
                    (names[i], names[j], str(gw_dist)) for (i, j, _, gw_dist) in batch
                ]
                gw_dist_writer.writerows(writelist)
            if gw_coupling_mat_csv is not None:
                if names is None:
                    raise Exception(
                        "Must supply list of cell identifiers for writing to file."
                    )
                writelist = [
                    (names[i], names[j], str(coupling_mat))
                    for (i, j, coupling_mat, _) in batch
                ]
                gw_coupling_mat_writer.writerows(writelist)
    if gw_dist_csv is not None:
        gw_dist_file.close()
    if gw_coupling_mat_csv is not None:
        gw_coupling_mat_file.close()

    if return_coupling_mats:
        return (gw_dmat, gw_coupling_mats)
    return (gw_dmat, None)


def compute_gw_distance_matrix(
    intracell_csv_loc: str,
    gw_dist_csv_loc: str,
    num_processes: int,
    gw_coupling_mat_csv_loc: Optional[str] = None,
    return_coupling_mats: bool = False,
    verbose: Optional[bool] = False,
) -> tuple[
    npt.NDArray[np.float_],  # Pairwise GW distance matrix (Squareform)
    Optional[list[tuple[int, int, npt.NDArray[np.float_]]]],
]:
    """Compute the matrix of pairwise Gromov-Wasserstein distances between cells.

    :param intracell_csv_loc: A file containing the intracell distance matrices
        for all cells.
    :param gw_dist_csv_loc: An output file containing the Gromov-Wasserstein
        distances, which will be created if it does not exist and overwritten if it
        does.
    :param gw_coupling_mat_csv_loc: If this argument is not None, for each pair
        of cells, the coupling matrices will be retained and written to this output
        file. If this argument is None, the coupling matrices will be discarded. Be
        warned that the coupling matrices are large.
    """

    cell_names_dmats = list(cell_iterator_csv(intracell_csv_loc))
    names: list[str]
    names = [name for name, _ in cell_names_dmats]
    # List of pairs (A, a) where A is a square matrix and `a` a probability distribution
    cell_dms: list[tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]]
    cell_dms = [
        (c := cell, np.ones((n := c.shape[0],)) / n) for _, cell in cell_names_dmats
    ]

    return gw_pairwise(
        cell_dms,
        num_processes,
        names,
        gw_dist_csv_loc,
        gw_coupling_mat_csv_loc,
        return_coupling_mats,
    )
