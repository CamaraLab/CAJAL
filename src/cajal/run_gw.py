"""Functionality to compute Gromov-Wasserstein distances \
using algorithms in Peyre et al. ICML 2016."""

from __future__ import annotations

import csv

# std lib dependencies
import itertools as it
import sys
from typing import Iterator, List, Optional, TypeVar

if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm  # type: ignore[assignment]

from .cajal_types import Distribution, DistanceMatrix, Matrix
from .utilities import cell_iterator_csv, icdm_csv_validate, uniform
from multiprocessing import Pool

import numpy as np

# import numpy.typing as npt
from scipy.sparse import coo_matrix
from scipy.spatial.distance import squareform

# external dependencies
from threadpoolctl import ThreadpoolController

from .gw_cython import GW_cell, gw_cython_core

T = TypeVar("T")


def _batched(itera: Iterator[T], n: int) -> Iterator[List[T]]:
    """Batch data into tuples of length n. The last batch may be shorter."""
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
        list[tuple[int, str, DistanceMatrix]],
        list[tuple[int, str, DistanceMatrix]],
    ]
]:
    """
    Return an iterator over pairs of distinct cells from a file.

    Increasing chunk_size increases memory usage but reduces the frequency of file reads.
    Note that for parallelization concerns it is best to communicate large batches of work \
    to a child process at one time. However, numpy is already parallelizing the GW computations \
    under the hood so this is probably an irrelevant concern.

    :param intracell_csv_loc: A full file path to a csv file.
    :param chunk_size: A size parameter.
csv
    :return: An iterator over pairs (list1, list2), where each element \
    in list1 and list2 is a triple
    (cell_id, cell_name, icdm), where cell_id is a natural number,
    cell_name is a string, and icdm is a square n x n distance matrix.
    cell_id is guaranteed to be unique.
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
                    squareform(np.array([float(x) for x in ell[1:]], dtype=np.float64)),
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
) -> Iterator[tuple[tuple[int, str, DistanceMatrix], tuple[int, str, DistanceMatrix]]]:
    """
    Iterate over pairs of cells in a CSV in a memory efficient way.

    This is almost equivalent to
    itertools.combinations(cell_iterator_csv(intracell_csv_loc),2) but
    with more efficient file IO.

    :param intracell_csv_loc: A full file path to a csv file.
    :param chunk_size: How many lines to read from the file at a time. Does not affect output.
    :return: an iterator over pairs of cells, each entry is of the form
        ((indexA, nameA, distance_matrixA),(indexB, nameB, distance_matrixB)),
        where `indexA` is the line number in the file, and `indexA < indexB`.
    """
    batched_it = _batched_cell_list_iterator_csv(intracell_csv_loc, chunk_size)
    return it.chain.from_iterable(
        (
            filter(lambda tup: tup[0][0] < tup[1][0], it.product(t1, t2))
            for t1, t2 in batched_it
        )
    )


def _init_gw_pool(GW_cells: list[GW_cell]):
    global _GW_CELLS
    _GW_CELLS = GW_cells  # type: ignore[name-defined]


controller = ThreadpoolController()


@controller.wrap(limits=1, user_api="blas")
def _gw_index(p: tuple[int, int]) -> tuple[int, int, Matrix, float]:
    i, j = p
    A: GW_cell
    B: GW_cell
    A = _GW_CELLS[i]  # type: ignore[name-defined]
    B = _GW_CELLS[j]  # type: ignore[name-defined]
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


def csv_output_writer(
    names: list[str],
    gw_dist_csv: Optional[str],
    gw_coupling_mat_npz: Optional[str],
    results_iterator: Iterator[tuple[int, int, Matrix, float]],
) -> Iterator[tuple[int, int, Matrix, float]]:
    """Write the input to file, and return the output unchanged.

    If gw_distance_file is not None, then it will be created,
    and the given GW distances will be written to that file.

    If gw_coupling_mat_file is not None, then it will be created,
    and the given GW coupling matrices will be written to that file.
    """

    write_gw_distances: bool = gw_dist_csv is not None
    if write_gw_distances:
        gw_dist_file = open(gw_dist_csv, "w", newline="")  # type: ignore[arg-type]
        gw_dist_writer = csv.writer(gw_dist_file)
        gw_dist_writer.writerow(["first_object", "second_object", "gw_distance"])

    write_gw_coupling_mats: bool = gw_coupling_mat_npz is not None

    if write_gw_coupling_mats:
        first_names = list()
        second_names = list()
        coo_data = list()
        coo_row = list()
        coo_col = list()

    for i, j, coupling_mat, gw_dist in results_iterator:
        if write_gw_distances:
            gw_dist_writer.writerow([names[i], names[j], str(gw_dist)])
        if write_gw_coupling_mats:
            first_names.append(names[i])
            second_names.append(names[j])
            coo = coo_matrix(coupling_mat.astype(np.float32))
            coo_data.append(coo.data)
            coo_row.append(coo.row)
            coo_col.append(coo.col)
        yield (i, j, coupling_mat, gw_dist)
    if write_gw_distances:
        gw_dist_file.close()
    if write_gw_coupling_mats:
        np.savez(
            gw_coupling_mat_npz,
            first_names=np.array(first_names),
            second_names=np.array(second_names),
            coo_data=np.stack(coo_data),
            coo_row=np.stack(coo_row),
            coo_col=np.stack(coo_col),
        )


#
def gw_pairwise_parallel(
    cells: list[
        tuple[
            DistanceMatrix,  # Squareform distance matrix
            Distribution,  # Probability distribution on cells
        ]
    ],
    num_processes: int,
    names: Optional[list[str]] = None,
    gw_dist_csv: Optional[str] = None,
    gw_coupling_mat_csv: Optional[str] = None,
    return_coupling_mats: bool = False,
) -> tuple[
    DistanceMatrix,  # Pairwise GW distance matrix (Squareform)
    Optional[list[tuple[int, int, Matrix]]],
]:
    """Compute the pairwise Gromov-Wasserstein distances between cells.

    Optionally one can also compute their coupling matrices.
    If appropriate file names are supplied, the output is also written to file.
    If computing a large number of coupling matrices, for reduced memory consumption it
    is suggested not to return the coupling matrices, and instead write them to file.

    :param cells: A list of pairs (A,a) where `A` is a squareform intracell
        distance matrix and `a` is a probability distribution on the points of
        `A`.
    :param num_processes: How many Python processes to run in parallel for the computation.
    :param names: A list of unique cell identifiers, where names[i] is the identifier
        for cell i. This argument is required if gw_dist_csv is not None, or if
        gw_coupling_mat_csv is not None, and is ignored otherwise.
    :param gw_dist_csv: If this field is a string giving a file path, the GW distances
        will be written to this file. A list of cell names must be supplied.
    :param gw_coupling_mat_csv: If this field is a string giving a file path,
        the GW coupling
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
        GW_cells.append(GW_cell(A, a))
    num_cells = len(cells)
    gw_dmat = np.zeros((num_cells, num_cells))
    if return_coupling_mats is not None:
        gw_coupling_mats = []
    NN = len(GW_cells)
    total_num_pairs = int((NN * (NN - 1)) / 2)
    ij = tqdm(it.combinations(range(num_cells), 2), total=total_num_pairs)
    with Pool(
        initializer=_init_gw_pool, initargs=(GW_cells,), processes=num_processes
    ) as pool:
        gw_data: Iterator[tuple[int, int, Matrix, float]]
        gw_data = pool.imap_unordered(_gw_index, ij, chunksize=20)
        if (gw_dist_csv is not None) or (gw_coupling_mat_csv is not None):
            if names is None:
                raise Exception(
                    "Must supply list of cell identifiers for writing to file."
                )
            gw_data = csv_output_writer(
                names,
                gw_dist_csv,
                gw_coupling_mat_csv,
                gw_data,
            )
        for i, j, coupling_mat, gw_dist in gw_data:
            gw_dmat[i, j] = gw_dist
            gw_dmat[j, i] = gw_dist
            if return_coupling_mats:
                gw_coupling_mats.append((i, j, coupling_mat))
    if return_coupling_mats:
        return (gw_dmat, gw_coupling_mats)
    return (gw_dmat, None)


@controller.wrap(limits=1, user_api="blas")
def gw(
    A: DistanceMatrix,
    a: Distribution,
    B: DistanceMatrix,
    b: Distribution,
    max_iters_descent: int = 1000,
    max_iters_ot: int = 200000,
) -> tuple[Matrix, float]:
    """Compute the Gromov-Wasserstein distance between two metric measure spaces."""
    Aa = A @ a
    c_A = ((A * A) @ a) @ a
    Bb = B @ b
    c_B = ((B * B) @ b) @ b
    return gw_cython_core(A, a, Aa, c_A, B, b, Bb, c_B, max_iters_descent, max_iters_ot)


def compute_gw_distance_matrix(
    intracell_csv_loc: str,
    gw_dist_csv_loc: str,
    num_processes: int,
    gw_coupling_mat_npz_loc: Optional[str] = None,
    return_coupling_mats: bool = False,
) -> tuple[
    DistanceMatrix,  # Pairwise GW distance matrix (Squareform)
    Optional[list[tuple[int, int, Matrix]]],
]:
    """Compute the matrix of pairwise Gromov-Wasserstein distances between cells.

    This function is a wrapper for :func:`cajal.run_gw.gw_pairwise_parallel` except
    that it reads icdm's from a file rather than from a list.
    For the file format of icdm's see :func:`cajal.run_gw.icdm_csv_validate`.

    :param intracell_csv_loc: A file containing the intracell distance matrices
        for all cells.

    For other parameters see :func:`cajal.run_gw.gw_pairwise_parallel`.
    """
    cell_names_dmats = list(cell_iterator_csv(intracell_csv_loc))
    names: list[str]

    names = [name for name, _ in cell_names_dmats]
    # List of pairs (A, a) where A is a square matrix and `a` a probability distribution
    cell_dms: list[tuple[DistanceMatrix, Distribution]]
    cell_dms = [(c := cell, uniform(c.shape[0])) for _, cell in cell_names_dmats]

    return gw_pairwise_parallel(
        cell_dms,
        num_processes,
        names,
        gw_dist_csv_loc,
        gw_coupling_mat_npz_loc,
        return_coupling_mats,
    )
