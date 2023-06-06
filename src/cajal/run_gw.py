"""
Functionality to compute Gromov-Wasserstein distances\
using algorithms in Peyre et al. ICML 2016
"""
# std lib dependencies
import itertools as it
import time
import csv
from typing import List, Iterable, Iterator, TypeVar, Optional
from math import sqrt, ceil


# external dependencies
import ot
import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import squareform
from scipy.sparse import coo_array

from .slb import slb2
from .pogrow import pogrow
from .gw_cython import gw_cython

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
        while header[0] == "#":
            header = next(csv_reader)
        if header[0] != "cell_id":
            raise ValueError("Expects header on first line starting with 'cell_id' ")
        linenum = 1
        for line in csv_reader:
            if line[0] == "#":
                continue
            try:
                float(line[1])
            except ValueError:
                print("Unexpected value at file line " + str(linenum) + ", token 2")
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
    batched_it = _batched_cell_list_iterator_csv(intracell_csv_loc, chunk_size)
    return it.chain.from_iterable(
        (
            filter(lambda tup: tup[0][0] < tup[1][0], it.product(t1, t2))
            for t1, t2 in batched_it
        )
    )


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
    gw_dist = log["gw_dist"]
    # Should be unnecessary but floating point
    if gw_dist < 0:
        gw_dist = 0
    return sqrt(gw_dist) / 2.0


# def slb2(fst_mat: npt.NDArray, snd_mat: npt.NDArray) -> float:
#     """
#     Accepts two vectorform distance matrices.
#     """
#     fst_mat = np.sort(fst_mat)
#     snd_mat = np.sort(snd_mat)
#     ND, MD = fst_mat.shape[0], snd_mat.shape[0]
#     N, M = ceil(sqrt(2 * ND)), ceil(sqrt(2 * MD))
#     assert ND * 2 == N * (N - 1)
#     assert MD * 2 == M * (M - 1)
#     fst_diffs = np.diff(fst_mat, prepend=0.0)
#     snd_diffs = np.diff(snd_mat, prepend=0.0)
#     fst_mat_x = np.linspace(start=1 / N + 2 / (N**2), stop=1, num=ND)
#     snd_mat_x = np.linspace(start=1 / M + 2 / (M**2), stop=1, num=MD)
#     x = np.concatenate((fst_mat_x, snd_mat_x))
#     assert x.shape == (ND + MD,)
#     indices = np.argsort(x)
#     T = np.concatenate((fst_diffs, -snd_diffs))[indices]
#     np.cumsum(T, out=T)
#     np.abs(T, out=T)
#     np.square(T, out=T)
#     a = x[indices]
#     t = np.diff(a, append=a[-1])
#     assert np.all(t >= 0.0)
#     return sqrt(np.dot(T, t)) / 2


def compute_slb2_distance_matrix(
    intracell_csv_loc: str,
    slb2_dist_csv_loc: str,
    verbose: Optional[bool] = False,
) -> None:
    cell_pairs = cell_pair_iterator_csv(intracell_csv_loc, 100)
    write_gw_dists(
        slb2_dist_csv_loc,
        (
            (
                cellA_name,
                cellB_name,
                slb2(
                    squareform(cellA_icdm, force="tovector"),
                    squareform(cellB_icdm, force="tovector"),
                ),
            )
            for (_, cellA_name, cellA_icdm), (_, cellB_name, cellB_icdm) in cell_pairs
        ),
        True,
    )


def write_gw_dists(
    gw_dist_csv_loc: str,
    name_name_dist: Iterator[tuple[str, str, float]],
    verbose: Optional[bool] = False,
) -> None:
    chunk_size = 100
    counter = 0
    start = time.time()
    batched = _batched(name_name_dist, chunk_size)
    with open(gw_dist_csv_loc, "w", newline="") as gw_csv_file:
        csvwriter = csv.writer(gw_csv_file, delimiter=",")
        header = ["first_object", "second_object", "gw_dist"]
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


def write_dists_and_coupling_mats(
    gw_dist_csv_loc: str,
    gw_coupling_mat_csv_loc: str,
    name_name_dist_coupling: Iterator[
        tuple[tuple[str, int, str, int, list[float]], tuple[str, str, float]]
    ],
    chunk_size: int = 500,
    verbose: Optional[bool] = False,
) -> None:
    counter = 0
    start = time.time()
    batched = _batched(name_name_dist_coupling, chunk_size)
    with open(gw_dist_csv_loc, "w", newline="") as gw_dist_csv_file, open(
        gw_coupling_mat_csv_loc, "w", newline=""
    ) as gw_coupling_mat_csv_file:
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
        for batch in batched:
            couplings, dists = [list(tup) for tup in zip(*batch)]
            couplings = [
                [A_name, A_sidelength, B_name, B_sidelength] + coupling_mat
                for (
                    A_name,
                    A_sidelength,
                    B_name,
                    B_sidelength,
                    coupling_mat,
                ) in couplings
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
    # return [x for ell in coupling_mat for x in ell]
    coo = coo_array(coupling_mat)
    ell = [coo.nnz]
    ell += list(coo.data)
    ell += list(coo.row)
    ell += list(coo.col)
    return ell


def _gw_dist_coupling(
    cellA_name: str,
    cellA_icdm: npt.NDArray[np.float_],
    cellB_name: str,
    cellB_icdm: npt.NDArray[np.float_],
) -> tuple[tuple[str, int, str, int, list[float]], tuple[str, str, float]]:
    cellA_sidelength = cellA_icdm.shape[0]
    cellB_sidelength = cellB_icdm.shape[0]
    coupling_mat, log = ot.gromov.gromov_wasserstein(
        cellA_icdm,
        cellB_icdm,
        ot.unif(cellA_sidelength),
        ot.unif(cellB_sidelength),
        "square_loss",
        log=True,
    )
    coupling_mat = _coupling_mat_reformat(coupling_mat)
    gw_dist = log["gw_dist"]
    # This should be unnecessary but floating point reasons
    if gw_dist < 0:
        gw_dist = 0
    return (cellA_name, cellA_sidelength, cellB_name, cellB_sidelength, coupling_mat), (
        cellA_name,
        cellB_name,
        sqrt(gw_dist) / 2.0,
    )


def gw_custom(
    cell_list1: list[npt.NDArray[np.float_]],  # Squareform
    cell_list2: list[npt.NDArray[np.float_]],  # Squareform
    distributions_1: list[npt.NDArray[np.float_]],
    distributions_2: list[npt.NDArray[np.float_]],
    indices: Iterable[tuple[int, int]],
    max_iters_ot: int = 100000,
    max_iters_descent: int = 1000,
) -> Iterable[float]:
    cell_list1 = [np.asarray(a, dtype=np.float64, order="C") for a in cell_list1]
    cell_list2 = [np.asarray(a, dtype=np.float64, order="C") for a in cell_list2]
    c_C = [
        np.matmul(np.multiply(A, A), distr)[:, np.newaxis]
        for A, distr in zip(cell_list1, distributions_1)
    ]
    c_Cbar = [
        np.matmul(distr[np.newaxis, :], np.multiply(A.T, A.T))
        for A, distr in zip(cell_list2, distributions_2)
    ]
    retlist: list[float] = []
    for i, j in indices:
        retlist.append(
            gw_cython(
                cell_list1[i],
                cell_list2[j],
                distributions_1[i],
                distributions_2[j],
                c_C[i],
                c_Cbar[j],
                max_iters_ot,
                max_iters_descent,
            )
        )
    return retlist


def compute_gw_distance_matrix(
    intracell_csv_loc: str,
    gw_dist_csv_loc: str,
    gw_coupling_mat_csv_loc: Optional[str] = None,
    verbose: Optional[bool] = False,
) -> None:
    """
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
    chunk_size = 100
    cell_pairs = cell_pair_iterator_csv(intracell_csv_loc, chunk_size)

    if gw_coupling_mat_csv_loc is not None:
        write_data = (
            _gw_dist_coupling(cellA_name, cellA_icdm, cellB_name, cellB_icdm)
            for (_, cellA_name, cellA_icdm), (_, cellB_name, cellB_icdm) in cell_pairs
        )
        write_dists_and_coupling_mats(
            gw_dist_csv_loc, gw_coupling_mat_csv_loc, write_data, verbose=verbose
        )
    else:
        write_dists = (
            (cellA_name, cellB_name, gw(cellA_icdm, cellB_icdm))
            for (_, cellA_name, cellA_icdm), (_, cellB_name, cellB_icdm) in cell_pairs
        )
        write_gw_dists(gw_dist_csv_loc, write_dists, verbose=verbose)


def pogrow_pairwise(a: list[npt.NDArray[np.float_]], it: int, alpha: float):
    """
    elements of a should be in square form
    """
    C_sq = [np.average(np.multiply(A, A), axis=1) for A in a]
    Cbar_sq = [np.average(np.multiply(A, A), axis=0) for A in a]
    retlist = []
    for i in range(len(a)):
        a_i = a[i]
        C_sq_i = C_sq[i]
        for j in range(i + 1, len(a)):
            LC_tensor_C = C_sq_i[:, np.newaxis] + Cbar_sq[j]
            assert len(LC_tensor_C.shape) == 2
            T = pogrow(a_i, a[j], it, alpha)
            LC_tensor_C -= 2 * np.matmul(a[i], np.matmul(T, a[j].T))
            retlist.append(sqrt(np.sum(np.multiply(LC_tensor_C, T))) / 2)
    return retlist
