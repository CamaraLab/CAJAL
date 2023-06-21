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
from scipy import sparse
from scipy import cluster
from scipy.sparse import coo_array
from multiprocessing import Pool

from .slb import slb2 as slb2_cython
from .pogrow import pogrow
from .gw_cython import gw_cython, frobenius, quantized_gw_2

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


def slb2(fst_mat: npt.NDArray, snd_mat: npt.NDArray) -> float:
    """
    Accepts two vectorform distance matrices.
    """
    fst_mat = np.sort(fst_mat)
    snd_mat = np.sort(snd_mat)
    ND, MD = fst_mat.shape[0], snd_mat.shape[0]
    N, M = ceil(sqrt(2 * ND)), ceil(sqrt(2 * MD))
    assert ND * 2 == N * (N - 1)
    assert MD * 2 == M * (M - 1)
    fst_diffs = np.diff(fst_mat, prepend=0.0)
    snd_diffs = np.diff(snd_mat, prepend=0.0)
    fst_mat_x = np.linspace(start=1 / N + 2 / (N**2), stop=1, num=ND)
    snd_mat_x = np.linspace(start=1 / M + 2 / (M**2), stop=1, num=MD)
    x = np.concatenate((fst_mat_x, snd_mat_x))
    assert x.shape == (ND + MD,)
    indices = np.argsort(x)
    T = np.concatenate((fst_diffs, -snd_diffs))[indices]
    np.cumsum(T, out=T)
    np.abs(T, out=T)
    np.square(T, out=T)
    a = x[indices]
    t = np.diff(a, append=a[-1])
    assert np.all(t >= 0.0)
    return sqrt(np.dot(T, t)) / 2


def init_slb_worker(cells):
    global _VF_CELLS
    _VF_CELLS = cells


def slb_by_indices(p: tuple[int, int]):
    i, j = p
    return (i, j, slb2_cython(_VF_CELLS[i], _VF_CELLS[j]))


def compute_slb2_distance_matrix(
    intracell_csv_loc: str,
    slb2_dist_csv_loc: str,
    num_processes: int,
    chunksize: int,
    verbose: Optional[bool] = False,
) -> None:
    start = time.time()
    names, cells = zip(
        *(
            (name, np.sort(squareform(cell)))
            for name, cell in cell_iterator_csv(intracell_csv_loc)
        )
    )
    N = len(cells)
    indices = it.combinations(iter(range(N)), 2)

    with Pool(
        processes=num_processes, initializer=init_slb_worker, initargs=(cells,)
    ) as pool:
        slb_dists = pool.imap_unordered(slb_by_indices, indices, chunksize=chunksize)
        with open(slb2_dist_csv_loc, "w", newline="") as outfile:
            csvwriter = csv.writer(outfile)
            stop = time.time()
            print("Init time: " + str(stop - start))
            start = time.time()
            for t in _batched(slb_dists, 2000):
                t = [(names[i], names[j], slb_dist) for i, j, slb_dist in t]
                csvwriter.writerows(t)
    stop = time.time()
    print("GW + File IO time " + str(stop - start))


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


class quantized_icdm:
    n: int
    # 2 dimensional square matrix of side length n.
    icdm: npt.NDArray[np.float64]
    # "distribution" is a dimensional vector of length n,
    # a probability distribution on points of the space
    distribution: npt.NDArray[np.float64]
    ns: int
    # A square sub-matrix of icdm, the distance matrix between sampled points. Of side length ns.
    sub_icdm: npt.NDArray[np.float64]
    # q_indices is a 1-dimensional array of integers of length ns+1. For i,j < ns,
    # icdm[sample_indices[i],sample_indices[j]]==sub_icdm[i,j].
    # sample_indices[ns]==n.
    q_indices: npt.NDArray[np.int_]
    # The quantized distribution; a 1-dimensional array of length ns.
    q_distribution: npt.NDArray[np.float64]
    c_A: float
    c_As: float
    A_s_a_s: npt.NDArray[np.float64]
    # This field is equal to np.dot(np.dot(np.multiply(icdm,icdm),distribution),distribution)

    def __init__(
        self,
        cell_dm: npt.NDArray[np.float64],
        p: npt.NDArray[np.float64],
        num_clusters: int,
    ):
        assert len(cell_dm.shape) == 2
        self.n = cell_dm.shape[0]
        cell_dm_sq = np.multiply(cell_dm, cell_dm)
        self.c_A = np.dot(np.dot(cell_dm_sq, p), p)
        Z = cluster.hierarchy.linkage(squareform(cell_dm), method="centroid")
        clusters = cluster.hierarchy.fcluster(
            Z, num_clusters, criterion="maxclust", depth=0
        )
        actual_num_clusters: int = len(set(clusters))
        self.ns = actual_num_clusters
        indices: npt.NDArray[np.int_] = np.argsort(clusters)
        original_cell_dm = cell_dm
        cell_dm = cell_dm[indices, :][:, indices]
        p = p[indices]
        q: list[float]
        q = []
        clusters = np.sort(clusters)
        for i in range(1, actual_num_clusters + 1):
            permutation = np.nonzero(clusters == i)[0]
            this_cluster = cell_dm[permutation, :][:, permutation]
            medoid = np.argmin(sum(this_cluster))
            new_local_indices = np.argsort(this_cluster[medoid])
            cell_dm[permutation, :] = cell_dm[permutation[new_local_indices], :]
            cell_dm[:, permutation] = cell_dm[:, permutation[new_local_indices]]
            indices[permutation] = indices[permutation[new_local_indices]]
            p[permutation] = p[permutation[new_local_indices]]
            q.append(np.sum(p[permutation]))
        self.icdm = np.asarray(cell_dm, order="C")
        self.distribution = p
        q_arr = np.array(q, dtype=np.float64, order="C")
        self.q_distribution = q_arr
        assert abs(np.sum(q_arr) - 1.0) < 1e-7
        medoids = np.nonzero(np.r_[1, np.diff(clusters)])[0]
        A_s = cell_dm[medoids, :][:, medoids]
        assert np.all(np.equal(original_cell_dm[:, indices][indices, :], cell_dm))
        self.sub_icdm = np.asarray(A_s, order="C")
        self.q_indices = np.asarray(
            np.nonzero(np.r_[1, np.diff(clusters), 1])[0], order="C"
        )
        self.c_As = np.dot(np.multiply(A_s, A_s), q_arr) @ q_arr
        self.A_s_a_s = np.dot(A_s, q_arr)


# def init_worker(cell_name):
#     global _PREPROCESSED_CELLS
#     global _NAMES
#     global _OUT_CSV_ROOT
#     _NAMES = names
#     _OUT_CSV_ROOT = out_csv_root


def quantized_gw(A: quantized_icdm, B: quantized_icdm):
    T_rows, T_cols, T_data = quantized_gw_2(
        A.distribution,
        A.sub_icdm,
        A.q_indices,
        A.q_distribution,
        A.A_s_a_s,
        A.c_As,
        B.distribution,
        B.sub_icdm,
        B.q_indices,
        B.q_distribution,
        B.A_s_a_s,
        B.c_As,
    )
    P = sparse.coo_matrix((T_data, (T_rows, T_cols)), shape=(A.n, B.n)).tocsr()
    gw_loss = A.c_A + B.c_A - 2.0 * frobenius(A.icdm, P.dot(P.dot(B.icdm).T))
    return sqrt(gw_loss) / 2.0


# def block_quantized_gw(
#       list_pair : tuple[
#             list[tuple[int,tuple[str,quantized_icdm]]],
#             list[tuple[int,tuple[str,quantized_icdm]]]
#         ]
# ):
#     gw_list=[]
#     cell_listA, cell_listB = list_pair
#     for i, (Aname, A) in cell_listA:
#         for j, (Bname, B) in cell_listB:
#             if i < j:
#                 gw_list.append((Aname,Bname,quantized_gw(A,B)))
#     return gw_list


# def quantized_gw_parallel(
#         intracell_csv_loc : str,
#         num_processes : int,
#         chunk_size : int,
#         num_clusters : int,
#         out_csv : str,
#         verbose : bool = False
# ):
#     cell_iterator=cell_iterator_csv(intracell_csv_loc)
#     quantized_cells=((name,quantized_icdm(
#         cell_dm,
#         np.ones((cell_dm.shape[0],))/cell_dm.shape[0],
#         num_clusters)) for name,cell_dm in cell_iterator)
#     cells_batched = _batched(enumerate(quantized_cells),chunk_size)
#     cell_batch_pairs= it.combinations_with_replacement(cells_batched,2)
#     k=0
#     with Pool(processes=num_processes) as pool:
#         gw_dists=pool.imap_unordered(block_quantized_gw, cell_batch_pairs)
#         with open(out_csv,'w',newline='') as outcsvfile:
#             csvwriter=csv.writer(outcsvfile)
#             for block in gw_dists:
#                 print(k)
#                 k+=1
#                 csvwriter.writerows(block)


def block_quantized_gw(indices):
    (i0, i1), (j0, j1) = indices

    gw_list = []
    for i in range(i0, i1):
        A = _QUANTIZED_CELLS[i]
        for j in range(j0, j1):
            if i < j:
                B = _QUANTIZED_CELLS[j]
                gw_list.append((i, j, quantized_gw(A, B)))
    return gw_list


def init_pool(quantized_cells):
    global _QUANTIZED_CELLS
    _QUANTIZED_CELLS = quantized_cells


# def quantized_gw_parallel(
#         intracell_csv_loc : str,
#         num_processes : int,
#         chunk_size : int,
#         num_clusters : int,
#         out_csv : str,
#         verbose : bool = False
# ):
#     names, cell_dms = zip(*cell_iterator_csv(intracell_csv_loc))
#     quantized_cells=\
#         [ quantized_icdm(cell_dm,
#                          np.ones((cell_dm.shape[0],))/cell_dm.shape[0],
#                          num_clusters) for cell_dm in cell_dms ]
#     N = len(quantized_cells)
#     indices= list(iter(range(0,N,chunk_size)))
#     if indices[-1]!=N:
#         indices.append(N)
#     index_pairs = it.combinations_with_replacement(it.pairwise(indices),2)
#     gw_time = 0.0
#     fileio_time = 0.0
#     gw_start=time.time()
#     with Pool(
#             initializer=init_pool,
#             initargs = (quantized_cells,),
#             processes=num_processes) as pool:
#         gw_dists=pool.imap_unordered(block_quantized_gw, index_pairs)
#         gw_stop=time.time()
#         gw_time+=gw_stop-gw_start
#         with open(out_csv,'w',newline='') as outcsvfile:
#             csvwriter=csv.writer(outcsvfile)
#             gw_start=time.time()
#             for block in gw_dists:
#                 block = [ (names[i],names[j],gw_dist) for (i,j,gw_dist) in block]
#                 gw_stop=time.time()
#                 gw_time+=gw_stop-gw_start
#                 csvwriter.writerows(block)
#                 gw_start=time.time()
#                 fileio_time+=(gw_start-gw_stop)
#     print("GW time: "+str(gw_time))
#     print("File IO time: "+str(fileio_time))


def quantized_gw_index(p: tuple[int, int]):
    i, j = p
    return (i, j, quantized_gw(_QUANTIZED_CELLS[i], _QUANTIZED_CELLS[j]))


def quantized_gw_parallel(
    intracell_csv_loc: str,
    num_processes: int,
    chunksize: int,
    num_clusters: int,
    out_csv: str,
    verbose: bool = False,
):
    names, cell_dms = zip(*cell_iterator_csv(intracell_csv_loc))
    quantized_cells = [
        quantized_icdm(
            cell_dm, np.ones((cell_dm.shape[0],)) / cell_dm.shape[0], num_clusters
        )
        for cell_dm in cell_dms
    ]
    N = len(quantized_cells)
    index_pairs = it.combinations(iter(range(N)), 2)

    gw_time = 0.0
    fileio_time = 0.0
    gw_start = time.time()
    with Pool(
        initializer=init_pool, initargs=(quantized_cells,), processes=num_processes
    ) as pool:
        gw_dists = pool.imap_unordered(
            quantized_gw_index, index_pairs, chunksize=chunksize
        )
        gw_stop = time.time()
        gw_time += gw_stop - gw_start
        with open(out_csv, "w", newline="") as outcsvfile:
            csvwriter = csv.writer(outcsvfile)
            gw_start = time.time()
            t = _batched(gw_dists, 2000)
            for block in t:
                block = [(names[i], names[j], gw_dist) for (i, j, gw_dist) in block]
                gw_stop = time.time()
                gw_time += gw_stop - gw_start
                csvwriter.writerows(block)
                gw_start = time.time()
                fileio_time += gw_start - gw_stop
    print("GW time: " + str(gw_time))
    print("File IO time: " + str(fileio_time))


def init_slb2_pool(sorted_cells):
    global _SORTED_CELLS
    _SORTED_CELLS = sorted_cells


def global_slb2_pool(p: tuple[int, int]):
    i, j = p
    return (i, j, slb2(_SORTED_CELLS[i], _SORTED_CELLS[j]))


# def combined_slb2_quantized_gw(
#     intracell_csv_loc: str,
#     num_processes: int,
#     chunksize: int,
#     num_clusters: int,
#     out_csv: str,
#     confidence_parameter: float,
#     verbose: bool = False,
# ):
#     # First we compute the slb2 intracell distance for everything.
#     names, cell_dms = zip(*cell_iterator_csv(intracell_csv_loc))
#     N = len(names)
#     cell_dms_sorted = [np.sort(squareform(cell, force="tovector")) for cell in cell_dms]
#     with Pool(
#         initializer=init_slb2_pool, initargs=(cell_dms_sorted,), processes=num_processes
#     ) as pool:
#         slb2_dists = pool.imap_unordered(
#             global_slb2, it.combinations(iter(range(N)), 2), chunksize=chunksize
#         )
