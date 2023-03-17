"""
Helper functions.
"""
import os
from dataclasses import dataclass
from multiprocessing import RawArray
import csv
from scipy.spatial.distance import squareform
from scipy.sparse import coo_array, transpose
import itertools as it
import math
from typing import Tuple, List, Iterator, Optional, TypeVar, Generic


import numpy as np
import numpy.typing as npt


def pj(*paths):
    return os.path.abspath(os.path.join(*paths))


def read_gw_dists(
    gw_dist_file_loc: str, header: bool
) -> tuple[list[str], dict[tuple[str, str], float]]:
    r"""
    Read a GW distance matrix into memory.
    
    :param gw_dist_file_loc: A file path to a Gromov-Wasserstein distance matrix. \
    The distance matrix should be a CSV file with exactly three columns and possibly \
    a single header line (which is ignored). All \
    following lines should be two strings cell_name1, cell_name2 followed by a \
    floating point real number.

    :param header: If `header` is True, the very first line of the file is discarded. \
    If `header` is False, all lines are assumed to be relevant.
    
    :returns: A pair (cell_names, gw_dist_dictionary), where \
    cell_names is a list of cell names in alphabetical order, gw_dist_dictionary \
    is a dictionary of the GW distances  which can be read like \
    gw_dist_dictionary[(cell_name1, cell_name2)], where cell_name1 and cell_name2 \
    are in alphabetical order. gw_dist_list is a vector-form array (rank 1) of the \
    GW distances.
    """
    gw_dist_dict: dict[tuple[str, str], float] = {}
    with open(gw_dist_file_loc, "r", newline="") as gw_file:
        csvreader = csv.reader(gw_file)
        if header:
            _ = next(csvreader)
        for first_cell, second_cell, gw_dist_str in csvreader:
            gw_dist = float(gw_dist_str)
            first_cell, second_cell = sorted([first_cell, second_cell])
            gw_dist_dict[(first_cell, second_cell)] = gw_dist
    all_cells_set = set()
    for cell_1, cell_2 in gw_dist_dict:
        all_cells_set.add(cell_1)
        all_cells_set.add(cell_2)
    all_cells = sorted(list(all_cells_set))
    return all_cells, gw_dist_dict


def dist_mat_of_dict(
    cell_names: list[str], gw_dist_dictionary: dict[tuple[str, str], int]
) -> npt.NDArray[np.float_]:
    """
    Given a list of cell names and a distance dictionary, return a vectorform distance \
    matrix containing the pairwise GW distances between all cells in `cell_names`, and \
    in the same order. \
    It is assumed that the keys in distance_dictionary are in alphabetical order.
    """
    dist_list: list[float] = []
    for first_cell, second_cell in it.combinations(cell_names, 2):
        first_cell, second_cell = sorted([first_cell, second_cell])
        dist_list.append(gw_dist_dictionary[(first_cell, second_cell)])
    return np.array(dist_list, dtype=np.float_)


def read_gw_couplings(
    gw_couplings_file_loc: str, header: bool
) -> dict[tuple[str, str], float]:
    gw_coupling_mat_dict: dict[tuple[str, str], npt.NDArray[np.float_]] = {}
    with open(gw_couplings_file_loc, "r", newline="") as gw_file:
        csvreader = csv.reader(gw_file)
        linenum = 0
        if header:
            _ = next(csvreader)
            linenum += 1
        for line in csvreader:
            linenum += 1
            cellA_name = line[0]
            cellA_sidelength = int(line[1])
            cellB_name = line[2]
            cellB_sidelength = int(line[3])
            num_non_zero = int(line[4])
            rest = line[:4]
            if 3 * num_non_zero != len(rest):
                raise Exception(
                    "On line " + str(linenum) + " data not in COO matrix form."
                )
            data = [float(x) for x in rest[:num_non_zero]]
            rows = [int(x) for x in rest[num_non_zero : (2 * num_non_zero)]]
            cols = [int(x) for x in rest[(2 * num_non_zero) :]]
            coo = coo_array(
                (data, (rows, cols)), shape=(cellA_sidelength, cellB_sidelength)
            )
            if cellA_name < cellB_name:
                gw_coupling_mat_dict[(cellA_name, cellB_name)] = coo
            else:
                gw_coupling_mat_dict[(cellB_name, cellA_name)] = transpose(coo)
    return gw_coupling_mat_dict


T = TypeVar("T")


@dataclass
class Err(Generic[T]):
    code: T


def write_csv_block(
    out_csv: str,
    sidelength: int,
    dist_mats: Iterator[Tuple[str, Err[T] | npt.NDArray[np.float_]]],
    batch_size: int,
) -> list[tuple[str, Err[T]]]:
    """
    :param sidelength: The side length of all matrices in dist_mats.
    :param dist_mats: an iterator over pairs (name, arr), where arr is an
    vector-form array (rank 1) or an error code.
    """
    failed_cells: list[tuple[str, Err[T]]] = []
    with open(out_csv, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=",")
        firstline = ["cell_id"] + [
            "d_%d_%d" % (i, j) for i, j in it.combinations(range(sidelength), 2)
        ]
        csvwriter.writerow(firstline)
        while next_batch := list(it.islice(dist_mats, batch_size)):
            good_cells: list[list[str | float]] = []
            for name, cell in next_batch:
                match cell:
                    case Err(_):
                        failed_cells.append((name, cell))
                    case cell:
                        good_cells.append([name] + cell.tolist())
            csvwriter.writerows(good_cells)
    return failed_cells
