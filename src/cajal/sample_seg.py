# Functions for sampling points from a 2D segmented image
import os
import warnings
import itertools as it

from typing import List, Iterator, Tuple
import numpy as np
import numpy.typing as npt
from skimage import measure
import tifffile
from scipy.spatial.distance import pdist
from pathos.pools import ProcessPool


from .utilities import write_csv_block


def _filter_to_cells(segmask: npt.NDArray[np.int_], background: int) -> list[int]:
    """
    Return a list of identifiers for cells in the interior of the image.
    """
    cell_ids = set(np.unique(segmask))
    remove_cells = set()
    remove_cells.add(background)
    remove_cells.update(np.unique(segmask[0, :]))
    remove_cells.update(np.unique(segmask[-1, :]))
    remove_cells.update(np.unique(segmask[:, 0]))
    remove_cells.update(np.unique(segmask[:, -1]))
    return list(cell_ids.difference(remove_cells))


def cell_boundaries(
    imarray: npt.NDArray[np.int_],
    n_sample: int,
    background: int = 0,
    discard_cells_with_holes: bool = False,
    only_longest: bool = False,
) -> List[Tuple[int, npt.NDArray[np.float64]]]:
    """
    Sample n coordinates from the boundary of each cell in a segmented image,
    skipping cells that touch the border of the image

    :param imarray: 2D segmented image where the pixels belonging to\
          different cells have different values
    :param n_sample: number of pixel coordinates to sample from boundary of each cell
    :param background: value of background pixels, this will not be saved as a boundary
    :param discard_cells_with_holes: \
          if discard_cells_with_holes is true, we discard any cells \
          with more than one boundary (e.g., an annulus) with a \
          warning. Else, the behavior is determined by only_longest.
    :param only_longest: if discard_cells_with_holes is true, \
          only_longest is irrelevant. Otherwise, this determines whether \
          we sample points from only the longest boundary (presumably \
          the exterior) or from all boundaries, exterior and interior.
    :return:
       list of float numpy arrays of shape (n_sample, 2) \
       containing points sampled from the contours.
    """

    cell_id_list = _filter_to_cells(imarray, background)
    outlist: List[Tuple[int, npt.NDArray[np.float64]]] = []
    for cell in cell_id_list:
        cell_imarray = (imarray == cell) * 1
        boundary_pts_list = measure.find_contours(
            cell_imarray, 0.5, fully_connected="high"
        )
        if discard_cells_with_holes and len(boundary_pts_list) > 1:
            warnings.warn("More than one boundary for cell " + str(cell))
            continue
        boundary_pts: npt.NDArray[np.float64]
        if only_longest:
            boundary_pts_list.sort(key=lambda ell: ell.shape[0])
            boundary_pts = boundary_pts_list[0]
        else:
            boundary_pts = np.concatenate(boundary_pts_list)
        if boundary_pts.shape[0] < n_sample:
            warnings.warn(
                "Fewer than "
                + str(n_sample)
                + " pixels around boundary of cell "
                + str(cell)
            )
        indices = np.linspace(0, boundary_pts.shape[0] - 1, n_sample)
        outlist.append((cell, boundary_pts[indices.astype("uint32")]))
    return list(outlist)


def _compute_intracell_all(
    infolder: str,
    n_sample: int,
    pool: ProcessPool,
    background: int,
    discard_cells_with_holes: bool,
    only_longest: bool,
) -> Iterator[Tuple[str, npt.NDArray[np.float64]]]:
    file_names = [
        file_name
        for file_name in os.listdir(infolder)
        if os.path.splitext(file_name)[1] in [".tif", ".tiff", ".TIF", ".TIFF"]
    ]
    cell_names = [os.path.splitext(file_name)[0] for file_name in file_names]

    # compute_cell_boundaries: Callable[[str], List[Tuple[int, npt.NDArray[np.float64]]]]
    def compute_cell_boundaries(file_name: str):
        return cell_boundaries(
            tifffile.imread(os.path.join(infolder, file_name)),  # type: ignore
            n_sample,
            background,
            discard_cells_with_holes,
            only_longest,
        )

    cell_names_repeat: Iterator[Iterator[str]]
    cell_names_repeat = map(it.repeat, cell_names)
    cell_bdary_lists: Iterator[
        Tuple[Iterator[str], Iterator[Tuple[int, npt.NDArray[np.float64]]]]
    ]
    cell_bdary_lists = zip(
        cell_names_repeat, pool.imap(compute_cell_boundaries, file_names, chunksize=100)
    )
    cell_bdary_list_iters: Iterator[
        Iterator[Tuple[str, Tuple[int, npt.NDArray[np.float64]]]]
    ]
    cell_bdary_list_iters = map(lambda tup: zip(tup[0], tup[1]), cell_bdary_lists)
    cell_bdary_list_flattened: Iterator[Tuple[str, Tuple[int, npt.NDArray[np.float64]]]]
    cell_bdary_list_flattened = it.chain.from_iterable(cell_bdary_list_iters)

    def restructure_and_get_pdist(
        tup: tuple[str, tuple[int, npt.NDArray[np.float64]]]
    ) -> tuple[str, npt.NDArray[np.float64]]:
        name = tup[0] + "_" + str(tup[1][0])
        pd = pdist(tup[1][1])
        return name, pd

    return pool.imap(
        restructure_and_get_pdist, cell_bdary_list_flattened, chunksize=1000
    )


def compute_icdm_all(
    infolder: str,
    out_csv: str,
    n_sample: int,
    num_processes: int = 8,
    background: int = 0,
    discard_cells_with_holes: bool = False,
    only_longest: bool = False,
) -> None:
    """
    Read in each segmented image in a folder (assumed to be .tif), \
    save n pixel coordinates sampled from the boundary
    of each cell in the segmented image, \
    skipping cells that touch the border of the image.

    :param infolder: path to folder containing .tif files.
    :param out_csv: path to csv file to save cell boundaries.
    :param n_sample: number of pixel coordinates \
           to sample from boundary of each cell
    :param discard_cells_with_holes: \
        if discard_cells_with_holes is true, we discard any cells \
        with more than one boundary (e.g., an annulus) with a \
        warning. Else, the behavior is determined by only_longest.
    :param background: value which characterizes the color of the background pixels, \
         this will not be saved as a boundary
    :param only_longest: if discard_cells_with_holes is true, \
         only_longest is irrelevant. Otherwise, this determines whether \
         we sample points from only the longest boundary (presumably \
         the exterior) or from all boundaries, exterior and interior.

    :param num_processes: How many threads to run while sampling.
    :return: None (writes to file)
    """

    pool = ProcessPool(nodes=num_processes)
    name_dist_mat_pairs = _compute_intracell_all(
        infolder, n_sample, pool, background, discard_cells_with_holes, only_longest
    )
    batch_size: int = 1000
    write_csv_block(out_csv, n_sample, name_dist_mat_pairs, batch_size)
    pool.close()
    pool.join()
    pool.clear()
    return None
