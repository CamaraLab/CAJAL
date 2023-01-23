# Functions for sampling points from a 2D segmented image
import numpy as np
import numpy.typing as npt
import os
from skimage import measure
import tifffile
import warnings
from scipy.spatial.distance import pdist
from typing import List, Iterator, Tuple, Callable
import itertools as it
from pathos.pools import ProcessPool
from tinydb import TinyDB

from cajal.utilities import pj, write_tinydb_block

# def save_cell_boundaries(imarray, outfile, n_sample, background=0):
#     """
#     Save n pixel coordinates sampled from the boundary of each cell in a segmented image,
#     skipping cells that touch the border of the image

#     Args:
#         imarray (numpy array): 2D segmented image where the pixels belonging to \
#         different cells have different values
#         outfile (string): file path to save cell boundaries - output files will be named\
#          with _cellN.csv where N is
#             the pixel value denoting that cell in the segmented image
#         n_sample (integer): number of pixel coordinates to sample from boundary of each cell
#         background (integer, float): value of background pixels, this will not be saved as a boundary

#     Result:
#         None (writes to file)
#     """
#     cell_ids = set(np.unique(imarray))
#     remove_cells = set()
#     remove_cells.update(np.unique(imarray[0, :]))
#     remove_cells.update(np.unique(imarray[-1, :]))
#     remove_cells.update(np.unique(imarray[:, 0]))
#     remove_cells.update(np.unique(imarray[:, -1]))
#     cell_ids = list(cell_ids.difference(remove_cells))
#     for cell in cell_ids:
#         if cell == background:  # Don't draw a boundary around background
#             continue
#         cell_imarray = (imarray == cell) * 1
#         boundary_pts = measure.find_contours(cell_imarray, 0.5, fully_connected='high')
#         if len(boundary_pts) > 1:
#             warnings.warn("More than one boundary for cell " + str(cell))
#             continue
#         if boundary_pts[0].shape[0] < n_sample:
#             warnings.warn("Fewer than " + str(n_sample) + " pixels around boundary of cell " + str(cell))
#         np.savetxt(outfile.replace(".csv","") + "_cell"+str(cell)+".csv",
#                    boundary_pts[0][np.linspace(0, boundary_pts[0].shape[0]-1, n_sample).astype("uint32")],
#                    delimiter=",")

def cell_boundaries(imarray : npt.NDArray[np.int_],
                    n_sample: int,
                    background : int = 0,
                    discard_cells_with_holes : bool = False,
                    only_longest : bool = False
                    ) -> List[Tuple[int,npt.NDArray[np.float_]]]:
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
    
    cell_ids = set(np.unique(imarray))
    remove_cells = set()
    remove_cells.update(np.unique(imarray[0, :]))
    remove_cells.update(np.unique(imarray[-1, :]))
    remove_cells.update(np.unique(imarray[:, 0]))
    remove_cells.update(np.unique(imarray[:, -1]))
    cell_id_list = list(cell_ids.difference(remove_cells))
    
    outlist : List[npt.NDArray[np.float_]] = []
    for cell in cell_id_list:
        if cell == background:  # Don't draw a boundary around background
            continue
        cell_imarray = (imarray == cell) * 1
        boundary_pts_list = measure.find_contours(cell_imarray, 0.5, fully_connected='high')
        if discard_cells_with_holes and len(boundary_pts) > 1:
            warnings.warn("More than one boundary for cell " + str(cell))
            continue
        boundary_pts : npt.NDArray[np.float_]
        if only_longest:
            boundary_pts_list.sort(key = lambda l : l.shape[0])
            boundary_pts = boundary_pts_list[0]
        else:
            boundary_pts = np.concatenate(boundary_pts_list)
        if boundary_pts.shape[0] < n_sample:
            warnings.warn("Fewer than " + str(n_sample) + \
                          " pixels around boundary of cell " + str(cell))
        indices = np.linspace(0, boundary_pts.shape[0]-1, n_sample)
        outlist.append(boundary_pts[indices.astype("uint32")])
    return list(enumerate(outlist))

def _compute_intracell_all(
        infolder : str,
        n_sample: int,
        pool : ProcessPool,
        background : int,
        discard_cells_with_holes : bool,
        only_longest : bool
)-> Iterator[Tuple[str,npt.NDArray[np.float_]]]:
    file_names =\
        [file_name for file_name in os.listdir(infolder)
         if os.path.splitext(file_name)[1]
                  in [".tif", ".tiff", ".TIF", ".TIFF"]]
    cell_names = [os.path.splitext(file_name)[0] for file_name in file_names]
    compute_cell_boundaries : Callable[[str],List[Tuple[int,npt.NDArray[np.float_]]]]
    compute_cell_boundaries = lambda file_name : cell_boundaries(
        tifffile.imread(os.path.join(infolder,file_name)), #type: ignore
        n_sample,
        background,
        discard_cells_with_holes,
        only_longest)
    cell_names_repeat : Iterator[Iterator[str]]
    cell_names_repeat = map(it.repeat, cell_names)
    cell_bdary_lists : Iterator[Tuple[Iterator[str],Iterator[Tuple[int,npt.NDArray[np.float_]]]]]
    cell_bdary_lists = zip(
        cell_names_repeat,
        pool.imap(compute_cell_boundaries,file_names,chunksize=100))
    cell_bdary_list_iters : Iterator[Iterator[Tuple[str,Tuple[int,npt.NDArray[np.float_]]]]]
    cell_bdary_list_iters =\
        map(lambda tup : zip(tup[0],tup[1]), cell_bdary_lists)
    cell_bdary_list_flattened : Iterator[Tuple[str,Tuple[int,npt.NDArray[np.float_]]]]
    cell_bdary_list_flattened = it.chain.from_iterable(cell_bdary_list_iters)

    restructure_and_get_pdist : Callable[[Tuple[str,Tuple[int,npt.NDArray[np.float_]]]],\
                           Tuple[str,npt.NDArray[np.float_]]]
    restructure = lambda tup : (tup[0] + '_' + str(tup[1][0]), pdist(tup[1][1]))
    return pool.imap(restructure, cell_bdary_list_flattened ,chunksize=1000)

def compute_and_save_intracell_all(
        infolder : str,
        db_name: str,
        n_sample: int,
        num_cores: int = 8,
        background: int = 0,
        discard_cells_with_holes : bool = False,
        only_longest : bool = False
        ) -> None:
    """
    Read in each segmented image in a folder (assumed to be .tif), \
    save n pixel coordinates sampled from the boundary
    of each cell in the segmented image, \
    skipping cells that touch the border of the image.

    :param infolder: path to folder containing .tif files.
    :param outfolder: path to folder to save cell boundaries.
    :param n_sample: number of pixel coordinates to sample from boundary of each cell
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

    :param num_cores: How many threads to run while sampling.
    :return: None (writes to file)
    """
    
    output_db = TinyDB(db_name + ".json")
    pool = ProcessPool(nodes=num_cores)
    name_dist_mat_pairs = _compute_intracell_all(
        infolder,
        n_sample,
        pool,
        background,
        discard_cells_with_holes,
        only_longest)
    batch_size : int =1000
    write_tinydb_block(output_db, name_dist_mat_pairs, batch_size)
    pool.close()
    pool.join()
    pool.clear()
    return None
