# Functions for sampling points from a 2D segmented image
import numpy as np
import os
from skimage import measure
import tifffile
import warnings

from CAJAL.lib.run_gw import pj


def save_cell_boundaries(imarray, outfile, n_sample, background=0):
    """
    Save n pixel coordinates sampled from the boundary of each cell in a segmented image,
    skipping cells that touch the border of the image

    Args:
        imarray (numpy array): 2D segmented image where the pixels belonging to different cells have different values
        outfile (string): file path to save cell boundaries - output files will be named with _cellN.csv where N is
            the pixel value denoting that cell in the segmented image
        n_sample (integer): number of pixel coordinates to sample from boundary of each cell
        background (integer, float): value of background pixels, this will not be saved as a boundary

    Result:
        None (writes to file)
    """
    cell_ids = set(np.unique(imarray))
    remove_cells = set()
    remove_cells.update(np.unique(imarray[0, :]))
    remove_cells.update(np.unique(imarray[-1, :]))
    remove_cells.update(np.unique(imarray[:, 0]))
    remove_cells.update(np.unique(imarray[:, -1]))
    cell_ids = list(cell_ids.difference(remove_cells))

    for cell in cell_ids:
        if cell == background:  # Don't draw a boundary around background
            continue
        cell_imarray = (imarray == cell) * 1
        boundary_pts = measure.find_contours(cell_imarray, 0.5, fully_connected='high')
        if len(boundary_pts) > 1:
            warnings.warn("More than one boundary for cell " + str(cell))
            continue
        if boundary_pts[0].shape[0] < n_sample:
            warnings.warn("Fewer than " + str(n_sample) + " pixels around boundary of cell " + str(cell))
        np.savetxt(outfile.replace(".csv","") + "_cell"+str(cell)+".csv",
                   boundary_pts[0][np.linspace(0, boundary_pts[0].shape[0]-1, n_sample).astype("uint32")],
                   delimiter=",")


def save_boundaries_tiff(image_file, infolder, outfolder, n_sample, background=0):
    """
    Read in segmented image (assumed to be .tif), save n pixel coordinates sampled from the boundary of each cell
    in a segmented image, skipping cells that touch the border of the image

    Args:
        image_file (string): file name of .tif file
        infolder (string): path to folder containing .tif file
        outfolder (string): path to folder to save cell boundaries - output files will be named with _cellN.csv
            where N is the pixel value denoting that cell in the segmented image
        n_sample (integer): number of pixel coordinates to sample from boundary of each cell
        background (integer, float): value of background pixels, this will not be saved as a boundary

    Result:
        None (writes to file)
    """
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)
    imarray = tifffile.imread(pj(infolder, image_file))
    save_cell_boundaries(imarray, pj(outfolder, image_file[:-4] + ".csv"), n_sample, background)


def save_boundaries_all(infolder, outfolder, n_sample, background=0):
    """
    Read in each segmented image in a folder (assumed to be .tif), save n pixel coordinates sampled from the boundary
    of each cell in the segmented image, skipping cells that touch the border of the image

    Args:
        infolder (string): path to folder containing .tif files
        outfolder (string): path to folder to save cell boundaries - output files will be named with _cellN.csv
            where N is the pixel value denoting that cell in the segmented image
        n_sample (integer): number of pixel coordinates to sample from boundary of each cell
        background (integer, float): value of background pixels, this will not be saved as a boundary

    Result:
        None (writes to file)
    """
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)
    file_names = os.listdir(infolder)
    for image_file in file_names:
        save_boundaries_tiff(image_file, infolder, outfolder, n_sample, background)
