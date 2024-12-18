from cajal.sample_seg import cell_boundaries, compute_icdm_all
import tifffile
import os


def test():
    cell_boundaries(
        tifffile.imread("CAJAL/data/tiff_images_cleaned/epd210cmdil3_5.tif"),
        30,
        0,
        True,
        False,
    )
    compute_icdm_all(
        "CAJAL/data/tiff_images_cleaned", "CAJAL/data/tiff_sampling.csv", 50
    )
    os.remove("CAJAL/data/tiff_sampling.csv")
