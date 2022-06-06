import unittest
import numpy as np
import pandas as pd
import os
import random
from CAJAL.lib import sample_seg
from CAJAL.lib.run_gw import pj


class TestExamplesClass(unittest.TestCase):

    def test_sample_image(self):
        # don't error
        infolder = "../data/tif_files/"
        outfolder = "../data/test_data/test_tiff_50"
        sample_seg.save_boundaries_tiff(os.listdir(infolder)[0], infolder, outfolder, n_sample=50, background=0)

    def test_sample_folder(self):
        # don't error
        infolder = "../data/tif_files/"
        outfolder = "../data/test_data/test_tiff_50"
        sample_seg.save_boundaries_all(infolder, outfolder, n_sample=50, background=0)


if __name__ == '__main__':
    unittest.main()
