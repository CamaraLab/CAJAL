import unittest
import numpy as np
import pandas as pd
import os
import random
from CAJAL.lib import sample_swc
from CAJAL.lib.utilities import pj


class TestExamplesClass(unittest.TestCase):

    def test_sample(self):
        # sampled pts on all SWC types should match previous run
        infolder = "../data/swc_files/"
        file_name = os.listdir(infolder)[0]
        sampled_pts, _, _, _ = sample_swc.get_sample_pts(file_name=file_name, infolder=infolder,
                                                         types_keep=(1, 2, 3, 4),
                                                         goal_num_pts=50, min_step_change=1e-7,
                                                         max_iters=50, verbose=False)
        prev_sampled_pts = pd.read_csv(pj("../data/sampled_pts/example_sampled_50",
                                          file_name.replace(".swc", ".csv")), header=None)
        self.assertEqual(np.allclose(sampled_pts, prev_sampled_pts), True)

    def test_sample_bdad(self):
        # sampled pts on just dendrite SWC types should match previous run
        infolder = "../data/swc_files/"
        file_name = os.listdir(infolder)[0]
        sampled_pts, _, _, _ = sample_swc.get_sample_pts(file_name=file_name, infolder=infolder,
                                                         types_keep=(3, 4),
                                                         goal_num_pts=50, min_step_change=1e-7,
                                                         max_iters=50, verbose=False)
        prev_sampled_pts = pd.read_csv(pj("../data/sampled_pts/example_sampled_bdad_50",
                                          file_name.replace(".swc", ".csv")), header=None)
        self.assertEqual(np.allclose(sampled_pts, prev_sampled_pts), True)

    def test_geodesic(self):
        # geodesic network distance on sampled points of all SWC types should match previous run
        infolder = "../data/swc_files/"
        file_name = os.listdir(infolder)[0]
        geo_dist_mat = sample_swc.get_geodesic(file_name=file_name, infolder=infolder,
                                               types_keep=(1, 2, 3, 4),
                                               goal_num_pts=50, min_step_change=1e-7,
                                               max_iters=50, verbose=False)
        prev_geo_dist_mat = np.loadtxt(pj("../data/sampled_pts/example_geodesic_50", file_name.replace(".swc", "_dist.txt")))
        self.assertEqual(np.allclose(geo_dist_mat, prev_geo_dist_mat), True)

    def test_geodesic_bdad(self):
        # geodesic network distance on sampled points of just dendrite SWC types should match previous run
        infolder = "../data/swc_files/"
        file_name = os.listdir(infolder)[0]
        geo_dist_mat = sample_swc.get_geodesic(file_name=file_name, infolder=infolder,
                                               types_keep=(3, 4),
                                               goal_num_pts=50, min_step_change=1e-7,
                                               max_iters=50, verbose=False)
        prev_geo_dist_mat = np.loadtxt(pj("../data/sampled_pts/example_geodesic_bdad_50", file_name.replace(".swc", "_dist.txt")))
        self.assertEqual(np.allclose(geo_dist_mat, prev_geo_dist_mat), True)

    def test_sample_all(self):
        # after saving sampled points for all cells in parallel, a random sample should match previous runs
        infolder = "../data/swc_files"
        outfolder = "../data/test_data/test_sampled_50"
        sample_swc.save_sample_pts_parallel(infolder, outfolder,
                                            types_keep=(1, 2, 3, 4), goal_num_pts=50, min_step_change=1e-7,
                                            max_iters=50, num_cores=8)
        sample_files = random.sample(os.listdir(outfolder), 10)
        for file_name in sample_files:
            prev_sampled_pts = pd.read_csv(pj("../data/sampled_pts/example_sampled_50", file_name), header=None)
            sampled_pts = pd.read_csv(pj(outfolder, file_name), header=None)
            self.assertEqual(np.allclose(sampled_pts, prev_sampled_pts), True)

    def test_geodesic_all(self):
        # after saving geodesic dist of points for all cells in parallel, a random sample should match previous runs
        infolder = "../data/swc_files"
        outfolder = "../data/test_data/test_geodesic_50"
        sample_swc.save_geodesic_parallel(infolder, outfolder,
                                          types_keep=(1, 2, 3, 4), goal_num_pts=50, min_step_change=1e-7,
                                          max_iters=50, num_cores=8)
        sample_files = random.sample(os.listdir(outfolder), 10)
        for file_name in sample_files:
            prev_sampled_pts = pd.read_csv(pj("../data/sampled_pts/example_geodesic_50", file_name), header=None)
            sampled_pts = pd.read_csv(pj(outfolder, file_name), header=None)
            self.assertEqual(np.allclose(sampled_pts, prev_sampled_pts), True)


if __name__ == '__main__':
    unittest.main()
