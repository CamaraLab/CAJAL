import unittest
import os
import numpy as np
from CAJAL.lib import average_shape_swc as avg_shape
from CAJAL.lib.utilities import pj, load_dist_mat, list_sort_files


class TestErrorsClass(unittest.TestCase):

    def test_get_spt(self):
        # don't error
        data_dir = "../data/sampled_pts/example_geodesic_50"
        file_name = os.listdir(data_dir)[0]
        spt = avg_shape.get_spt(load_dist_mat(pj(data_dir, file_name)))

    def test_avg_spt(self):
        data_dir = "../data/sampled_pts/example_geodesic_50"
        data_prefix = "a10_full"
        files_list = list_sort_files(data_dir, data_prefix)
        match_list = np.load("../data/gw_results/a10_full_geodesic_gw_matching.npz")
        gw_dist = load_dist_mat("../data/gw_results/a10_full_geodesic_gw_dist_mat.txt")
        cluster_ids = [1]
        clusters = [1]*len(files_list)
        avg_shape.get_avg_shape_spt(cluster_ids, clusters, data_dir, files_list, match_list, gw_dist, k=3)


if __name__ == '__main__':
    unittest.main()