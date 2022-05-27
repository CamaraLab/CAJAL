import unittest
from scipy.spatial.distance import squareform
import numpy as np
import os
import time

from CAJAL.lib import run_gw, calculate_gw_pairwise as cgp


class TestExamplesClass(unittest.TestCase):
    
    def test_identity(self):
        # distance from cell to itself should be close to 0
        data_dir = "../data/example_sampled_50/"
        file_name = os.listdir(data_dir)[0]
        dist_list = [run_gw.get_distances_one(run_gw.pj(data_dir, file_name)),
                     run_gw.get_distances_one(run_gw.pj(data_dir, file_name))]
        gw_dist = run_gw.distance_matrix_preload_global(dist_list)
        self.assertEqual(len(gw_dist), 1)
        self.assertLess(gw_dist[0], 1e-5)
        distances_dir = "../data/example_geodesic_50/"
        file_name = os.listdir(distances_dir)[0]
        dist_list = [run_gw.read_mp_array(squareform(np.loadtxt(run_gw.pj(distances_dir, file_name)))),
                     run_gw.read_mp_array(squareform(np.loadtxt(run_gw.pj(distances_dir, file_name))))]
        gw_dist = run_gw.distance_matrix_preload_global(dist_list)
        self.assertEqual(len(gw_dist), 1)
        self.assertLess(gw_dist[0], 1e-5)
        
    def test_small_example_euclidean(self):
        # distance matrix should be close enough to saved distance matrix
        start = time.time()
        cgp.run_euclidean_example("test_example_euclidean")
        time_elapsed = time.time() - start
        ex_dist_mat = squareform(np.loadtxt("../data/gw_results/example_euclidean_gw_dist_mat.txt"))
        test_dist_mat = squareform(np.loadtxt("../data/gw_results/test_example_euclidean_gw_dist_mat.txt"))
        self.assertEqual(np.allclose(ex_dist_mat, test_dist_mat), True)
        self.assertLess(time_elapsed, 60)  # will depend on machine, ensure timing doesn't significantly change

    def test_small_example_geodesic(self):
        start = time.time()
        cgp.run_geodesic_example("test_example_geodesic")
        time_elapsed = time.time() - start
        ex_dist_mat = squareform(np.loadtxt("../data/gw_results/example_geodesic_gw_dist_mat.txt"))
        test_dist_mat = squareform(np.loadtxt("../data/gw_results/test_example_geodesic_gw_dist_mat.txt"))
        self.assertEqual(np.allclose(ex_dist_mat, test_dist_mat), True)
        self.assertLess(time_elapsed, 60)  # will depend on machine, ensure timing doesn't significantly change


if __name__ == '__main__':
    unittest.main()
