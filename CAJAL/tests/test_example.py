import unittest
from scipy.spatial.distance import squareform
import numpy as np

from CAJAL.scripts import calculate_gw_pairwise as cgp


class TestExamplesClass(unittest.TestCase):
    
    def test_identity(self):
        # distance from cell to itself should be 0
        pass
        
    def test_small_example_euclidean(self):
        # distance matrix should be close enough to saved distance matrix
        cgp.run_euclidean_example("test_example_euclidean")
        ex_dist_mat = squareform(np.loadtxt("gw_results/example_euclidean_gw_dist_mat.txt"))
        test_dist_mat = squareform(np.loadtxt("gw_results/test_example_euclidean_gw_dist_mat.txt"))
        self.assertEqual(np.allclose(ex_dist_mat, test_dist_mat), True)

    def test_small_example_geodesic(self):
        cgp.run_geodesic_example("test_example_geodesic")
        ex_dist_mat = squareform(np.loadtxt("gw_results/example_geodesic_gw_dist_mat.txt"))
        test_dist_mat = squareform(np.loadtxt("gw_results/test_example_geodesic_gw_dist_mat.txt"))
        self.assertEqual(np.allclose(ex_dist_mat, test_dist_mat), True)


if __name__ == '__main__':
    unittest.main()
