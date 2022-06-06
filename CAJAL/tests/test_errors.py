import unittest
from CAJAL.lib import run_gw


class TestErrorsClass(unittest.TestCase):
    
    def test_empty_folder(self):
        # if given empty folder for data or distance, should return empty list
        dist_mat_list = run_gw.get_distances_all(data_dir="../data/test_data/error_tests/empty_folder")
        self.assertEqual(run_gw.distance_matrix_preload_global(dist_mat_list), [])
        
    def test_diff_numpts(self):
        # if data or distance files have different number of points, should error nicely
        with self.assertRaises(Exception) as context:
            run_gw.get_distances_all(data_dir="../data/test_data/error_tests/bad_files")
        self.assertTrue("Point cloud data files do not have same number of points" in str(context.exception))
        
    def test_filetype(self):
        # make sure behaves how we expect on various input file formats
        # particularly stop treating first line as header
        pass
        
    def test_data_as_dist(self):
        # error if given multi-column file as distance
        with self.assertRaises(Exception) as context:
            run_gw.load_distances_global(distances_dir="../data/test_data/error_tests/bad_files")
        self.assertTrue("Distance files must be in vector form with one value per line" in str(context.exception))


if __name__ == '__main__':
    unittest.main()
