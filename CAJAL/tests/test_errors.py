import unittest


class TestErrorsClass(unittest.TestCase):
    
    def test_empty_folder(self):
        # if given empty folder for data or distance, should error nicely
        pass
        
    def test_diff_numpts(self):
        # if data or distance files have different number of points, should error nicely
        pass
        
    def test_filetype(self):
        # make sure behaves how we expect on various input file formats
        # particularly stop treating first line as header
        pass
        
    def test_data_as_dist(self):
        # error if given multi-column file as distance
        pass
        
    def test_pdist_metric(self):
        # error nicely if metric not a pdist metric
        pass


if __name__ == '__main__':
    unittest.main()
