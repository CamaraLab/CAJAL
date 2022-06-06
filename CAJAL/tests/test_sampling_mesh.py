import unittest
from CAJAL.lib import sample_mesh


class TestExamplesClass(unittest.TestCase):

    def test_save_all(self):
        # don't error
        infolder = "../data/obj_files"
        outfolder = "../data/test_data/test_obj_50"
        sample_mesh.save_sample_from_obj_parallel(infolder, outfolder, 50, num_cores=8)

    def test_geodesic_all_heat(self):
        # don't error
        infolder = "../data/obj_files"
        outfolder = "../data/test_data/test_obj_geodesic_heat_50"
        sample_mesh.save_geodesic_from_obj_parallel(infolder, outfolder, 50, method="heat", connect=False, num_cores=8)

    def test_geodesic_all_networkx(self):
        # 500 sec
        # don't error
        infolder = "../data/obj_files"
        outfolder = "../data/test_data/test_obj_geodesic_networkx_50"
        sample_mesh.save_geodesic_from_obj_parallel(infolder, outfolder, 50, method="networkx", connect=True, num_cores=8)


if __name__ == '__main__':
    unittest.main()
