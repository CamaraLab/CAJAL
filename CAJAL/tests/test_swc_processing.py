import unittest
import os
from CAJAL.lib import swc_processing as sp
from CAJAL.lib.run_gw import pj


class TestExamplesClass(unittest.TestCase):

    def test_all_reformat(self):
        infolder = "../data/swc_files/"
        outfolder = "../data/test_data"
        file_name = os.listdir(infolder)[0]
        sp.reformat_swc_file(pj(infolder, file_name), pj(outfolder, file_name), new_index=1, sequential=True,
                             rmdisconnect=True, dummy_root=True, keep_header=True)

    def test_wrong_type(self):
        infolder = "../data/obj_files/"
        outfolder = "../data/test_data"
        file_name = os.listdir(infolder)[0]
        sp.reformat_swc_file(pj(infolder, file_name), pj(outfolder, file_name), new_index=1, sequential=True,
                             rmdisconnect=True, dummy_root=True, keep_header=True)


if __name__ == '__main__':
    unittest.main()