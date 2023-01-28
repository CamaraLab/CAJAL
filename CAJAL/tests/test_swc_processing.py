# import unittest
# import os
# from CAJAL.lib import swc_processing as sp
# from CAJAL.lib.utilities import pj


# class TestExamplesClass(unittest.TestCase):

#     def test_all_reformat(self):
#         infolder = "../data/swc_files/"
#         outfolder = "../data/test_data"
#         file_name = os.listdir(infolder)[0]
#         sp.reformat_swc_file(pj(infolder, file_name), pj(outfolder, file_name), new_index=1, sequential=True,
#                              rmdisconnect=True, dummy_root=True, keep_header=True)

#     def test_wrong_type(self):
#         infolder = "../data/obj_files/"
#         outfolder = "../data/test_data"
#         file_name = os.listdir(infolder)[0]
#         sp.reformat_swc_file(pj(infolder, file_name), pj(outfolder, file_name), new_index=1, sequential=True,
#                              rmdisconnect=True, dummy_root=True, keep_header=True)


# if __name__ == '__main__':
#     unittest.main()

from cajal import sample_swc
import os

def test_read_and_filter():
    swcdir = '../CAJAL/data/swc_files/'
    swc_filelist = os.listdir(swcdir)
    swc_dict_pairs =[]
    for file in swc_filelist:
        swc_dict_pairs.append(sample_swc._read_swc_typed(swcdir+file))
    # swcs_only is a list of SWCForests
    swcs_only = [pair[0] for pair in swc_dict_pairs]
    root_components = []
    for forest in swcs_only:
        first_tree = forest.pop(0)
        assert (first_tree.root.structure_id == 1)
        for tree in forest:
            assert(1 not in sample_swc._node_type_counts(tree))
        root_components.append(first_tree)
    node_type_dicts_before = []
    node_type_dicts_after = []
    test_key_dict = [1,3,4]
    for tree in root_components:
        node_type_dicts_before.append(sample_swc._node_type_counts(tree))
        sample_swc._filter_by_node_type_iterative(tree, test_key_dict)
        node_type_dicts_after.append(sample_swc._node_type_counts(tree))
    for i in range(len(node_type_dicts_before)):
        for k in node_type_dicts_after[i]:
            assert(k in test_key_dict)
            assert(node_type_dicts_after[i][k] == node_type_dicts_before[i][k])


