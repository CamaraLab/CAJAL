from src.cajal.utilities import (
    read_gw_dists,
    dist_mat_of_dict,
    read_gw_couplings,
    avg_shape_spt,
)
import numpy as np


def test():
    cell_names, gw_dist_dictionary = read_gw_dists("tests/gw.csv", True)
    assert isinstance(cell_names, list)
    assert isinstance(gw_dist_dictionary, dict)
    dist_mat_of_dict(gw_dist_dictionary, [cell_names[3], cell_names[2], cell_names[1]])
    dist_mat_of_dict(gw_dist_dictionary)
    coupling_mats = read_gw_couplings("tests/gw_coupling_mat.csv", True)
    assert isinstance(coupling_mats, dict)
    icdm = np.loadtxt(
        "tests/icdm.csv", skiprows=1, delimiter=",", usecols=range(1, 1226)
    )
    icdm_dict = {}
    for x, y in zip(cell_names, icdm):
        icdm_dict[x] = y
    avg_shape_spt(cell_names, gw_dist_dictionary, icdm_dict, coupling_mats, 3)
