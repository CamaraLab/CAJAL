from cajal.utilities import (
    read_gw_dists,
    dist_mat_of_dict,
    read_gw_couplings,
    avg_shape,
    avg_shape_spt,
    leiden_clustering,
    louvain_clustering,
)
import numpy as np


def test():
    cell_names, gw_dist_dictionary = read_gw_dists("tests/gw.csv", True)
    assert isinstance(cell_names, list)
    assert isinstance(gw_dist_dictionary, dict)
    dist_mat_of_dict(gw_dist_dictionary, [cell_names[3], cell_names[2], cell_names[1]])
    gmat = dist_mat_of_dict(gw_dist_dictionary, cell_names)
    assert np.all(gmat >= 0)
    coupling_mats = read_gw_couplings("tests/gw_coupling_mat.npz")
    assert isinstance(coupling_mats, dict)
    icdm = np.loadtxt(
        "tests/icdm.csv", skiprows=1, delimiter=",", usecols=range(1, 1226)
    )
    icdm_dict = {}
    for x, y in zip(cell_names, icdm):
        icdm_dict[x] = y
    dmat_avg_capped, dmat_avg_uncapped = avg_shape(
        cell_names, gw_dist_dictionary, icdm_dict, coupling_mats
    )
    assert np.all(dmat_avg_capped >= 0)
    assert np.all(dmat_avg_uncapped >= 0)

    avg_shape_spt(cell_names, gw_dist_dictionary, icdm_dict, coupling_mats, 3)
    leiden_clustering(gmat)
    louvain_clustering(gmat, 3)
