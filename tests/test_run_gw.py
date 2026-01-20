# from cajal.run_gw import compute_gw_distance_matrix, compute_gw_distance_matrix_query_target
from src.cajal.run_gw import compute_gw_distance_matrix, compute_gw_distance_matrix_query_target
from src.cajal.utilities import (
    read_gw_dists,
    read_gw_couplings,
    cell_iterator_csv,
    avg_shape_spt,
)
import os


def test_allbyall():
    compute_gw_distance_matrix(
        intracell_csv_loc="tests/icdm.csv",
        gw_dist_csv_loc="tests/gw1.csv",
        num_processes=2,
        gw_coupling_mat_npz_loc=None,
        return_coupling_mats=False,
    )
    os.remove("tests/gw1.csv")
    compute_gw_distance_matrix(
        intracell_csv_loc="tests/icdm.csv",
        gw_dist_csv_loc="tests/gw.csv",
        num_processes=2,
        return_coupling_mats=True,
        gw_coupling_mat_npz_loc="tests/gw_coupling_mat.npz",
    )

    names, gw_dists = read_gw_dists("tests/gw.csv", header=True)
    gw_coupling_mat_dict = read_gw_couplings("tests/gw_coupling_mat.npz")
    cell_icdms = dict(cell_iterator_csv("tests/icdm.csv", as_squareform=False))
    avg_shape_spt(
        names, gw_dists, cell_icdms, gw_coupling_mat_dict=gw_coupling_mat_dict, k=7
    )


def test_query_target():
    intracell_csv_loc="tests/icdm.csv"

    # Test run with saving data
    dists, _ = compute_gw_distance_matrix_query_target(
        query_intracell_csv_loc=intracell_csv_loc,
        target_intracell_csv_loc=intracell_csv_loc,
        gw_dist_csv_loc="tests/gw.csv",
        num_processes=2,
        return_coupling_mats=True,
        gw_coupling_mat_csv_loc="tests/gw_coupling_mat.csv",
        verbose=False,
    )
    os.remove("tests/gw.csv")
    os.remove("tests/gw_coupling_mat.csv")

    # Make sure results are consistent with all-by-all
    dists2, _ = compute_gw_distance_matrix(
        intracell_csv_loc=intracell_csv_loc,
        gw_dist_csv_loc=None,
        num_processes=2,
        return_coupling_mats=True,
        gw_coupling_mat_csv_loc=None,
        verbose=False,
    )

    assert (dists.round(5) == dists2.round(5)).all()
