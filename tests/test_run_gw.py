from cajal.run_gw import compute_gw_distance_matrix
import os


def test():
    compute_gw_distance_matrix(
        intracell_csv_loc="tests/icdm.csv",
        gw_dist_csv_loc="tests/gw1.csv",
        num_processes=2,
        gw_coupling_mat_csv_loc=None,
        return_coupling_mats=False,
        verbose=False,
    )
    os.remove("tests/gw1.csv")
    compute_gw_distance_matrix(
        intracell_csv_loc="tests/icdm.csv",
        gw_dist_csv_loc="tests/gw.csv",
        num_processes=2,
        return_coupling_mats=True,
        gw_coupling_mat_csv_loc="tests/gw_coupling_mat.csv",
        verbose=False,
    )
