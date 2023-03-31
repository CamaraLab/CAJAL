from src.cajal.run_gw import compute_gw_distance_matrix
import os


def test():
    compute_gw_distance_matrix("tests/icdm.csv", "tests/gw1.csv", None, True)
    os.remove("tests/gw1.csv")
    compute_gw_distance_matrix(
        "tests/icdm.csv", "tests/gw.csv", "tests/gw_coupling_mat.csv"
    )
