from src.cajal.run_gw import compute_gw_distance_matrix
import os


def test():
    compute_gw_distance_matrix(
        "CAJAL/data/icdm_euclidean.csv", "CAJAL/data/test_gw.csv", None, True
    )
    os.remove("CAJAL/data/test_gw.csv")
    compute_gw_distance_matrix(
        "CAJAL/data/icdm_geodesic.csv",
        "CAJAL/data/test_gw.csv",
        "CAJAL/data/test_gw_coupling_mats.csv",
    )
    os.remove("CAJAL/data/test_gw_coupling_mats.csv")
