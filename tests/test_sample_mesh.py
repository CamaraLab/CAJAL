from src.cajal.sample_mesh import compute_icdm_all
import os


def test():
    for b in [True, False]:
        for s in ["euclidean", "geodesic"]:
            for m in ["heat", "networkx"]:
                outcsv = "tests/mesh_icdm_" + str(b) + s + m + ".csv"
                compute_icdm_all("tests/obj", outcsv, s, 20, 2, s, m)
                os.remove(outcsv)
