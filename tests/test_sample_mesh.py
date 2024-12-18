from cajal.sample_mesh import (
    compute_icdm_all,
    _connect_helper,
    cell_generator,
    get_geodesic,
    sample_vertices,
)
import os


def test():
    for b in [True, False]:
        for s in ["euclidean", "geodesic"]:
            outcsv = "tests/mesh_icdm_" + str(b) + s + "heat" + ".csv"
            compute_icdm_all("tests/obj", outcsv, s, 20, 2, s, "heat")
            os.remove(outcsv)

    for m in ["heat", "networkx"]:
        for t in [True, False]:
            cell_gen = map(_connect_helper, cell_generator("tests/obj", t))
            name, vertices, faces = next(cell_gen)
            sample_vertices(vertices, 20)
            get_geodesic(vertices, faces, 20, m)
