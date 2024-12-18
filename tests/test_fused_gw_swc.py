import os
from cajal.sample_swc import compute_icdm_all_euclidean, compute_icdm_all_geodesic
from cajal.fused_gw_swc import fused_gromov_wasserstein_parallel


def test_fgw_par():
    swc_dir = "tests/swc"
    out_eu_csv = "tests/icdm_euclidean.csv"
    out_eu_nt = "tests/eu_node_types.npy"
    compute_icdm_all_euclidean(
        infolder=swc_dir,
        out_csv=out_eu_csv,
        out_node_types=out_eu_nt,
        n_sample=50,
        num_processes=10,
    )
    compute_icdm_all_geodesic(
        infolder=swc_dir,
        out_csv="tests/icdm_geodesic.csv",
        out_node_types="tests/geo_node_types.npy",
        n_sample=50,
        num_processes=10,
    )
    out_geo_csv = "tests/icdm_geodesic.csv"
    out_geo_nt = "tests/geo_node_types.npy"

    fused_gromov_wasserstein_parallel(
        intracell_csv_loc=out_geo_csv,
        swc_node_types=out_geo_nt,
        fgw_dist_csv_loc="tests/geo_fgw.csv",
        num_processes=2,
        soma_dendrite_penalty=1000.0,
        basal_apical_penalty=1000.0,
    )

    os.remove(out_eu_csv)
    os.remove(out_geo_csv)
    os.remove(out_eu_nt)
    os.remove(out_geo_nt)
