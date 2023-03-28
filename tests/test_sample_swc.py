from src.cajal.sample_swc import compute_icdm_all_euclidean, compute_icdm_all_geodesic


def test_compute_geodesic():
    swc_dir = "CAJAL/data/swc"
    compute_icdm_all_euclidean(
        infolder=swc_dir,
        out_csv="CAJAL/data/icdm_euclidean.csv",
        n_sample=50,
        num_cores=10,
    )
    compute_icdm_all_geodesic(
        infolder=swc_dir,
        out_csv="CAJAL/data/icdm_geodesic.csv",
        n_sample=50,
        num_cores=10,
    )
