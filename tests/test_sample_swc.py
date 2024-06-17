# from src.cajal.sample_swc import compute_icdm_all_euclidean, compute_icdm_all_geodesic
from src.cajal.swc import get_filenames, default_name_validate
from src.cajal.sample_swc import (
    read_preprocess_compute_geodesic,
    read_preprocess_compute_euclidean,
    compute_icdm_all_euclidean,
    compute_icdm_all_geodesic,
)
from src.cajal.utilities import Err
import os


def test_rpcg():
    swc_dir = "tests/swc"
    cell_names, file_paths = get_filenames(swc_dir, default_name_validate)
    t = map(
        lambda file_path: read_preprocess_compute_geodesic(
            file_path, 20, lambda forest: forest[0]
        ),
        file_paths,
    )
    for i, a in enumerate(t):
        if i > 20:
            break
    t = map(
        lambda file_path: read_preprocess_compute_euclidean(
            file_path,
            20,
            lambda forest: forest if len(forest) == 1 else Err("Too many components."),
        ),
        file_paths,
    )
    for i, a in enumerate(t):
        if i > 20:
            break


def test_compute_icdm_both():
    swc_dir = "tests/swc"
    compute_icdm_all_euclidean(
        infolder=swc_dir,
        out_csv="tests/icdm_euclidean.csv",
        n_sample=50,
    )
    compute_icdm_all_geodesic(
        infolder=swc_dir,
        out_csv="tests/icdm_geodesic.csv",
        n_sample=50,
        num_processes=10,
    )
    os.remove("tests/icdm_euclidean.csv")
    os.remove("tests/icdm_geodesic.csv")
