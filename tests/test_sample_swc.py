# from src.cajal.sample_swc import compute_icdm_all_euclidean, compute_icdm_all_geodesic
from cajal.swc import get_filenames, default_name_validate
from cajal.sample_swc import (
    read_preprocess_compute_geodesic,
    read_preprocess_compute_euclidean,
)
from cajal.utilities import Err


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
