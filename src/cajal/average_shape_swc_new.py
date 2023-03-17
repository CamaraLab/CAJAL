from .utilities import dist_mat_of_dict
from scipy.spatial.distance import squareform

import numpy as np
import numpy.typing as npt


def identify_medoid(
    cell_names: list, gw_dist_dict: dict[tuple[str, str], float]
) -> str:
    """
    Identify the medoid cell in cell_names.
    """


def cap(a: npt.NDArray[np.float_], c: float) -> npt.NDArray[np.float_]:
    """
    All values above c in a are replaced with c.
    """
    a1 = np.copy(a)
    a1[a1 >= c] = c
    return a1


def get_avg_shape_spt(
    obj_names: list,
    gw_dist_dict: dict[tuple[str, str], float],
    iodms: dict[str, npt.NDArray[np.float_]],
    gw_coupling_mat_dict: dict[tuple[str, str], npt.NDArray[np.float_]],
    k: int,
) -> None:
    """
    :param iodms: (intra-object distance matrices) - \
    Maps object names to intra-object distance matrices. Matrices are assumed to be given \
    in vector form.
    :param k: how many neighbors in the nearest-neighbors graph.
    """

    num_objects = len(obj_names)
    medoid = identify_medoid(obj_names, gw_dist_dict)
    medoid_matrix = iodms[medoid]
    # Rescale to unit step size.
    medoid_matrix = medoid_matrix / np.min(medoid_matrix)
    dmat_accumulator_uncapped = np.copy(medoid_matrix)
    dmat_accumulator_capped = cap(medoid_matrix, 2.0)
    others = [obj for obj in obj_names if obj != medoid]
    for obj in (obj for obj in obj_names if obj != medoid):
        iodm = iodms[obj]
        # Rescale to unit step size.
        iodm = iodm / np.min(iodm)
        reoriented_iodm = orient(medoid, obj, iodm, gw_coupling_mat_dict)
        dmat_accumulator_uncapped += reoriented_iodm
        dmat_accumulator_capped += cap(reoriented_iodm, 2.0)
    dmat_avg_uncapped = dmat_accumulator_1 / num_objects
    dmat_avg_capped = dmat_accumulator_2 / num_objects
    dmat_avg_uncapped = squareform(dmat_avg_1)
    # So that 0s don't get caught in min
    dmat_avg_uncapped[dmat_avg_uncapped == 0] = np.max(dmat_avg_uncapped)
    confidence = np.min(dmat_avg_uncapped, axis=0)
    d = squareform(dmat_avg_capped)
    cutoff = np.percentile(d, (k + 1.0) / d.shape[0] * 100, axis=0)  # knn graph
    d[np.greater(d, cutoff)] = 0
    d = np.maximum(d, d.T)
    # Get shortest path tree
    spt = dijkstra(d, directed=False, indices=0, return_predecessors=True)
    # Get graph representation by only keeping distances on edges from spt
    mask = np.array([True] * (d.shape[0] * d.shape[1])).reshape(d.shape)
    for i in range(1, len(spt[1])):
        if spt[1][i] == -9999:
            print("Disconnected", i)
            continue
        mask[i, spt[1][i]] = False
        mask[spt[1][i], i] = False
    d_avg[mask] = 0
    return d_avg, confidence
