from .utilities import dist_mat_of_dict
from .visualization import knn_graph
from scipy.spatial.distance import squareform
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
import numpy as np
import numpy.typing as npt


def identify_medoid(
    cell_names: list, gw_dist_dict: dict[tuple[str, str], float]
) -> str:
    """
    Identify the medoid cell in cell_names.
    """
    return cell_names[
        np.argmin(squareform(dist_mat_of_dict(gw_dist_dict, cell_names)).sum(axis=0))
    ]


def cap(a: npt.NDArray[np.float_], c: float) -> npt.NDArray[np.float_]:
    """
    Return a copy of `a` where values above `c` in `a` are replaced with `c`.
    """
    a1 = np.copy(a)
    a1[a1 >= c] = c
    return a1


def orient(
    medoid: str,
    obj_name: str,
    iodm: npt.NDArray[np.float_],
    gw_coupling_mat_dict: dict[tuple[str, str], coo_matrix],
) -> npt.NDArray[np.float_]:
    """
    :param medoid: String naming the medoid object, its key in iodm
    :param obj_name: String naming the object to be compared to
    :param iodm: intra-object distance matrix given in square form
    :param gw_coupling_mat_dict: maps pairs (objA_name, objB_name) to scipy COO matrices
    :return: "oriented" squareform distance matrix
    """
    if obj_name < medoid:
        gw_coupling_mat = gw_coupling_mat_dict[(obj_name, medoid)]
    else:
        gw_coupling_mat = coo_matrix.transpose(gw_coupling_mat_dict[(medoid, obj_name)])
    i_reorder = np.asarray(np.argmax(gw_coupling_mat, axis=0))[0]
    return iodm[i_reorder][:, i_reorder]


def get_avg_shape_spt(
    obj_names: list[str],
    gw_dist_dict: dict[tuple[str, str], float],
    iodms: dict[str, npt.NDArray[np.float_]],
    gw_coupling_mat_dict: dict[tuple[str, str], coo_matrix],
    k: int,
):
    """
    :param obj_names: Keys for the gw_dist_dict and iodms.
    :gw_dist_dict: Dictionary mapping ordered pairs (cellA_name, cellB_name) \
    to Gromov-Wasserstein distances.
    :param iodms: (intra-object distance matrices) - \
    Maps object names to intra-object distance matrices. Matrices are assumed to be given \
    in vector form rather than squareform.
    :gw_coupling_mat_dict: Dictionary mapping ordered pairs (cellA_name, cellB_name) to
    Gromov-Wasserstein coupling matrices from cellA to cellB.
    :param k: how many neighbors in the nearest-neighbors graph.
    """
    num_objects = len(obj_names)
    medoid = identify_medoid(obj_names, gw_dist_dict)
    medoid_matrix = iodms[medoid]
    # Rescale to unit step size.
    medoid_matrix = medoid_matrix / np.min(medoid_matrix)
    square_medoid_matrix = squareform(medoid_matrix, force="tomatrix")
    dmat_accumulator_uncapped = np.copy(medoid_matrix)
    dmat_accumulator_capped = cap(medoid_matrix, 2.0)
    others = (obj for obj in obj_names if obj != medoid)
    for obj_name in others:
        iodm = iodms[obj_name]
        # Rescale to unit step size.
        iodm = iodm / np.min(iodm)
        reoriented_iodm = squareform(
            orient(
                medoid,
                obj_name,
                squareform(iodm, force="tomatrix"),
                gw_coupling_mat_dict,
            ),
            force="tovector",
        )
        # reoriented_iodm is not a distance matrix - it is a "pseudodistance matrix".
        # If X and Y are sets and Y is a metric space, and f : X -> Y, then \
        # d_X(x0, x1) := d_Y(f(x0),f(x1)) is a pseudometric on X.
        dmat_accumulator_uncapped += reoriented_iodm
        dmat_accumulator_capped += cap(reoriented_iodm, 2.0)
    # dmat_avg_uncapped can have any positive values, but none are zero,
    # because medoid_matrix is not zero anywhere.
    dmat_avg_uncapped = dmat_accumulator_uncapped / num_objects
    # dmat_avg_capped has values between 0 and 2, exclusive.
    dmat_avg_capped = dmat_accumulator_capped / num_objects
    dmat_avg_uncapped = squareform(dmat_avg_uncapped)
    # So that 0s along diagonal don't get caught in min
    np.fill_diagonal(dmat_avg_uncapped, np.max(dmat_avg_uncapped))
    # When confidence at a node in the average graph is high, the node is not
    # very close to its nearest neighbor.  We can think of this as saying that
    # this node in the averaged graph is a kind of poorly amalgamated blend of
    # different features in different graphs.  Conversely, when confidence is
    # low, and the node is close to its nearest neighbor, we interpret this as
    # meaning that this node and its nearest neighbor appear together in many
    # of the graphs being averaged, so this is potentially a good
    # representation of some edge that really appears in many of the graphs.
    confidence = np.min(dmat_avg_uncapped, axis=0)
    d = squareform(dmat_avg_capped)
    G = knn_graph(d, k)
    d = np.multiply(d, G)
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
    retmat = squareform(dmat_avg_capped)
    retmat[mask] = 0
    return retmat, confidence
