from numba import jit, prange
import numpy as np
import numpy.typing as npt

from scipy.cluster.vq import kmeans2

from .run_gw import DistanceMatrix, Distribution, Matrix, gw
from scipy.spatial.distance import squareform, pdist


@jit(nopython=True, parallel=True)
def deformation(
    A: npt.NDArray[np.float64],
    B: npt.NDArray[np.float64],
    coupling_mat: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    :param A: A two-dimensional intracell distance matrix, square of shape (N,N).
    :param B: A two-dimensional intracell distance matrix, square of size (M,M).
    :coupling_mat: A matrix of shape (N,M), a probability distribution; the
        marginal distribution along the axes should represent an appropriate
        probability distribution on points of A and B respectively.
    :return: A pair of square matrices, of shape (N,N) and (M,M) respectively,
        with nonnegative entries, representing the contribution of each of
        these to the Gromov-Wasserstein distance.
    """

    a = np.sum(coupling_mat, axis=1)
    b = np.sum(coupling_mat, axis=0)
    assert a.shape[0] == A.shape[0]
    assert b.shape[0] == B.shape[0]

    A_heatmap = np.zeros((a.shape[0],))
    B_heatmap = np.zeros((b.shape[0],))

    for i in prange(a.shape[0]):
        for ell in prange(b.shape[0]):
            for j in prange(a.shape[0]):
                for k in prange(b.shape[0]):
                    f = (
                        (A[i, j] - B[k, ell]) ** 2
                        * coupling_mat[i, k]
                        * coupling_mat[j, ell]
                    )
                    A_heatmap[i] += f
                    B_heatmap[ell] += f
    return A_heatmap, B_heatmap


def _remove_empty_clusters(
    centroid_cloud: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int_],
):
    """
    Given a vector of labels for points, relabel the points such that empty labels not
    present in the vector are removed. Reindex the points of centroid_cloud so that points
    corresponding to empty labels are deleted.

    :param centroid_cloud: of shape (n,k) - k is the dimensionality of the
        space, n is the number of centroids
    :param labels: of shape (m,), entries range from [0,n).
    :return: A new centroid cloud `new_centroid_cloud` and a new set of labels
        `new_labels`, where `new_centroid_cloud` is of shape `(p,k)` (for p <= n) and
        `new_labels` is of shape `(n,)`; it contains every integer in [0,p) at least once.

    """
    nonempty_clusters = set(labels)
    if len(nonempty_clusters) == centroid_cloud.shape[0]:
        return centroid_cloud, labels
    assert len(nonempty_clusters) < centroid_cloud.shape[0]
    new_indices = np.array(sorted(nonempty_clusters))
    new_centroids = centroid_cloud[new_indices, :]
    new_labels = np.searchsorted(new_indices, labels)
    return new_centroids, new_labels


def distribution_of_clustering(
    labels: npt.NDArray[np.int_],
) -> npt.NDArray[np.float64]:
    """
    :param labels: An array of integer labels for classes/clusters.
    :return: A probability distribution on the clusters which assigns to each cluster
        the number of points in that cluster divided by the total number of points.
    """

    counts = []
    for i in range(max(labels) + 1):
        counts.append(np.count_nonzero(labels == i))
    counts_arr = np.array(counts, dtype=np.float64)
    return counts_arr / np.sum(counts_arr)


def quantized_icdm_lightweight(
    ptcloud: Matrix,
    num_clusters: int,
) -> tuple[DistanceMatrix, Distribution, npt.NDArray[np.int_]]:
    """
    Compute a quantized ICDM for a point cloud.

    :param ptcloud: Of shape `(n,k)`, where `n` is the number of points in the point cloud
    and `n` is the dimension of the ambient space.
    :param num_clusters: How many clusters to group the points of `ptcloud` into.
    :return: A tuple `(A,a,indices)`, where `A` is the intracell distance matrix between
        the centroids of the clusters, `a` is the quantized probability distribution on
        the points of A, and `indices` is a vector of integer indices which maps points
        in the original `ptcloud` to their corresponding cluster in `A`.
        `A` is *not* guaranteed to have side length `num_clusters` but it should be
        reasonably close.
    """
    # Cluster the points of ptcloud into num_clusters many groups,
    # and return the matrix of centroids of each cluster.
    centroids, cluster_labels = kmeans2(ptcloud, num_clusters, minit="++")
    # Purge the empty clusters.
    centroids, cluster_labels = _remove_empty_clusters(centroids, cluster_labels)
    # Form the new probability distribution on the centroids.
    distribution: Distribution = distribution_of_clustering(cluster_labels)
    assert np.all(distribution > 0)
    assert distribution.shape[0] == centroids.shape[0]
    centroids_icdm = squareform(pdist(centroids), force="tomatrix")
    return centroids_icdm, distribution, cluster_labels


def heatmap_of_ptclouds(
    ptcloudA: npt.NDArray[np.float64],
    num_clusters_A: int,
    ptcloudB: npt.NDArray[np.float64],
    num_clusters_B: int,
) -> tuple[
    npt.NDArray[np.int_],
    npt.NDArray[np.float64],
    npt.NDArray[np.int_],
    npt.NDArray[np.float64],
]:
    """
    :param ptcloudA: A point cloud of shape `(n,k)`, where `n` is the number of points in the cloud
    and `k` is the dimension of the underlying space.
    :param num_clusters_A: How many clusters to cluster the point cloud of `ptcloudA` into.
    :param ptcloudB: A point cloud of shape `(m,p)`, where `m` is the number of points in the cloud
    and `p` is the dimension of the underlying space. We need not have `m==n` or `k==p`.
    :param num_clusters_B: How many clusters to cluster the point cloud of `ptcloudB` into.
    :return: a tuple `(A_cluster_labels, A_heatmap, B_cluster_labels, B_heatmap)` where:
        - `A_cluster_labels` is a vector of non-negative integer cluster labels for points of
           `ptcloudA`,
        - `A_heatmap` is a vector of floats measuring the distortion of each cluster
          (indexed by entries in `A_cluster_labels`),
        - `B_cluster_labels` is similar to `A_cluster_labels`,
        - `B_heatmap` is similar to `A_heatmap`
    """
    A_centroids_icdm, A_distribution, A_cluster_labels = quantized_icdm_lightweight(
        ptcloudA, num_clusters_A
    )
    B_centroids_icdm, B_distribution, B_cluster_labels = quantized_icdm_lightweight(
        ptcloudB, num_clusters_B
    )
    coupling_mat, _ = gw(
        A_centroids_icdm, A_distribution, B_centroids_icdm, B_distribution
    )
    A_heatmap, B_heatmap = deformation(A_centroids_icdm, B_centroids_icdm, coupling_mat)
    return A_cluster_labels, A_heatmap, B_cluster_labels, B_heatmap


def navis_heatmap(
    swcA, num_clusters_A: int, swcB, num_clusters_B: int, inplace: bool = False
):
    """
    :param swcA: A SWC Skeleton, a navis object as returned by navis.read_swc()
    :param num_clusters_A: How many clusters to partition the nodes of swcA into
    :param swcB: A SWC Skeleton, a navis object as returned by navis.read_swc()
    :param num_clusters_A: How many clusters to partition the nodes of swcB into
    :param inplace: Whether to return an SWC object with the node data structure
        modified to include the heatmap, or modify the supplied SWC objects in place.
    :return: A tuple `(swcA, swcB)` where the Pandas tables of nodes for
       `swcA` and `swcB` have both had two new columns added, called 'clusters' and 'distortion'.
       'clusters' has nonnegative integer values. 'distortion' is a vector of floats,
       indicating how much distortion each cluster contributes to the overall
       Gromov-Wasserstein cost.
       The 'distortion' column can be used to visualize the distortion associated to
       SWC objects using the navis rendering.
    """

    # Turn the SWC skeleton for A into a point cloud.
    ptcloudA = swcA.nodes[["x", "y", "z"]].to_numpy()
    ptcloudB = swcB.nodes[["x", "y", "z"]].to_numpy()

    A_cluster_labels, A_heatmap, B_cluster_labels, B_heatmap = heatmap_of_ptclouds(
        ptcloudA, num_clusters_A, ptcloudB, num_clusters_B
    )

    if not inplace:
        swcA = swcA.copy()
        swcB = swcB.copy()
    swcA.nodes["clusters"] = A_cluster_labels
    swcB.nodes["clusters"] = B_cluster_labels
    swcA.nodes["distortion"] = A_heatmap[A_cluster_labels]
    swcB.nodes["distortion"] = B_heatmap[B_cluster_labels]
    return (swcA, swcB)
