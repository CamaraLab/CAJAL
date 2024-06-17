"""
n-modality version of the weighted nearest neighbors algorithm
"""

from typing import Optional
import itertools as it
import pdb
import numpy as np
import numpy.typing as npt
from sklearn.manifold import Isomap
from skdim.id import MADA
from scipy.spatial.distance import squareform, pdist
from scipy.special import softmax
from .run_gw import DistanceMatrix


class Modality:
    def local_bandwidth(self, margin_count: int = 20):
        n_obsv = self.nn_index_arr.shape[0]
        jaccard_counts = np.zeros(shape=self.nn_index_arr.shape, dtype=int)
        for i, j in it.combinations(range(0, n_obsv), 2):
            jaccard_counts[i, j] = np.intersect1d(
                self.nn_index_arr[i], self.nn_index_arr[j]
            ).shape[0]
        jaccard_counts = np.maximum(jaccard_counts, jaccard_counts.T)
        local_bandwidths = []
        for i in range(n_obsv):
            jaccard_argsort = np.argsort(jaccard_counts[i])
            # jaccard_argsort_filtered
            jaccard_argsort_filtered = jaccard_argsort[
                np.nonzero(np.sort(jaccard_counts[i]))[0]
            ]
            if jaccard_argsort_filtered.shape[0] > margin_count:
                jaccard_argsort_filtered = jaccard_argsort_filtered[:margin_count]
            marginal_obsvns = self.obsv[jaccard_argsort_filtered]
            avg_distance = (
                np.linalg.norm(self.obsv[i] - marginal_obsvns, axis=1).sum()
                / jaccard_argsort_filtered.shape[0]
            )
            local_bandwidths.append(avg_distance)
        return np.array(local_bandwidths)

    def __init__(self, obsv: npt.NDArray[np.float64]):
        """
        The primary constructor for the Modality class is used when the user has direct access to
        a sequence of observations in n dimensional space.

        :param obsv: A `k` row by `n` column matrix, where `k` is the number of observations,
            and `n` is the dimensionality of the space from which the observations are taken.
        """
        self.obsv = obsv
        self.dmat: DistanceMatrix = squareform(pdist(obsv))
        self.nn_index_arr = np.argsort(self.dmat, axis=1)
        self.bandwidth = self.local_bandwidth()
        self.nn_distance = self.dmat[np.arange(obsv.shape[0]), self.nn_index_arr[:, 1]]
        self.dim = obsv.shape[1]

    @staticmethod
    def of_dmat(
        dmat: DistanceMatrix, intrinsic_dim: Optional[int] = None, n_neighbors=20
    ):
        m = Modality(np.array([[0, 1], [1, 0]]))
        if intrinsic_dim is None:
            intrinsic_dim: int = int(MADA(DM=True).fit_transform(np.copy(dmat)))
        embedding = Isomap(
            n_neighbors=n_neighbors, n_components=intrinsic_dim, metric="precomputed"
        ).fit_transform(np.copy(dmat))
        m.dim = intrinsic_dim
        m.obsv = embedding
        m.dmat = dmat
        m.nn_index_arr = np.argsort(dmat, axis=1)
        m.bandwidth = m.local_bandwidth()
        m.nn_distance = m.dmat[np.arange(embedding.shape[0]), m.nn_index_arr[:, 1]]
        return m


def wnn(modalities: list[Modality], k: int, epsilon: float = 1e-4):
    """
    Compute the weighted nearest neighbors pairing, following
    [Integrated analysis of multimodal single-cell data]\
    (https://www.sciencedirect.com/science/article/pii/S0092867421005833.)

    This algorithm differs from the published algorithm in the paper in a few ways.
    In particular we do not take the L2 normalization of columns of the matrix before we begin.

    :param modalities: list of modalities
    :param k: how many nearest neighbors to consider
    """
    if k <= 0:
        raise Exception("k should be a positive integer.")
    if not modalities:
        raise Exception("Empty list.")
    n_obsvs: int = modalities[0].obsv.shape[0]
    for modality in modalities[1:]:
        if modality.obsv.shape[0] != n_obsvs:
            raise Exception(
                "Must be a consistent count of exceptions across all modalities."
            )

    # Notation follows the paper when appropriate.
    #
    M = len(modalities)

    def pairwise_predictions(m: Modality, n: Modality):
        """
        We measure the ability of modality m to predict modality n.
        """
        prediction_vectors = n.obsv[m.nn_index_arr[:, 1 : (k + 1)], :].sum(axis=1) / k
        assert prediction_vectors.shape == n.obsv.shape
        return prediction_vectors

    def cross_modality_affinities(
        m: Modality, prediction_vectors: npt.NDArray[np.float64]
    ):
        """
        :param m: "target" modality (to be predicted)
        :param prediction_vectors: `n_obsvs` rows, `k_n` columns,
            where `k_n` is the dimension of Euclidean space for the target modality `m`.
            The vectors for feature `m` predicted using the nearest neighbors in `n`.
        """
        prediction_distance = np.linalg.norm(prediction_vectors - m.obsv, axis=1)
        assert prediction_distance.shape == (m.obsv.shape[0],)
        return np.exp(
            -np.maximum(prediction_distance - m.nn_distance, 0)
            / (m.bandwidth - m.nn_distance)
        )

    def theta_m_i_j(m: Modality):
        return np.exp(
            -np.maximum(m.dmat - m.nn_distance[:,np.newaxis],0)
            /(m.bandwidth[:,np.newaxis] - m.nn_distance[:,np.newaxis])
            )

    def all_pairwise_affinities(modalities: list[Modality]):
        pairwise_affinities = np.zeros(shape=(M, M, n_obsvs), dtype=float)
        for i in range(M):
            m = modalities[i]
            for j in range(M):
                n = modalities[j]
                # theta_m,n = pairwise_affinities[i,j]
                # ability of j to predict i, ability of n to predict m
                pairwise_affinities[i, j,:] = cross_modality_affinities(
                    m, pairwise_predictions(n, m)
                )
        return pairwise_affinities

    pairwise_affinities = all_pairwise_affinities(modalities)
    theta = np.stack([theta_m_i_j(m) for m in modalities],axis=0)
    pairwise_affinity_ratios = np.zeros(shape=(M, M - 1, n_obsvs), dtype=float)
    for i in range(M):
        for j in range(M):
            if j >= i:
                k = j - 1
            else:
                k = j
            pairwise_affinity_ratios[i, k, :] = pairwise_affinities[i, i, :] / (
                pairwise_affinities[i, j, :] + epsilon
            )

    similarity_matrix = (softmax(pairwise_affinity_ratios, axis=(0,1)).sum(axis=1)[:,:,np.newaxis] * theta).sum(axis=0)
    similarity_matrix[np.arange(n_obsvs),np.arange(n_obsvs)]=0
    assert np.all(similarity_matrix[similarity_matrix<0] > -1e-10)
    assert np.all((similarity_matrix[similarity_matrix>1]-1) < 1e-10)
    similarity_matrix[similarity_matrix<0]=0
    similarity_matrix[similarity_matrix>1]=1
    return similarity_matrix
