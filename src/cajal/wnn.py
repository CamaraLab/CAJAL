import numpy as np
import numpy.typing as npt
from sklearn.manifold import Isomap
from skdim.id import MADA
import anndata as ad

from .run_gw import DistanceMatrix
from .pyWNN import *


def wnn(gw_dmat: DistanceMatrix,
        features: npt.NDArray[np.float_],
        # feature_dmat
        n_neighbors=20,
        # gw_dimension=None
        ):
    """
    :param gw_dmat: A Gromov-Wasserstein distance matrix, `n x n`.
    :param features: An `n x m` matrix, where each row is a feature vector for
        one of the cells.
    """

    # Embed the Gromov-Wasserstein distance matrix into Euclidean
    # space via the Isomap algorithm.
    intrinsic_dim: int = int(MADA(DM=True).fit_transform(np.copy(gw_dmat)))
    gw_embedding=Isomap(n_neighbors=n_neighbors, n_components=intrinsic_dim,
                        metric='precomputed').fit_transform(np.copy(gw_dmat))
    adata = ad.AnnData(shape=gw_embedding.shape)
    adata.obsm['gw'] = gw_embedding
    adata.obsm['features'] = features
    adata = pyWNN(adata, reps=['gw', 'features'], n_neighbors = n_neighbors, npcs=[intrinsic_dim, features.shape[1]]).compute_wnn(adata)
    return adata
    # adata = WNNobj.compute_wnn(adata)
