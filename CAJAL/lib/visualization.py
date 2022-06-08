# Functions for visualizing and clustering the GW morphology space
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import leidenalg
import community as community_louvain
import igraph as ig
import networkx as nx
import umap
import warnings


def get_umap(gw_mat, **kwargs):
    """
    Compute UMAP embedding of points using GW distance

    Args:
        gw_mat (numpy array): NxN distance matrix of GW distance between cells
        **kwargs: arguments passed into umap.UMAP

    Returns:
        UMAP embedding
    """
    if "metric" in kwargs:
        warnings.warn("Do not specify a 'metric' argument for UMAP, running without")
        del kwargs["metric"]
    if "n_components" in kwargs and kwargs["n_components"] != 2:
        warnings.warn("Provided plotting functions assume a 2-dimensional UMAP embedding")
    reducer = umap.UMAP(**kwargs,
                        metric="precomputed")
    embedding = reducer.fit_transform(gw_mat)
    return embedding


def plot_umap(embedding, step=1, figsize=10, **kwargs):
    """
    Plot scatter plot of points/cells in 2D UMAP embedding

    Args:
        embedding (numpy array): 2D UMAP embedding of cells
        step (integer): plot only 1/step points, useful for visualizing large datasets
        figsize (integer): useful for Jupyter Notebooks, where the default figsize is too small

    Returns:
        None (plot using matplotlib)
    """
    fig = plt.figure(facecolor="white", figsize=(figsize, figsize))
    return plt.scatter(embedding[::step, 0], embedding[::step, 1], **kwargs)


def louvain_clustering(gw_mat, nn=5):
    """
    Compute clustering of cells based on GW distance, using Louvain clustering on a KNN graph

    Args:
        gw_mat (numpy array): NxN distance matrix of GW distance between cells
        nn (integer): number of neighbors in KNN graph

    Returns:
        numpy array of cluster assignment for each cell
    """
    nn = NearestNeighbors(n_neighbors=nn, metric="precomputed")
    nn.fit(gw_mat)
    adj_mat = nn.kneighbors_graph(gw_mat).todense()
    np.fill_diagonal(adj_mat, 0)

    graph = nx.convert_matrix.from_numpy_matrix(adj_mat)
    louvain_clus_dict = community_louvain.best_partition(graph)
    louvain_clus = np.array([louvain_clus_dict[x] for x in range(gw_mat.shape[0])])
    return louvain_clus


def leiden_clustering(gw_mat, nn=5, resolution=None):
    """
    Compute clustering of cells based on GW distance, using Leiden clustering on a KNN graph

    Args:
        gw_mat (numpy array): NxN distance matrix of GW distance between cells
        nn (integer): number of neighbors in KNN graph
        resolution (float, or None): If None, use modularity to get optimal partition.
            If float, get partition at set resolution.

    Returns:
        numpy array of cluster assignment for each cell
    """
    nn = NearestNeighbors(n_neighbors=nn, metric="precomputed")
    nn.fit(gw_mat)
    adj_mat = nn.kneighbors_graph(gw_mat).todense()
    np.fill_diagonal(adj_mat, 0)

    graph = ig.Graph.Adjacency((adj_mat > 0).tolist())
    graph.es['weight'] = adj_mat[adj_mat.nonzero()]
    graph.vs['label'] = range(adj_mat.shape[0])

    if resolution is None:
        leiden_clus = np.array(leidenalg.find_partition_multiplex([graph], leidenalg.ModularityVertexPartition)[0])
    else:
        leiden_clus = np.array(leidenalg.find_partition_multiplex([graph], leidenalg.CPMVertexPartition,
                                                                  resolution_parameter=resolution)[0])
    return leiden_clus

