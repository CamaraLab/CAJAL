# Functions for visualizing and clustering the GW morphology space
from typing import Optional
import warnings

import numpy as np
import numpy.typing as npt
import matplotlib
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import leidenalg
import community as community_louvain
import igraph as ig
import itertools as it
import networkx as nx
import umap


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
        warnings.warn(
            "Provided plotting functions assume a 2-dimensional UMAP embedding"
        )
    reducer = umap.UMAP(**kwargs, metric="precomputed")
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


def to_colors(ell: list) -> tuple[list[int], list]:
    """
    :param ell: List of elements of some type A, where A has ==.
    :return: (outlist, categories), where categories is the list of unique values of ell, ordered by their first appearance,
    and outlist is a list of integers such that len(outlist)==len(ell) and categories[outlist[i]]==ell[i] for all i in the domain.
    """
    counter: int = 0
    categories = []
    outlist = []
    mydict: dict = {}
    for element in ell:
        if element in mydict:
            outlist.append(mydict[element])
        else:
            mydict[element] = counter
            counter += 1
            outlist.append(mydict[element])
            categories.append(element)
    return outlist, categories


def plot_categorical_data(
    embedding: npt.NDArray[np.float_],
    categories: list,
    size: int,
    names: Optional[list[str]],
    savefile: Optional[str],
) -> None:
    """
    Generate a plot of categorical data with a colorcoded legend.

    :param embedding: array of shape (n,2) containing the data to be plotted.
    :param categories: list of the categories for the data, where the k-th row is
    the category of the k-th data point(x,y). Each category will be plotted
    using a different color. `categories` should contain values from a
    discrete type, e.g., int, string
    :param names: List of names for the data points, they will be superimposed on the drawing.
    :param savefile: filename to save to, i.e., 'myplot.png'
    """
    colors, colorkey = to_colors(categories)
    x_values: list[float] = list(embedding[:, 0])
    y_values: list[float] = list(embedding[:, 1])
    num_pts = len(x_values)
    nested_pts_lists: list[list[tuple[Optional[str], float, float]]] = []
    maybe_names = names if names is not None else it.repeat(None, num_pts)
    for name, x, y, color in zip(maybe_names, x_values, y_values, colors):
        if color == len(nested_pts_lists):
            nested_pts_lists.append([(name, x, y)])
        elif color < len(nested_pts_lists):
            nested_pts_lists[color].append((name, x, y))
        else:
            raise Exception("Wah")

    fig, ax = plt.subplots(figsize=(10, 10), facecolor="white")
    counter = 0
    annotations = []
    for ell in nested_pts_lists:
        name_list = []
        x_list = []
        y_list = []
        for name, x, y in ell:
            annotations.append(plt.annotate(name, (x, y)))
            name_list.append(x)
            x_list.append(x)
            y_list.append(y)
        x_arr = np.array(x_list, dtype=np.float_)
        y_arr = np.array(y_list, dtype=np.float_)
        length = len(x_list)
        color = matplotlib.colormaps["tab20"].colors[counter]
        color_arr = np.tile(color, (length, 1))
        ax.scatter(
            x_arr,
            y_arr,
            s=size,
            c=color_arr,
            label=colorkey[counter],
            edgecolors="none",
        )
        counter += 1
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position(
        [
            box.x0 + box.width * 0.2,
            box.y0 + box.height * 0.2,
            box.width * 0.8,
            box.height * 0.8,
        ]
    )
    # Put a legend to the right of the current axis
    lgd = ax.legend(loc="center right", bbox_to_anchor=(-0.1, 0.5))
    if savefile is not None:
        plt.savefig(
            savefile,
            bbox_extra_artists=[lgd, ax] + annotations,
            bbox_inches="tight",
            pad_inches=0.2,
        )
    else:
        plt.show()


def knn_graph(dmat: npt.NDArray[np.float_], nn: int) -> npt.NDArray[np.int_]:
    """
    :param dmat: squareform distance matrix
    :param nn: (nearest neighbors) - in the returned graph, nodes v and w will be \
    connected if v is one of the `nn` nearest neighbors of w, or conversely.
    :return: A (1,0)-valued adjacency matrix for a nearest neighbors graph.
    """
    a = np.argpartition(dmat, nn, axis=0)
    sidelength = dmat.shape[0]
    graph = np.zeros((sidelength, sidelength), dtype=np.int_)
    graph[a[1 : (nn + 1), :]] = 1
    graph = np.max(graph, graph.T)
    return graph


def louvain_clustering(
    gw_mat: npt.NDArray[np.float_], k: int = 5
) -> npt.NDArray[np.int_]:
    """
    Compute clustering of cells based on GW distance, using Louvain clustering on a KNN graph

    Args:
        gw_mat (numpy array): NxN distance matrix of GW distance between cells
        nn (integer): number of neighbors in KNN graph

    Returns:
        numpy array of shape (num_cells,) the cluster assignment for each cell
    """
    nn = NearestNeighbors(n_neighbors=k, metric="precomputed")
    nn.fit(gw_mat)
    adj_mat = nn.kneighbors_graph(gw_mat).todense()
    np.fill_diagonal(adj_mat, 0)

    graph = nx.convert_matrix.from_numpy_matrix(adj_mat)
    # louvain_clus_dict is a dictionary whose keys are nodes of `graph` and whose
    # values are natural numbers indicating communities.
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
    graph.es["weight"] = adj_mat[adj_mat.nonzero()]
    graph.vs["label"] = range(adj_mat.shape[0])

    if resolution is None:
        leiden_clus = np.array(
            leidenalg.find_partition_multiplex(
                [graph], leidenalg.ModularityVertexPartition
            )[0]
        )
    else:
        leiden_clus = np.array(
            leidenalg.find_partition_multiplex(
                [graph], leidenalg.CPMVertexPartition, resolution_parameter=resolution
            )[0]
        )
    return leiden_clus
