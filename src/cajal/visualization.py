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


def plot_umap(
    embedding: npt.NDArray[np.float_], step: int = 1, figsize: int = 10, **kwargs
) -> matplotlib.collections.PathCollection:
    """
    Use matplotlib to create a scatter plot of points/cells in 2D UMAP embedding.

    :param embedding: 2D UMAP embedding of cells
    :param step: plot only 1/step points, useful for visualizing large datasets
    :param figsize: useful for Jupyter Notebooks, where the default figsize is too small
    :return: A matplotlib scatterplot of the data.
    """
    fig = plt.figure(facecolor="white", figsize=(figsize, figsize))
    return plt.scatter(embedding[::step, 0], embedding[::step, 1], **kwargs)


def to_colors(ell: list) -> tuple[list[int], list]:
    """
    :param ell: List of elements of some type A, where A has ==.
    :return: (outlist, categories), where categories is the list of unique values of ell, \
    ordered by their first appearance,
    and outlist is a list of integers such that len(outlist)==len(ell) and \
    categories[outlist[i]]==ell[i] for all i in the domain.
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
    :return: A (1,0)-valued adjacency matrix for a nearest neighbors graph, same shape as dmat.
    """
    a = np.argpartition(dmat, nn + 1, axis=0)
    sidelength = dmat.shape[0]
    graph = np.zeros((sidelength, sidelength), dtype=np.int_)
    for i in range(graph.shape[1]):
        graph[a[0 : (nn + 1), i], i] = 1
    graph = np.maximum(graph, graph.T)
    np.fill_diagonal(graph, 0)
    return graph

