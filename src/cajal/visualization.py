# Functions for visualizing and clustering the GW morphology space
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import leidenalg
import community as community_louvain
import igraph as ig
import itertools as it
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

def plot_categorical_data(
    embedding : npt.NDArray[np.float_],
    categories : list,
    names : Optional[list[str]],
    savefile : Optional[str]
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
    x_values : list[float] = list(embedding[:,0])
    y_values : list[float] = list(embedding[:,1])
    num_pts = len(x_values)
    nested_pts_lists = []
    if names is None:
        names = it.repeat(None,num_pts)
    for (name,x,y,color) in zip(names,x_values, y_values, colors):
        if color == len(nested_pts_lists):
            nested_pts_lists.append([ (name,x,y) ])
        elif color < len(nested_pts_lists):
            nested_pts_lists[color].append( (name,x,y) )
        else:
            raise Exception("Wah")
   
    fig, ax = plt.subplots(figsize = (10,10),facecolor='white')
    counter = 0
    annotations = []
    for ell in nested_pts_lists:
        name_list =[]
        x_list = []
        y_list = []
        for name, x, y in ell:
            annotations.append(plt.annotate(name, (x,y)))
            name_list.append(x)
            x_list.append(x)
            y_list.append(y)
        x_arr = np.array(x_list,dtype=np.float_)
        y_arr = np.array(y_list,dtype=np.float_)
        length = len(x_list)
        color = matplotlib.colormaps['tab20'].colors[counter]
        colors= np.tile(color,(length,1))
        ax.scatter(x_arr, y_arr, s=200, 
                   c=colors,
                   label=colorkey[counter],
                   edgecolors='none')
        counter +=1
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0+box.width * 0.2,
                     box.y0+ box.height * 0.2,
                     box.width * 0.8,
                     box.height * 0.8])
    # Put a legend to the right of the current axis
    lgd = ax.legend(loc='center right', bbox_to_anchor=(-0.1, 0.5))
    if savefile is not None:
        plt.savefig(savefile, bbox_extra_artists=[lgd,ax]+annotations, bbox_inches='tight', pad_inches=.2)
    else:
        plt.show()


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

