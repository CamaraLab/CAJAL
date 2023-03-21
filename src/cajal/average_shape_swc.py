"""
Compute average graph shape of neurons based on geodesic GW clusters
"""
from typing import Collection
import copy
import os
import numpy as np
from scipy.sparse.csgraph import dijkstra
import itertools as it
import networkx as nx
import matplotlib.pyplot as plt
from .utilities import pj, load_dist_mat

# import utilities - this works
# from utilities import pj, load_dist_mat


def get_spt(dist_mat):
    """
    Gets shortest path tree given the geodesic/graph distance between points

    Args:
        dist_mat (numpy array): graph distance matrix representing a tree structure
            (i.e. geodesic distance of a neuron trace)

    Returns:
        adjacency matrix of shortest path tree (graph distance matrix with non-shortest path edges set to 0)
    """
    dist_mat = copy.deepcopy(dist_mat)
    dist_orig = copy.deepcopy(dist_mat)

    # Each node has direct "edge" to soma because it is a full distance matrix
    # That edge should be the sum of the actual shortest path to the soma
    # This incentivizes taking more edges, therefor should take the actual shortest path
    dist_mat = dist_mat - np.min(dist_mat[dist_mat != 0]) / 2
    dist_mat = np.maximum(dist_mat, 0)

    dist_mat = np.maximum(dist_mat, dist_mat.T)

    # Get shortest path tree
    spt = dijkstra(dist_mat, directed=False, indices=0, return_predecessors=True)

    # Get graph representation by only keeping distances on edges from spt
    mask = np.array([True] * (dist_mat.shape[0] * dist_mat.shape[1])).reshape(
        dist_mat.shape
    )
    for i in range(1, len(spt[1])):
        mask[i, spt[1][i]] = False
        mask[spt[1][i], i] = False
    dist_orig[mask] = 0
    return dist_orig


def get_avg_shape_spt(
    cluster_ids: Collection[int],
    clusters: npt.NDArray[np.int_],
    data_dir: str,
    files_list: list[str],
    match_list,
    gw_dist,
    k=3,
):
    """
    :param cluster_ids: A (possibly restricted) collection of the clusters that we want\
    to compute the average shape of. In the case where you want to compute the average shape of\
    a single cluster, this set should only contain one element, the index of the cluster.
    :param clusters: Shape (num_cells,). Indices represent cells; the entry at index k \
    is an integer representing the cluster to which cell k belongs.
    """

    # pairs.shape == (n *(n-1)/2,2)
    pairs = np.array(list(it.combinations(range(len(files_list)), 2)))

    # Clusters is an array of shape (num_cells,).
    # np.isin(clusters, cluster_ids) is of shape (num_cells,).
    # At index k, it is True if clusters[k] is in cluster_ids, else False.
    # indices is then a vector of indices for the cells which lie in the cluster.
    indices = np.where(np.isin(np.array(clusters), cluster_ids))[0]

    # Get medoid cell as one with lowest avg distance to others in cluster
    medoid = indices[np.argmin(np.sum(gw_dist[indices][:, indices], axis=0))]

    # Get average distance matrix
    d_avg = load_dist_mat(pj(data_dir, files_list[medoid]))
    d_avg = d_avg / np.min(d_avg[d_avg != 0])  # normalize step size
    d_avg_total = copy.deepcopy(d_avg)
    d_avg[d_avg > 2] = 2
    for i in indices[indices != medoid]:
        pairs_index: npt.NDArray[np.int_]
        # `pairs_index` should be an array containing exactly one element,
        # which is the row of (medoid,i) or (i, medoid)
        pairs_index = np.where(
            np.logical_or(
                np.logical_and(pairs[:, 0] == medoid, pairs[:, 1] == i),
                np.logical_and(pairs[:, 1] == medoid, pairs[:, 0] == i),
            )
        )[0]
        # Get the coupling matrix between the two cells
        match_mat = match_list[match_list.files[int(pairs_index)]]

        # Because pairs_index is an _array, pairs[pairs_index] is subscripting
        # by that array, i.e., it returns [ pairs[x] ] for x in pairs_index.
        # Since pairs_index contains one element, this is just [[i,medoid]]
        # (note the nesting).
        # Thus pairs[pairs_index] != medoid is either [[True, False]] or
        # [[False, True]] is of shape (1,2).
        # Now, np.where() will return an ordered pair (arr0, arr1),
        # where both are of rank 1.
        # arr0 contains the indices of the nonzero matrices along the 0th axis
        # and arr1 contains the indices of the nonzero matrices along the 1st axis.
        # In our case we care about which columns are nonzero,
        # because the columns are either [T] or [F].
        # Thus, we want arr1 = np.where(...)[1].
        # This will return the array of all indices where this is nonzero,
        # which is again a single element, which in our case is either zero or one.
        # Ultimately, axis is 0 if i is the 0th element of the pair,
        # and 1 if i is the 1st element of the pair.

        # Now, in the coupling matrix, if i < medoid so the coupling
        # matrix is a coupling from i to medoid,
        # we get the index along the 0th axis (along rows, i.e. within columns)
        # of the max value of the coupling.
        # If i_reorder[j]=k, then it means that in column j, the
        # mode of the distribution is at index k, i.e., we are most likely to
        # send j to k.
        i_reorder = np.argmax(
            match_mat, axis=int(np.where(pairs[pairs_index] != medoid)[1])
        )

        # pairs_
        di = load_dist_mat(pj(data_dir, files_list[i]))
        di = di / np.min(di[di != 0])  # normalize step size
        # The new distribution is given by reordering the labels
        # according to this map, with the possibility that nodes are repeated.
        # Note that the new matrix will not represent a metric space,
        # as there may be repeated points with zero distance.
        di = di[i_reorder][:, i_reorder]
        d_avg_total = d_avg_total + di
        di[di > 2] = 2
        d_avg = d_avg + di
    d_avg = d_avg / len(indices)
    d_avg_total = d_avg_total / len(indices)
    d_avg_total[d_avg_total == 0] = np.max(
        d_avg_total
    )  # So that 0s don't get caught in min
    confidence = np.min(d_avg_total, axis=0)

    d = copy.deepcopy(d_avg)

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
    return d_avg, get_spt(load_dist_mat(pj(data_dir, files_list[medoid]))), confidence


def plot_networkx(d, ax, color=None, layout="spring"):
    # Color is a vector of values for each node - edges will be colored based on their end node
    nx_graph = nx.convert_matrix.from_numpy_matrix(d)
    layout_dict = {
        "circular": nx.circular_layout,
        "kamada_kawai": nx.kamada_kawai_layout,
        "planar": nx.planar_layout,
        "random": nx.random_layout,
        "shell": nx.shell_layout,
        "spectral": nx.spectral_layout,
        "spring": nx.spring_layout,
    }
    layout_func = layout_dict.get(layout)
    pos = layout_func(nx_graph)
    if color is None:
        color = range(len(nx_graph.nodes))
    edge_end = [i[1] for i in nx_graph.edges]
    plot_color = np.array(color)[edge_end]
    nx.draw_networkx_edges(
        nx_graph, pos, alpha=1, width=2, ax=ax, edge_color=plot_color
    )
    nx.draw_networkx_nodes(
        nx_graph, pos, nodelist=nx_graph.nodes, node_color=color, node_size=2, ax=ax
    )


def plot_avg_medoid(avg_spt, medoid_spt, color=None, figheight=6):
    fig = plt.figure(figsize=(figheight * 2, figheight))
    ax = fig.add_subplot(1, 2, 1)
    plot_networkx(medoid_spt, ax, color=color)
    plt.title("Clusters medoid")
    ax = fig.add_subplot(1, 2, 2)
    plot_networkx(avg_spt, ax, color=color)
    plt.title("Clusters avg")


def plot_all_avg_shapes(
    cluster_ids_list,
    clusters,
    data_dir,
    files_list,
    match_list,
    gw_dist,
    k=3,
    figheight=6,
):
    nrow = len(cluster_ids_list)
    fig = plt.figure(figsize=(figheight * 2, figheight * nrow))
    for a in range(nrow):
        cluster_ids = cluster_ids_list[a]
        avg_spt, medoid_spt, confidence = get_avg_shape_spt(
            cluster_ids, clusters, data_dir, files_list, match_list, gw_dist, k
        )
        ax = fig.add_subplot(nrow, 2, 2 * a + 1)
        plot_networkx(medoid_spt, ax, color=confidence)
        plt.title("Clusters medoid: " + ",".join([str(i) for i in cluster_ids]))
        ax = fig.add_subplot(nrow, 2, 2 * a + 2)
        plot_networkx(avg_spt, ax, color=confidence)
        plt.title("Clusters avg: " + ",".join([str(i) for i in cluster_ids]))
