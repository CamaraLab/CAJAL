# Compute average graph shape of neurons based on geodesic GW clusters
import copy
import os
import numpy as np
from scipy.sparse.csgraph import dijkstra
import itertools as it
import networkx as nx
from CAJAL.lib.utilities import pj, load_dist_mat


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
    mask = np.array([True] * (dist_mat.shape[0] * dist_mat.shape[1])).reshape(dist_mat.shape)
    for i in range(1, len(spt[1])):
        mask[i, spt[1][i]] = False
        mask[spt[1][i], i] = False
    dist_orig[mask] = 0
    return dist_orig


def get_avg_shape_spt(cluster_ids, clusters, data_dir, files_list, match_list, gw_dist, k=3):
    pairs = np.array(list(it.combinations(range(len(files_list)), 2)))

    # Get cell indices in cluster
    indices = np.where(np.isin(np.array(clusters), cluster_ids))[0]

    # Get medoid cell as one with lowest avg distance to others in cluster
    medoid = indices[np.argmin(np.sum(gw_dist[indices][:, indices], axis=0))]

    # Get average distance matrix
    d_avg = load_dist_mat(pj(data_dir, files_list[medoid]))
    d_avg = d_avg / np.min(d_avg[d_avg != 0])  # normalize step size
    d_avg_total = copy.deepcopy(d_avg)
    d_avg[d_avg > 2] = 2
    for i in indices[indices != medoid]:
        pairs_index = np.where(np.logical_or(np.logical_and(pairs[:, 0] == medoid, pairs[:, 1] == i),
                                             np.logical_and(pairs[:, 1] == medoid, pairs[:, 0] == i)))[0]
        match_mat = match_list[match_list.files[int(pairs_index)]]
        i_reorder = np.argmax(match_mat, axis=int(np.where(pairs[pairs_index] != medoid)[1]))
        di = load_dist_mat(pj(data_dir, files_list[i]))
        di = di / np.min(di[di != 0])  # normalize step size
        di = di[i_reorder][:, i_reorder]
        d_avg_total = d_avg_total + di
        di[di > 2] = 2
        d_avg = d_avg + di
    d_avg = d_avg / len(indices)
    d_avg_total = d_avg_total / len(indices)
    d_avg_total[d_avg_total == 0] = np.max(d_avg_total)  # So that 0s don't get caught in min
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


def plot_networkx(d, ax, color=None, layout="spring", **kwargs):
    # Color is a vector of values for each node - edges will be colored based on their end node
    nx_graph = nx.convert_matrix.from_numpy_matrix(d)
    layout_dict = {
        "spring": nx.spring_layout,
        "spectral": nx.spectral_layout,
        "circular": nx.circular_layout
    }
    layout_func = layout_dict.get(layout)
    pos = layout_func(nx_graph)
    if color is None:
        color = range(len(nx_graph.nodes))
    edge_end = [i[1] for i in nx_graph.edges]
    plot_color = np.array(color)[edge_end]
    nx.draw_networkx_edges(nx_graph, pos, ax=ax, edge_color=plot_color, **kwargs)
    # nx.draw_networkx_nodes(nx_graph, pos, nodelist=nx_graph.nodes, node_color=color,
    #                        with_labels=False, node_size=2, ax=ax)
