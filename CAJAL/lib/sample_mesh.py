# Functions for sampling points from a triangular mesh
import os
import csv
import numpy as np
import potpourri3d as pp3d
import networkx as nx
from scipy.spatial.distance import squareform, cdist, euclidean
import trimesh
import itertools as it
import warnings
from multiprocessing import Pool

from CAJAL.lib.run_gw import pj


def read_obj(file_path):
    """
    Reads in the vertices and triangular faces of a .obj file

    Args:
        file_path (string): Path to .obj file

    Returns:
        vertices (numpy array): 3D coordinates for vertices
        faces (numpy array): row of vertices contained in each face
    """
    obj_file = open(file_path, "r")
    obj_split = csv.reader(obj_file, delimiter=" ")
    vertices = []
    faces = []
    for line in obj_split:
        if line[0] == "v":
            vertices.append([float(x) for x in line[1:]])
        elif line[0] == "f":
            faces.append([float(x) for x in line[1:]])
        # Skipping over any vertex textures or normals
    obj_file.close()
    vertices = np.array(vertices)
    faces = np.array(faces)-1
    faces = faces.astype("int64")
    return vertices, faces


def sample_vertices(vertices, n_sample):
    """
    Evenly samples n vertices. Most .obj vertices are ordered counter-clockwise, so evenly sampling from vertex matrix
    can roughly approximate even sampling across the mesh

    Args:
        vertices (numpy array): 3D coordinates for vertices
        n_sample (integer): number of vertices to sample

    Returns:
        numpy array of sampled vertices
    """
    return vertices[np.linspace(0, vertices.shape[0]-1, n_sample).astype("uint32"), :]


def save_sample_vertices(vertices, n_sample, outfile):
    """
    Evenly sample n vertices and save in csv

    Args:
        vertices (numpy array): 3D coordinates for vertices
        n_sample (integer): number of vertices to sample
        outfile (string): file path to write vertices

    Returns:
        None (writes to file)
    """
    new_vertices = sample_vertices(vertices, n_sample)
    np.savetxt(outfile, new_vertices, delimiter=",", fmt="%.16f")


def save_sample_from_obj(file_name, infolder, outfolder, n_sample):
    """
    Evenly sample n vertices from .obj and save in csv

    Args:
        file_name (string): .obj file name
        infolder (string): folder containing .obj file
        outfolder (string): folder to save sampled vertices csv
        n_sample (integer): number of vertices to sample

    Returns:
        None (writes to file)
    """
    vertices, faces = read_obj(pj(infolder, file_name))
    save_sample_vertices(vertices, n_sample, pj(outfolder, file_name.replace(".obj",".csv")))


def save_sample_from_obj_parallel(infolder, outfolder, n_sample, num_cores=8):
    """
    Computes geodesic distance in parallel processes for all meshes in .obj files in a directory

    Args:
        infolder(string): path to directory containing .obj files
        outfolder (string): path to directory to write distance matrices
        n_sample (integer): number of vertices to sample from each mesh
        num_cores (integer): number of processes to use for parallelization

    Returns:
        None (writes files to outfolder)
    """
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)
    arguments = [(file_name, infolder, outfolder, n_sample)
                 for file_name in os.listdir(infolder)]
    with Pool(processes=num_cores) as pool:
        pool.starmap(save_sample_from_obj, arguments)


def connect_mesh(vertices, faces):
    """
    Adds triangles to mesh to form a minimum spanning tree of its connected components

    Args:
        vertices (numpy array): 3D coordinates for vertices
        faces (numpy array): row of vertices contained in each face

    Returns:
        numpy array of new faces including connecting triangles
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    graph = trimesh.graph.vertex_adjacency_graph(mesh)
    connected_components = [i[1] for i in enumerate(nx.connected_components(graph))]
    if len(connected_components) == 1:
        return vertices, faces
    # Need to find a minimum spanning tree of edges I can add between components
    # Dictionary of faces I could add
    connections = {}
    # Graph of possible connections
    spt_graph = nx.Graph()
    # Add components as nodes in graph
    for i in range(len(connected_components)): spt_graph.add_node(i)
    # For every pair of components, find nearest points
    for i in range(len(connected_components)):
        for j in range(i+1, len(connected_components)):
            # Set bigger component as reference so we can make KDTree of that and loop over smaller
            ref, query = (i, j) if len(connected_components[i]) > len(connected_components[j]) else (j, i)
            ref_ids = list(connected_components[ref])
            query_ids = list(connected_components[query])
            dist_ij = cdist(vertices[ref_ids], vertices[query_ids])
            nearest = np.unravel_index(dist_ij.argmin(), dist_ij.shape)
            next_nearest = np.argsort(dist_ij[nearest[0]])[1]
            face = [ref_ids[nearest[0]], query_ids[nearest[1]], query_ids[next_nearest]]
            # Save face so I know to add it if this edge is in SPT
            connections[(i, j)] = [face]
            spt_graph.add_edge(i, j, weight=dist_ij[nearest])
    new_faces = []
    for u, v, w in nx.minimum_spanning_edges(spt_graph):
        new_faces = new_faces + connections[(min(u, v), max(u, v))]
    new_faces = np.array(new_faces)
    return np.vstack([faces, new_faces])


def get_geodesic_heat(vertices, faces, n_sample, connect=False):
    """
    Computes geodesic distance between n_sample points on the triangular mesh using heat method

    Args:
        vertices (numpy array): 3D coordinates for vertices
        faces (numpy array): row of vertices contained in each face
        n_sample (integer): number of vertices to sample
        connect (boolean): whether to check for disconnected meshes and connect them simply by adding faces

    Result:
        heat geodesic distance in vector form
    """
    if vertices.shape[0] < n_sample:
        warnings.warn("Fewer vertices than points to sample, skipping")
        return None
    if connect:
        faces = connect_mesh(vertices, faces)
    even_sample = np.linspace(0, vertices.shape[0]-1, n_sample).astype("uint32")
    solver = pp3d.MeshHeatMethodDistanceSolver(vertices, faces)
    dist_mat = solver.compute_distance(even_sample[0])[even_sample]
    for i in range(1, len(even_sample)):
        dist_mat = np.c_[dist_mat, solver.compute_distance(even_sample[i])[even_sample]]
    dist_mat = np.maximum(dist_mat, dist_mat.T)  # symmetrize
    dist_vec = squareform(dist_mat)
    return dist_vec


def save_geodesic_heat(vertices, faces, n_sample, outfile, connect=False):
    """
    Saves geodesic distance vector between n_sample points on the triangular mesh using heat method

    Args:
        vertices (numpy array): 3D coordinates for vertices
        faces (numpy array): row of vertices contained in each face
        n_sample (integer): number of vertices to sample
        outfile (string): file path to write to
        connect (boolean): whether to check for disconnected meshes and connect them simply by adding faces

    Result:
        None (writes to file)
    """
    dist_vec = get_geodesic_heat(vertices, faces, n_sample, connect)
    if dist_vec is not None:
        np.savetxt(outfile, dist_vec, fmt='%.8f')


def get_geodesic_networkx(vertices, faces, n_sample, connect=False):
    """
    Computes geodesic distance between n_sample points on the triangular mesh using
    graph distance on edges between vertices

    Args:
        vertices (numpy array): 3D coordinates for vertices
        faces (numpy array): row of vertices contained in each face
        n_sample (integer): number of vertices to sample
        connect (boolean): whether to check for disconnected meshes and connect them simply by adding faces

    Result:
        graph geodesic distance in vector form
    """
    if vertices.shape[0] < n_sample:
        warnings.warn("Fewer vertices than points to sample, skipping")
        return None
    if connect:
        faces = connect_mesh(vertices, faces)
    graph = nx.Graph()
    for i in range(vertices.shape[0]):
        graph.add_node(i)
    for i in range(faces.shape[0]):
        for edge_id in [(0, 1), (1, 2), (2, 0)]:
            v1 = faces[i, edge_id[0]]
            v2 = faces[i, edge_id[1]]
            graph.add_edge(v1, v2, weight=euclidean(vertices[v1], vertices[v2]))
    even_sample = np.linspace(0, vertices.shape[0]-1, n_sample).astype("uint32")
    arguments = list(it.combinations(even_sample, 2))
    dist_vec = np.zeros(len(arguments))
    for i in range(len(arguments)):
        pair = arguments[i]
        dist_vec[i] = nx.algorithms.shortest_paths.generic.shortest_path_length(graph, source=pair[0], target=pair[1],
                                                                                weight="weight")
    return dist_vec


def save_geodesic(vertices, faces, n_sample, outfile, method="networxk", connect=False):
    """
    Saves geodesic distance vector between n_sample points on the triangular mesh using
    graph distance on edges between vertices

    Args:
        vertices (numpy array): 3D coordinates for vertices
        faces (numpy array): row of vertices contained in each face
        n_sample (integer): number of vertices to sample
        outfile (string): file path to write to
        method (string): one of 'networxk' or 'heat', how to compute geodesic distance
                networkx is slower but more exact for non-watertight methods, heat is a faster approximation
        connect (boolean): whether to check for disconnected meshes and connect them simply by adding faces

    Result:
        None (writes to file)
    """
    if method == "networkx":
        dist_vec = get_geodesic_networkx(vertices, faces, n_sample, connect)
    elif method == "heat":
        dist_vec = get_geodesic_heat(vertices, faces, n_sample, connect)
    else:
        raise Exception("Invalid method, must be one of 'networkx' or 'heat'")
    if dist_vec is not None:
        np.savetxt(outfile, dist_vec, fmt='%.8f')


def save_geodesic_from_obj(file_name, infolder, outfolder, n_sample, method="networxk", connect=False):
    """
    Computes geodesic distance for mesh in .obj file

    Args:
        infolder(string): path to directory containing .obj files
        outfolder (string): path to directory to write distance matrices
        n_sample (integer): number of vertices to sample from each mesh
        method (string): one of 'networxk' or 'heat', how to compute geodesic distance
                networkx is slower but more exact for non-watertight methods, heat is a faster approximation
        connect (boolean): whether to check for disconnected meshes and connect them simply by adding faces

    Returns:
        None (writes files to outfolder)
    """
    vertices, faces = read_obj(pj(infolder, file_name))
    outfile = pj(outfolder, file_name.replace(".obj", "_dist.txt"))
    save_geodesic(vertices, faces, n_sample, outfile, method, connect)


def save_geodesic_from_obj_parallel(infolder, outfolder, n_sample, method="networkx", connect=False, num_cores=8):
    """
    Computes geodesic distance in parallel processes for all meshes in .obj files in a directory

    Args:
        infolder(string): path to directory containing .obj files
        outfolder (string): path to directory to write distance matrices
        n_sample (integer): number of vertices to sample from each mesh
        method (string): one of 'networxk' or 'heat', how to compute geodesic distance
                networkx is slower but more exact for non-watertight methods, heat is a faster approximation
        connect (boolean): whether to check for disconnected meshes and connect them simply by adding faces
        num_cores (integer): number of processes to use for parallelization

    Returns:
        None (writes files to outfolder)
    """
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)
    arguments = [(file_name, infolder, outfolder, n_sample, method, connect)
                 for file_name in os.listdir(infolder)]
    with Pool(processes=num_cores) as pool:
        pool.starmap(save_geodesic_from_obj, arguments)
