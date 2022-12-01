# Functions for sampling even points from an SWC reconstruction of a neuron
import re
import numpy as np
from CAJAL.lib.utilities import pj
from scipy.spatial.distance import euclidean, squareform
import networkx as nx
import warnings
from multiprocessing import Pool
# import time
import os
from collections.abc import Iterable

StructureID = int
CoordTriple = tuple[float,float,float]

def read_swc(file_path):
    """
    Reads an SWC file and returns a list of lists of strings.

    The SWC file should conform to the documentation here: http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html

    In particular, all rows should be either a comment starting with the character "#" or should have at least eight strings separated by whitespace.
    
    read_swc(file_path)[i] is the i-th non-comment row, split into a list of strings by whitespace.

    If there are fewer than eight whitespace-separated tokens in the i-th row, an error is raised.

    If there are greater than eight whitespace-separated tokens in the i-th row, the first eight tokens are kept and the rest discarded.

    The eighth token is assumed to be the parent sample index, which is -1 for the root.

    read_swc expects the rows of the graph to be in topologically sorted order (parents before children) If this is not satisfied, read_swc raises an exception.
    In particular, the first node must be the root of the tree, and its parent has index -1.
        
    Args:
        file_path (string): absolute path to SWC file

    Returns:
        list of vertex rows from SWC file, where each vertex row is a list of eight strings.
    """
    vertices = []
    ids = set()
    ids.add("-1")
    with open(file_path, "r") as f:
        for line in f:
            if line[0] == "#":
                continue
            row = re.split("\s|\t", line.strip())[0:8]
            if len(row) < 8:
                raise TypeError("Row" + row + "in file" + file_path + "has fewer than eight whitespace-separated strings.")
            if row[7] not in ids:
                raise ValueError("SWC parent nodes must be listed before the child node that references them. The node with index "
                                 + row[0] + " was accessed before its parent "+ row[7])
            ids.add(row[0])
            vertices.append(row)
    return vertices

def prep_coord_dict(vertices, types_keep=(0, 1, 2, 3, 4), keep_disconnect=False):
    """
    Read through swc file list, get dictionary of vertex coordinates and total length of all segments

    Args:
        vertices (list): list of vertex rows from SWC file
        types_keep (tuple,list): list of SWC neuron part types to sample points from
            by default, uses only 1 (soma), 2 (axon), 3 (basal dendrite), 4 (apical dendrite)
        keep_disconnect (boolean): If False, will only keep branches connected to the soma.
            If True, will keep all branches, including free-floating ones

    Returns:
        vertices_keep: list of rows from SWC file that are connected to the soma
        vertex_coords: dictionary of xyz coordinates for the ID of each vertex in vertices_keep
        total_length: sum of segment lengths from branches of kept vertices
    """
    # in case types_keep are numbers
    types_keep = [str(x) for x in types_keep] if isinstance(types_keep, Iterable) else [str(types_keep)]

    vertices_keep = []
    vertex_coords = {}
    total_length = 0
    for v in vertices:
        this_id = int(v[0])
        this_coord = np.array((float(v[2]), float(v[3]), float(v[4])))
        pid = int(v[-1])
        if pid < 0:
            # If not keeping disconnected parts, only keep vertex without parent
            # if it has soma type or is first vertex
            if keep_disconnect or v[1] == "1" or len(vertices_keep) == 0:
                vertex_coords[this_id] = this_coord
                vertices_keep.append(v)
        elif pid >= 0 and pid in vertex_coords.keys():
            # keep branch vertex if connected to soma origin
            vertex_coords[this_id] = this_coord
            vertices_keep.append(v)
            seg_len = euclidean(vertex_coords[pid], this_coord)
            if v[1] in types_keep:
                total_length += seg_len
    return vertices_keep, vertex_coords, total_length


def sample_pts_step(vertices, vertex_coords, step_size, types_keep=(0, 1, 2, 3, 4)):
    """
    Sample points at every set interval (step_size) along branches of neuron

    Args:
        vertices (list): list of rows from SWC file that are connected to the soma
        vertex_coords (dict): dictionary of xyz coordinates for the ID of each vertex in vertices_keep
        step_size (float): even distance to sample points radially from soma
        types_keep (tuple,list): list of SWC neuron part types to sample points from
            by default, uses only 1 (soma), 2 (axon), 3 (basal dendrite), 4 (apical dendrite)

    Returns:
        sampled_pts_list: list of xyz coordinates for sampled points
        num_origins: number of connected parts of neuron, should be 1
    """
    vertex_dist = {}
    sampled_pts_list = []
    num_origins = 0

    # in case types_keep are numbers
    types_keep = [str(x) for x in types_keep] if isinstance(types_keep, Iterable) else [str(types_keep)]

    # loop through list of vertices, sampling points from edge of vertex to parent
    for v in vertices:
        this_id = int(v[0])
        this_coord = np.array((float(v[2]), float(v[3]), float(v[4])))
        pid = int(v[-1])
        if pid < 0:
            num_origins += 1
            vertex_dist[this_id] = 0
            sampled_pts_list.append(this_coord)
            continue
        seg_len = euclidean(vertex_coords[pid], this_coord)
        pts_dist = np.arange(step_size, seg_len + vertex_dist[pid], step_size)
        if v[1] in types_keep and len(pts_dist) > 0:
            pts_dist = pts_dist - vertex_dist[pid]
            new_dist = seg_len - pts_dist[-1]
            new_pts = [vertex_coords[pid] + (this_coord - vertex_coords[pid]) * x / seg_len for x in pts_dist]
            if v[1] in types_keep:
                sampled_pts_list.extend(new_pts)
            vertex_dist[this_id] = new_dist
        else:
            vertex_dist[this_id] = vertex_dist[pid] + seg_len
    return sampled_pts_list, num_origins


def sample_n_pts(vertices, vertex_coords, total_length, types_keep=(0, 1, 2, 3, 4),
                 goal_num_pts=50, min_step_change=1e-7, max_iters=50, verbose=False):
    """
    Use binary search to find step size between points that will sample the required number of points

    Args:
        vertices (list): list of rows from SWC file that are connected to the soma
        vertex_coords (dict): dictionary of xyz coordinates for the ID of each vertex in vertices_keep
        total_length (float): sum of segment lengths from branches of kept vertices
        types_keep (tuple,list): list of SWC neuron part types to sample points from
            by default, uses only 1 (soma), 2 (axon), 3 (basal dendrite), 4 (apical dendrite)
        goal_num_pts (integer): number of points to sample
        min_step_change (float): stops while loop from infinitely trying closer and closer step sizes
        max_iters (integer): maximum number of iterations of while loop
        verbose (boolean): if true, will print step size information for each search iteration

    Returns:
        sampled_pts: list of xyz coordinates of sampled points
        num_pts: actual number of points sampled
        step_size: step size that samples required number of points
        i: number of iterations to reach viable step size
    """
    num_pts = 0
    min_step_size = 0
    max_step_size = total_length
    prev_step_size = max_step_size
    step_size = (min_step_size + max_step_size) / 2.0
    i = 0
    while num_pts != goal_num_pts and abs(step_size - prev_step_size) > min_step_change and i < \
            max_iters:
        i += 1
        sampled_pts_list, num_origins = sample_pts_step(vertices, vertex_coords, step_size, types_keep)
        if num_origins > goal_num_pts:
            warnings.warn("More disconnected segments in neuron than points to sample, skipping")
            return None

        # continue binary search
        num_pts = len(sampled_pts_list)
        if num_pts < goal_num_pts:
            max_step_size = step_size
            prev_step_size = step_size
            step_size = (min_step_size + max_step_size) / 2.0
        elif num_pts > goal_num_pts:
            min_step_size = step_size
            prev_step_size = step_size
            step_size = (min_step_size + max_step_size) / 2.0
        # else will stop next loop

        if verbose:
            print("Iteration", i)
            print("Num pts", num_pts)
            print("Prev step size", prev_step_size)
            print("Step size", step_size)
            print("")
    if i == 0:
        raise Exception("Sampled 0 points from neuron, could be too large of min_step_change, or types_keep does not include values in second column of SWC files")
    else:
        sampled_pts = np.array(sampled_pts_list)
        return sampled_pts, num_pts, step_size, i


def sample_network_step(vertices, vertex_coords, step_size, types_keep=(0, 1, 2, 3, 4)):
    """
    Sample points at every set interval (step_size) along branches of neuron, return networkx

    Args:
        vertices (list): list of rows from SWC file that are connected to the soma
        vertex_coords (dict): dictionary of xyz coordinates for the ID of each vertex in vertices_keep
        step_size (float): even distance to sample points radially from soma
        types_keep (tuple,list): list of SWC neuron part types to sample points from
            by default, uses only 1 (soma), 2 (axon), 3 (basal dendrite), 4 (apical dendrite)

    Returns:
        graph: networkx graph of sampled points weighted by distance between points
    """
    vertex_dist = {}
    num_origins = 0
    graph = nx.Graph()
    prev_pts = {}  # Save last point before this one so can connect edge
    # pos = {}

    # in case types_keep are numbers
    types_keep = [str(x) for x in types_keep] if isinstance(types_keep, Iterable) else [str(types_keep)]

    # loop through list of vertices, sampling points from edge of vertex to parent
    for v in vertices:
        this_id = int(v[0])
        this_coord = np.array((float(v[2]), float(v[3]), float(v[4])))
        pid = int(v[-1])
        if pid < 0:
            num_origins += 1
            vertex_dist[this_id] = 0
            graph.add_node(str(this_id))
            # pos[str(this_id)] = this_coord[:2]
            prev_pts[this_id] = str(this_id)
            continue
        seg_len = euclidean(vertex_coords[pid], this_coord)
        pts_dist = np.arange(step_size, seg_len + vertex_dist[pid], step_size)
        if v[1] in types_keep and len(pts_dist) > 0:
            pts_dist = pts_dist - vertex_dist[pid]
            new_dist = seg_len - pts_dist[-1]
            new_pts = [vertex_coords[pid] + (this_coord - vertex_coords[pid]) * x / seg_len for x in pts_dist]
            new_pts_ids = [prev_pts[pid]] + [str(this_id) + "_" + str(x) for x in range(len(pts_dist))]
            new_pts_len = [vertex_dist[pid] + euclidean(new_pts[0], vertex_coords[pid])] + \
                          [euclidean(new_pts[i], new_pts[i - 1]) for i in range(1, len(new_pts))]
            # Add new points to graph, with edge weighted by euclidean to parent
            if v[1] in types_keep:
                for i in range(1, len(new_pts_ids)):
                    graph.add_node(new_pts_ids[i])
                    # pos[new_pts_ids[i]] = new_pts[i - 1][:2]
                    graph.add_edge(new_pts_ids[i - 1], new_pts_ids[i], weight=new_pts_len[i - 1])
            vertex_dist[this_id] = new_dist
            prev_pts[this_id] = new_pts_ids[-1]
        else:
            vertex_dist[this_id] = vertex_dist[pid] + seg_len
            prev_pts[this_id] = prev_pts[pid]
    return graph  # , pos


def get_sample_pts(file_name, infolder, types_keep=(0, 1, 2, 3, 4),
                   goal_num_pts=50, min_step_change=1e-7,
                   max_iters=50, keep_disconnect=True, verbose=False):
    """
    Sample points from SWC file

    Args:
        file_name (string): SWC file name (including .swc)
        infolder (string): path to folder containing SWC file
        types_keep (tuple,list): list of SWC neuron part types to sample points from
            by default, uses only 1 (soma), 2 (axon), 3 (basal dendrite), 4 (apical dendrite)
        goal_num_pts (integer): number of points to sample
        min_step_change (float): stops while loop from infinitely trying closer and closer step sizes
        max_iters (integer): maximum number of iterations of while loop
        keep_disconnect (boolean): if True, will keep all branches from SWC. if False, will keep only connected to soma
        verbose (boolean): if True, will print step size information for each search iteration

    Returns:
        list:
            [0]: list of xyz coordinates of sampled points
            [1]: actual number of points sampled
            [2]: step size that samples required number of points
            [3]: number of iterations to reach viable step size
    """

    if file_name[-4:] != ".SWC" and file_name[-4:] != ".swc":
        warnings.warn("Input file must be a .swc or .SWC file, skipping")
        return None

    # Read SWC file
    swc_list = read_swc(pj(infolder, file_name))

    # Get total length of segment type (for max step size)
    coord_list_out = prep_coord_dict(swc_list, types_keep, keep_disconnect=keep_disconnect)
    if coord_list_out is None:
        return None

    return sample_n_pts(coord_list_out[0], coord_list_out[1], coord_list_out[2], types_keep,
                        goal_num_pts, min_step_change, max_iters, verbose)


def save_sample_pts(file_name, infolder, outfolder, types_keep=(0, 1, 2, 3, 4),
                    goal_num_pts=50, min_step_change=1e-7,
                    max_iters=50, keep_disconnect=True, verbose=False):
    """
        Sample points from SWC file and save them in CSV

        Args:
            file_name (string): SWC file name (including .swc)
            infolder (string): path to folder containing SWC file
            outfolder (string): path to output folder to save CSVs
            types_keep (tuple,list): list of SWC neuron part types to sample points from
                by default, uses only 1 (soma), 2 (axon), 3 (basal dendrite), 4 (apical dendrite)
            goal_num_pts (integer): number of points to sample
            min_step_change (float): stops while loop from infinitely trying closer and closer step sizes
            max_iters (integer): maximum number of iterations of while loop
            keep_disconnect (boolean): if True, will keep all branches from SWC.
                if False, will keep only connected to soma
            verbose (boolean): if true, will print step size information for each search iteration

        Returns:
            Boolean success of sampling points from this SWC file
    """
    sample_pts_out = get_sample_pts(file_name, infolder, types_keep, goal_num_pts,
                                    min_step_change, max_iters, keep_disconnect, verbose)

    if sample_pts_out is None:
        return False

    if sample_pts_out[1] == goal_num_pts:
        np.savetxt(pj(outfolder, file_name[:-4] + ".csv"), np.array(sample_pts_out[0]), delimiter=",", fmt="%.16f")
        return True
    else:
        return False


def get_geodesic(file_name, infolder, types_keep=(0, 1, 2, 3, 4),
                 goal_num_pts=50, min_step_change=1e-7,
                 max_iters=50, verbose=False):
    """
    Sample points from SWC file, compute geodesic distance (networkx graph distance) between
    points, returns distance in vector form

    Args:
        file_name (string): SWC file name (including .swc)
        infolder (string): path to folder containing SWC file
        types_keep (tuple,list): list of SWC neuron part types to sample points from
            by default, uses only 1 (soma), 2 (axon), 3 (basal dendrite), 4 (apical dendrite)
        goal_num_pts (integer): number of points to sample
        min_step_change (float): stops while loop from infinitely trying closer and closer step sizes
        max_iters (integer): maximum number of iterations of while loop
        verbose (boolean): if true, will print step size information for each search iteration

    Returns:
        Boolean success of sampling points from this SWC file
    """

    if file_name[-4:] != ".SWC" and file_name[-4:] != ".swc":
        warnings.warn("Input file must be a .swc or .SWC file, skipping")
        return None

    # Read SWC file
    swc_list = read_swc(pj(infolder, file_name))

    # Get total length of segment type (for max step size)
    coord_list_out = prep_coord_dict(swc_list, types_keep, keep_disconnect=False)
    if coord_list_out is None:
        return None

    sample_pts_out = sample_n_pts(coord_list_out[0], coord_list_out[1], coord_list_out[2], types_keep,
                                  goal_num_pts, min_step_change, max_iters, verbose)
    if sample_pts_out is None:
        return None

    if sample_pts_out[1] == goal_num_pts:
        # sample_network, pos = sample_network_step(coord_list_out[0], coord_list_out[1], sample_pts_out[2], types_keep)
        sample_network = sample_network_step(coord_list_out[0], coord_list_out[1], sample_pts_out[2], types_keep)
        geo_dist_mat = squareform(nx.algorithms.shortest_paths.dense.floyd_warshall_numpy(sample_network))
        return geo_dist_mat
    else:
        return None


def save_geodesic(file_name, infolder, outfolder, types_keep=(0, 1, 2, 3, 4),
                  goal_num_pts=50, min_step_change=1e-7,
                  max_iters=50, verbose=False):
    """
    Sample points from SWC file, compute geodesic distance (networkx graph distance) between
    points, save distance in vector format

    Args:
        file_name (string): SWC file name (including .swc)
        infolder (string): path to folder containing SWC file
        outfolder (string): path to output folder to save distance vectors
        types_keep (tuple,list): list of SWC neuron part types to sample points from
            by default, uses only 1 (soma), 2 (axon), 3 (basal dendrite), 4 (apical dendrite)
        goal_num_pts (integer): number of points to sample
        min_step_change (float): stops while loop from infinitely trying closer and closer step sizes
        max_iters (integer): maximum number of iterations of while loop
        verbose (boolean): if true, will print step size information for each search iteration

    Returns:
        Boolean success of sampling points from this SWC file
    """
    geo_dist_mat = get_geodesic(file_name, infolder, types_keep, goal_num_pts, min_step_change, max_iters, verbose)

    if geo_dist_mat is not None:
        np.savetxt(pj(outfolder, file_name[:-4] + "_dist.txt"), geo_dist_mat, fmt='%.8f')
        return "succeeded"
    else:
        return "failed"


def save_sample_pts_parallel(infolder, outfolder, types_keep=(0, 1, 2, 3, 4),
                             goal_num_pts=50, min_step_change=1e-7,
                             max_iters=50, num_cores=8, keep_disconnect=True):
    """
    Parallelize sampling the same number of points from all SWC files in a folder

    Args:
        infolder (string): path to folder containing SWC files. Only files ending in ".SWC" or ".swc" will be
        processed; other files will be ignored with a warning.
        outfolder (string): path to output folder to save *.csv files.
        types_keep (tuple,list): list of SWC neuron part types to sample points from
            by default, uses only 1 (soma), 2 (axon), 3 (basal dendrite), 4 (apical dendrite)
        goal_num_pts (integer): number of points to sample
        min_step_change (float): stops while loop from infinitely trying closer and closer step sizes
        max_iters (integer): maximum number of iterations of while loop
        num_cores (integer): number of processes to use for parallelization
        keep_disconnect (boolean): if True, will keep all branches from SWC. if False, will keep only connected to soma

    Returns:
        A list of Booleans which describe the success or failure of each file.
    """
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)
    arguments = [(file_name, infolder, outfolder, types_keep, goal_num_pts,
                  min_step_change, max_iters, keep_disconnect, False)
                 for file_name in os.listdir(infolder)]
    # start = time.time()
    with Pool(processes=num_cores) as pool:
        return(pool.starmap(save_sample_pts, arguments))
    # print(time.time() - start)


def save_geodesic_parallel(infolder, outfolder, types_keep=(0, 1, 2, 3, 4),
                           goal_num_pts=50, min_step_change=1e-7,
                           max_iters=50, num_cores=8):
    """
    Parallelize sampling and computing geodesic distance for the same number of points from all SWC files in a folder

    Args:
        infolder (string): path to folder containing SWC files
        outfolder (string): path to output folder to save distance vectors
        types_keep (tuple,list): list of SWC neuron part types to sample points from
            by default, uses only 1 (soma), 2 (axon), 3 (basal dendrite), 4 (apical dendrite)
        goal_num_pts (integer): number of points to sample
        min_step_change (float): stops while loop from infinitely trying closer and closer step sizes
        max_iters (integer): maximum number of iterations of while loop
        num_cores (integer): number of processes to use for parallelization

    Returns:
        A list of Booleans indicating the success or failure for each file in the folder
    """
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)
    arguments = [(file_name, infolder, outfolder, types_keep, goal_num_pts, min_step_change, max_iters, False)
                 for file_name in os.listdir(infolder)]
    # start = time.time()
    with Pool(processes=num_cores) as pool:
        return(pool.starmap(save_geodesic, arguments))
    # print(time.time() - start)
