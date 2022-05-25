# Functions for sampling even points from an SWC reconstruction of a neuron

import re
import numpy as np
from run_gw import pj
from scipy.spatial.distance import euclidean, squareform
import networkx as nx
import warnings
from multiprocessing import Pool
import time
import os


def read_swc(file_path):
    """
    Read swc file into list of rows, check that file is sorted

    Args:
        file_path (string): path to SWC file

    Returns:
        list of vertex rows from SWC file
    """
    vertices = []
    ids = set()
    ids.add("-1")
    with open(file_path, "r") as f:
        for line in f:
            if line[0] == "#":
                continue
            row = re.split("\s|\t", line.strip())
            if row[-1] not in ids:
                warnings.warn("SWC parent nodes must be listed before the child node that references them")
                return None
            ids.add(row[0])
            vertices.append(row)
    return vertices


def prep_coord_dict(vertices, types_keep=(1, 2, 3, 4)):
    """
    Read through swc file list, get dictionary of vertex coordinates and total length of all segments

    Args:
        vertices (list): list of vertex rows from SWC file
        types_keep (tuple,list): list of SWC neuron part types to sample points from
            by default, uses only 1 (soma), 2 (axon), 3 (basal dendrite), 4 (apical dendrite)

    Returns:
        vertices_keep: list of rows from SWC file that are connected to the soma
        vertex_coords: dictionary of xyz coordinates for the ID of each vertex in vertices_keep
        total_length: sum of segment lengths from branches of kept vertices
    """
    vertices_keep = []
    vertex_coords = {}
    total_length = 0
    for v in vertices:
        this_id = int(v[0])
        this_coord = np.array((float(v[2]), float(v[3]), float(v[4])))
        pid = int(v[-1])
        if pid < 0 and v[1] == "1":
            # only keep soma origin vertices
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


def sample_pts_step(vertices, vertex_coords, step_size, types_keep=(1, 2, 3, 4)):
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


def sample_n_pts(vertices, vertex_coords, total_length, types_keep=(1, 2, 3, 4),
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
        step_size: step size which samples required number of points
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
    sampled_pts = np.array(sampled_pts_list)
    return sampled_pts, num_pts, step_size, i


def sample_network_step(vertices, vertex_coords, step_size, types_keep=(1, 2, 3, 4)):
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
    return graph # , pos


def save_sample_pts(file_name, infolder, outfolder, types_keep=(1, 2, 3, 4),
                    goal_num_pts=50, min_step_change=1e-7,
                    max_iters=50, verbose=False):
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
        verbose (boolean): if true, will print step size information for each search iteration

    Returns:
        Boolean success of sampling points from this SWC file
    """

    if file_name[-4:] != ".SWC" and file_name[-4:] != ".swc":
        warnings.warn("Input file must be a .swc or .SWC file, skipping")
        return False

    # Read SWC file
    swc_list = read_swc(pj(infolder, file_name))

    # Get total length of segment type (for max step size)
    coord_list_out = prep_coord_dict(swc_list, types_keep)
    if coord_list_out is None:
        return False

    sample_pts_out = sample_n_pts(coord_list_out[0], coord_list_out[1], coord_list_out[2], types_keep,
                                  goal_num_pts, min_step_change, max_iters, verbose)
    if sample_pts_out is None:
        return False

    if sample_pts_out[1] == goal_num_pts:
        np.savetxt("sampled_pts/" + outfolder + "/folder_" + infolder +
                   "_file_" + file_name[:-4] + ".csv", np.array(sample_pts_out[0]), delimiter=",", fmt="%.16f")
        return True
    else:
        return False


def save_sample_pts_wrapper(args):
    """
    Wraps save_sample_pts() so there is a function that only takes in a single list of args to call for parallelization
    """
    file_name, infolder, outfolder, types_keep, goal_num_pts, min_step_change, max_iters, verbose = args
    return save_sample_pts(file_name, infolder, outfolder, types_keep,
                           goal_num_pts, min_step_change,
                           max_iters, verbose)


def save_geodesic(file_name, infolder, outfolder, types_keep=(1, 2, 3, 4),
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

    if file_name[-4:] != ".SWC" and file_name[-4:] != ".swc":
        warnings.warn("Input file must be a .swc or .SWC file, skipping")
        return "failed"

    # Read SWC file
    swc_list = read_swc(pj(infolder, file_name))

    # Get total length of segment type (for max step size)
    coord_list_out = prep_coord_dict(swc_list, types_keep)
    if coord_list_out is None:
        return "failed"

    sample_pts_out = sample_n_pts(coord_list_out[0], coord_list_out[1], coord_list_out[2], types_keep,
                                  goal_num_pts, min_step_change, max_iters, verbose)
    if sample_pts_out is None:
        return "failed"

    if sample_pts_out[1] == goal_num_pts:
        # sample_network, pos = sample_network_step(coord_list_out[0], coord_list_out[1], sample_pts_out[2], types_keep)
        sample_network = sample_network_step(coord_list_out[0], coord_list_out[1], sample_pts_out[2], types_keep)
        geo_dist_mat = squareform(nx.algorithms.shortest_paths.dense.floyd_warshall_numpy(sample_network))
        np.savetxt("sampled_pts/" + outfolder + "/folder_" + infolder +
                   "_file_" + file_name[:-4] + "_dist.txt", geo_dist_mat, fmt='%.8f')
        return "succeeded"
    else:
        return "failed"


def save_geodesic_wrapper(args):
    """
    Wraps save_geodesic() so there is a function that only takes in a single list of params to call for parallelization
    """
    file_name, infolder, outfolder, types_keep, goal_num_pts, min_step_change, max_iters, verbose = args
    return save_geodesic(file_name, infolder, outfolder, types_keep,
                         goal_num_pts, min_step_change,
                         max_iters, verbose)


def save_sample_pts_parallel(infolder, outfolder, types_keep=(1, 2, 3, 4),
                             goal_num_pts=50, min_step_change=1e-7,
                             max_iters=50, num_cores=8):
    """
    Parallelize sampling the same number of points from all SWC files in a folder

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
        None
    """
    # os.mkdir("sampled_pts/"+params["outfolder"]+"_geodesic_"+str(goal_num_pts))
    arguments = [(file_name, infolder, outfolder, types_keep, goal_num_pts, min_step_change, max_iters, False)
                 for file_name in os.listdir(infolder)]
    pool = Pool(processes=num_cores)
    start = time.time()
    save_results = pool.map(save_sample_pts_wrapper, arguments)
    print(time.time() - start)


def save_geodesic_parallel(infolder, outfolder, types_keep=(1, 2, 3, 4),
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
        None
    """
    arguments = [(file_name, infolder, outfolder, types_keep, goal_num_pts, min_step_change, max_iters, False)
                 for file_name in os.listdir(infolder)]
    pool = Pool(processes=num_cores)
    start = time.time()
    save_results = pool.map(save_geodesic_wrapper, arguments)
    print(time.time() - start)
