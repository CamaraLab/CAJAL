"""
Functions for sampling points from an SWC reconstruction of a neuron.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import math
import re
import numpy as np
from functools import partial
import numpy.typing as npt
from scipy.spatial.distance import euclidean, squareform, pdist
import networkx as nx
import warnings
from typing import List, Dict, Tuple, Optional, Iterable, Iterator, Set, NewType, Callable, Literal
# from multiprocessing import Pool
from pathos.pools import ProcessPool
import os
from tinydb import TinyDB
from cajal.utilities import pj, write_tinydb_block

# Warning: Of 509 neurons downloaded from the Allen Brain Initiative database,
# about 5 had a height of at least 1000 nodes. Therefore on about 1% of test
# cases, recursive graph traversal algorithms will fail. For this reason we
# have tried to write our functions in an iterative style when possible.

@dataclass
class NeuronNode:
    sample_number : int
    structure_id : int
    coord_triple : tuple[float,float,float]
    radius : float
    parent_sample_number : int

@dataclass
class NeuronTree:
    root : NeuronNode
    child_subgraphs : List[NeuronTree]

# Convention: The *first element* of the SWC forest is always the component containing the soma.
SWCForest = List[NeuronTree]

def _read_swc_node_dict(file_path : str) -> dict[int,NeuronNode]:
    nodes : dict[int,NeuronNode] = {}
    with open(file_path, "r") as f:
        for line in f:
            if line[0] == "#":
                continue
            row = re.split("\s|\t", line.strip())[0:7]
            if len(row) < 7:
                raise TypeError("Row" + line + "in file" + file_path +
                                "has fewer than seven whitespace-separated strings.")
            nodes[int(row[0])]=\
                NeuronNode(
                    sample_number = int(row[0]),
                    structure_id = int(row[1]),
                    coord_triple = (float(row[2]), float(row[3]), float(row[4])),
                    radius = float(row[5]),
                    parent_sample_number = int(row[6])
                )
    return nodes

def _topological_sort_rec(
        index: int,
        nodes : dict[int,NeuronNode],
        tree : dict[int, NeuronTree],
        components: list[NeuronTree]) -> None:

    if index in tree:
        return
    parent_sample_number = nodes[index].parent_sample_number
    if parent_sample_number == -1:
        components.append(NeuronTree(root=nodes[index],child_subgraphs=[]))
        tree[index]=components[-1]
        return
    _topological_sort_rec(parent_sample_number,nodes,tree,components)
    tree[parent_sample_number].child_subgraphs.append(
        NeuronTree(root=nodes[index],child_subgraphs=[]))
    tree[index]=tree[parent_sample_number].child_subgraphs[-1]
    return
    
def _read_swc_typed(file_path : str) -> Tuple[SWCForest,Dict[int,NeuronTree]]:
    """
    Construct the graph (forest) associated to an SWC file, and return it together with a\
    dictionary mapping sample numbers for nodes to their positions in the graph.
    """
    nodes = _read_swc_node_dict(file_path)
    tree : Dict[int,NeuronTree] = {}
    components : List[NeuronTree] = []
    a_soma_node : Optional[int] = None
    for index in nodes:
        if nodes[index].structure_id == 1:
            a_soma_node = index
            break
    if a_soma_node is None:
        raise Exception("No soma nodes in SWC file.")
    _topological_sort_rec(a_soma_node,nodes,tree,components)
    
    for index in nodes:
        _topological_sort_rec(index,nodes,tree,components)
    return components, tree

def cell_iterator(infolder : str) -> Iterator[Tuple[str,SWCForest]]:
    file_names = [
        file_name for file_name in os.listdir(infolder)
        if os.path.splitext(file_name)[1] == ".swc"
        or os.path.splitext(file_name)[1] == ".SWC"
    ]
    cell_names = [os.path.splitext(file_name)[0] for file_name in file_names]
    if infolder[-1] == '/':
        all_files = [infolder + file_name for file_name in file_names]
    else:
        all_files = [infolder + '/' + file_name for file_name in file_names]
    cell_stream = map(lambda file_name : _read_swc_typed(file_name)[0], all_files)
    return zip(cell_names, cell_stream)

def _is_soma_node(node : NeuronNode) -> bool:
    return (node.structure_id == 1)
    
def _has_soma_node(tree : NeuronTree) -> bool:
    return (_is_soma_node(tree.root) or any(map(_has_soma_node,tree.child_subgraphs)))

def _validate_one_soma(forest : SWCForest) -> bool:
    return(list(map(_has_soma_node, forest)).count(True) == 1)
# all([list(map(has_soma_node, p[0])).count(True) == 1 for p in  all_neurons])
# print(one_component_w_soma)

def _filter_by_node_type_iterative(tree : NeuronTree, keep_only : List[int]) -> NeuronTree:
    """
    Modify tree in-place to eliminate all but the nodes whose structure_id is in keep_only.
    It is expected that the root of the tree is in keep_only; otherwise an exception is raised.
    (It is inconvenient to have to deal with the empty tree as a special case.)
    """
    if tree.root.structure_id not in keep_only:
        raise Exception("Root of tree is not in keep_only.")
    treelist = [tree]
    while bool(treelist):
        for tree0 in treelist:
            tree0.child_subgraphs = [child_tree for child_tree in tree0.child_subgraphs
                                     if child_tree.root.structure_id in keep_only]
        treelist = [child_tree for tree0 in treelist for child_tree in tree0.child_subgraphs]
    return tree

def _filter_forest(forest : SWCForest, keep_only : list[int]) -> SWCForest:
    return [ _filter_by_node_type_iterative(tree, keep_only) for tree in forest if tree.root.structure_id in keep_only]

def _filter_by_node_type_recursive(trees : SWCForest, keep_only : List[int]) -> SWCForest:
    new_tree_list =\
        [ (NeuronTree(root = tree.root,
                      child_subgraphs = _filter_by_node_type_recursive(tree.child_subgraphs, keep_only))
           if tree.root.structure_id in keep_only else None) for tree in trees]
    return [tree for tree in new_tree_list if tree is not None]

def _total_length(tree : NeuronTree) -> float:
    """
    Return the sum of lengths of all edges in the graph.
    """
    acc_length = 0.0
    treelist = [tree]
    while bool(treelist):
        for tree0 in treelist:
            for child_tree in tree0.child_subgraphs:
                acc_length += math.dist(tree0.root.coord_triple,child_tree.root.coord_triple)
        treelist = [child_tree for tree0 in treelist for child_tree in tree0.child_subgraphs]
    return acc_length

def _weighted_depth(tree : NeuronTree) -> float:
    """
    Return the weighted depth/ weighted height of the tree,
    i.e., the maximal geodesic distance from the root to any other point.
    """
    treelist = [(tree,0.0)]
    max_depth = 0.0
    
    while bool(treelist):
        newlist : list[tuple[NeuronTree,float]] = []
        for tree0, depth in treelist:
            if depth > max_depth:
                max_depth = depth
            for child_tree in tree0.child_subgraphs:
                newlist.append((child_tree,depth+math.dist(tree0.root.coord_triple,child_tree.root.coord_triple)))
        treelist=newlist
    return max_depth
    
# def _total_length(tree : NeuronTree) -> float:
#     return sum(map(lambda child_tree :
#                    math.dist(tree.root.coord_triple,child_tree.root.coord_triple) +
#                    _total_length(child_tree), tree.child_subgraphs))

def _discrete_depth(tree : NeuronTree) -> int:
    """
    Get the height of the tree in the graph-theoretic sense, i.e. the \
    longest path from the root to any other node (where length is understood in \
    the unweighted, integer-valued sense)
    """
    depth : int = 0
    treelist = tree.child_subgraphs
    while bool(treelist):
        treelist = [child_tree for tree in treelist for child_tree in tree.child_subgraphs]
        depth +=1
    return depth

def _count_nodes_helper(node_a : NeuronNode, node_b: NeuronNode,
                        stepsize: float, offset : float) -> tuple[int,float]:
    cumulative = math.dist(node_a.coord_triple, node_b.coord_triple)+offset
    num_intermediary_nodes = math.floor(cumulative / stepsize)
    leftover = cumulative - (num_intermediary_nodes * stepsize)
    return num_intermediary_nodes, leftover

def _count_nodes_at_given_stepsize(tree : NeuronTree, stepsize : float) -> int:
    treelist = [(tree,0.0)]
    acc : int = 0
    while bool(treelist):
        new_treelist : list[tuple[NeuronTree,float]] =[]
        for tree0, offset in treelist:
            for child_tree in tree0.child_subgraphs:
                nodes, new_offset = _count_nodes_helper(tree0.root,child_tree.root,stepsize,offset)
                acc += nodes
                new_treelist.append((child_tree,new_offset))
        treelist=new_treelist
    return acc

def _binary_stepwise_search(forest : SWCForest, num_samples : int) -> float:
    max_depth = max([_weighted_depth(tree) for tree in forest])
    max_reps = 50
    counter = 0
    step_size = max_depth
    adjustment = step_size / 2
    while(counter < max_reps):
        num_nodes_this_step_size = sum(map(lambda tree : _count_nodes_at_given_stepsize(tree, step_size), forest))
        if num_nodes_this_step_size < num_samples:
            step_size -= adjustment
        elif num_nodes_this_step_size > num_samples:
            step_size += adjustment
        else:
            return step_size
        adjustment /= 2
    raise Exception("Binary search timed out.")

def _branching_degree (forest : SWCForest) -> list[int]:
    treelist = forest
    branching_list : list[int] = []
    while bool(treelist):
        for tree in treelist:
            branching_list.append(len(tree.child_subgraphs))
        treelist = [child_tree for tree in treelist for child_tree in tree.child_subgraphs]
    return branching_list

def get_sample_pts_euclidean(
        forest : SWCForest,
        step_size : float) -> list[npt.NDArray[np.float_]]:
    # Samples points uniformly throughout the forest, starting at the roots, at
    # the given step size.
    sample_pts_list : list[npt.NDArray[np.float_]] = []
    for tree in forest:
        sample_pts_list.append(np.array(tree.root.coord_triple,dtype='f'))
    treelist = [(tree, 0.0) for tree in forest]
    while bool(treelist):
        new_treelist : list[tuple[NeuronTree,float]] = []
        for tree, offset in treelist:
            root_triple = np.array(tree.root.coord_triple,dtype='f')
            for child_tree in tree.child_subgraphs:
                child_triple = np.array(child_tree.root.coord_triple,dtype='f')
                dist = euclidean(child_triple,root_triple)
                assert(step_size >= offset)
                spacing = np.arange(start=step_size - offset,
                                    stop = dist,
                                    step = step_size) / dist
                for x in spacing:
                    sample_pts_list.append((root_triple * x)+(child_triple *(1-x)))
                d_plus_o = dist + offset
                y = d_plus_o - (step_size * math.floor(d_plus_o / step_size))
                assert(y >= 0)
                assert(y < step_size)
                new_treelist.append((child_tree, y))
        treelist=new_treelist
    return sample_pts_list

def icdm_euclidean(forest : SWCForest, num_samples : int) -> npt.NDArray[np.float_]:
    if len(forest) >= num_samples:
        pts : list[npt.NDArray[np.float_]] = []
        for i in range(num_samples):
            pts.append(np.array(forest[i].root.coord_triple))
    else:
        step_size = _binary_stepwise_search(forest, num_samples)
        pts = get_sample_pts_euclidean(forest, step_size)
    pts_matrix = np.stack(pts)
    return pdist(pts_matrix)  

@dataclass
class WeightedTreeRoot:
    subtrees : list[WeightedTreeChild]

@dataclass
class WeightedTreeChild :
    subtrees : list[WeightedTreeChild]
    depth : int
    unique_id : int
    parent : WeightedTree
    dist : float

WeightedTree = WeightedTreeRoot | WeightedTreeChild

def WeightedTree_of(tree : NeuronTree) -> WeightedTreeRoot :
    treelist = [tree]
    depth :int = 0
    wt = WeightedTreeRoot(subtrees=[])
    correspondence_dict : dict[int,WeightedTree] = { tree.root.sample_number : wt }
    while bool(treelist):
        depth +=1
        new_treelist : list[NeuronTree] = []
        for tree in treelist:
            wt_parent = correspondence_dict[tree.root.sample_number]
            root_triple = np.array(tree.root.coord_triple,dtype='f')
            for child_tree in tree.child_subgraphs:
                child_triple = np.array(child_tree.root.coord_triple,dtype='f')
                dist = euclidean(child_triple,root_triple)
                while len(child_tree.child_subgraphs) == 1:
                    child_tree=child_tree.child_subgraphs[0]
                    new_triple=np.array(child_tree.root.coord_triple,dtype='f')
                    dist += euclidean(child_triple,new_triple)
                    child_triple=new_triple
                new_wt = WeightedTreeChild(
                    subtrees=[],
                    depth=depth,
                    unique_id=child_tree.root.sample_number,
                    parent=wt_parent,
                    dist=dist)
                correspondence_dict[child_tree.root.sample_number]=new_wt
                wt_parent.subtrees.append(new_wt)
                new_treelist.append(child_tree)
        treelist=new_treelist
    return wt

def _node_type_counts(tree : NeuronTree) -> dict[int,int]:
    treelist = [tree]
    node_counts : dict[int,int] = {}
    while bool(treelist):
        for tree0 in treelist:
            if tree0.root.structure_id in node_counts:
                node_counts[tree0.root.structure_id] += 1
            else:
                node_counts[tree0.root.structure_id] =1
        treelist = [child_tree for tree0 in treelist for child_tree in tree0.child_subgraphs]
    return node_counts

def _num_nodes(tree : NeuronTree) -> int:
    type_count_dict = _node_type_counts(tree)
    return sum(type_count_dict[key] for key in type_count_dict)

def _sample_at_given_stepsize_wt(
        tree : WeightedTreeRoot,
        stepsize : float) -> list[tuple[WeightedTree,float]]:
    treelist : list[tuple[WeightedTree,float]] = [(tree,0.0)]
    master_list : list[tuple[WeightedTree,float]] = [(tree,0.0)]
    # acc : int = 0
    while bool(treelist):
        new_treelist : list[tuple[WeightedTree,float]] =[]
        for tree0, offset in treelist:
            for child_tree in tree0.subtrees:
                cumulative = child_tree.dist+offset
                num_intermediary_nodes = math.floor(cumulative/stepsize)
                leftover=cumulative-(num_intermediary_nodes * stepsize)
                for k in range(num_intermediary_nodes):
                    assert((cumulative - stepsize * (k+1)) <= child_tree.dist)
                    master_list.append((child_tree,cumulative - stepsize * (k+1)))
                new_treelist.append((child_tree,leftover))
        treelist=new_treelist
    return master_list

def _weighted_dist_from_root(wt : WeightedTree) -> float:
    x : float =0.0
    while isinstance(wt, WeightedTreeChild):
        x += wt.dist
        wt = wt.parent
    return x

def geodesic_distance(
        wt1 :WeightedTree, h1 : float,
        wt2 : WeightedTree, h2: float) -> float:
    """
    Let p1 be a point in a weighted tree which lies h1 above wt1.
    Let p2 be a point in a weighted tree which lies h2 above wt2.
    Return the geodesic distance between p1 and p2.
    """
    match wt1:
    # If wt1 is a root, we assume h1 is zero and p1 = wt1, otherwise the input is not sensible.
        case WeightedTreeRoot(_):
            assert(h1 == 0.0)
            return (_weighted_dist_from_root(wt2) - h2)
        case WeightedTreeChild(_, depth1, unique_id1, wt_parent1, d1):
        # Otherwise, suppose that wt1 is at an unweighted depth of depth1,
        # and that the distance between wt1 and its parent is d1.
            match wt2:
                case WeightedTreeRoot(_):
                # If wt2 is a root, then the approach is dual to what we have just done.
                    assert (h2 == 0.0)
                    return (_weighted_dist_from_root(wt_parent1) + d1 - h1)
                case WeightedTreeChild(_, depth2, unique_id2, wt_parent2, d2):
                # So let us consider the case where both wt1, wt2 are child nodes.
                    if unique_id1 == unique_id2:
                        return abs(h2 - h1)
                    # p1, p2 don't lie over the same child node.
                    # Thus, either one is an ancestor of the other
                    # (with one of wt1, wt2 strictly in between)
                    # or they have a common ancestor, and dist(p1, p2)
                    # is the sum of distances from p1, p2 to the common ancestor respectively.
                    # These three can be combined into the following problem:
                    # there is a minimal node in the weighted tree
                    # which lies above both wt1 and wt2, and dist(p1, p2) is the sum
                    # of the distances from p1, p2 respectively to that common minimal node.
                    # This includes the case where the minimal node is wt1 or wt2.
                    # To address these cases in a uniform way we use some cleverness with abs().
                    dist1 = -h1
                    dist2 = -h2
                    while depth1 > depth2:
                        dist1 += d1
                        match wt_parent1:
                            case WeightedTreeRoot(_):
                                raise Exception("Nodes constructed have wrong depth.")
                            case WeightedTreeChild(_, depth1, unique_id1, wt_parent1, d1):
                                pass
                    while depth2 > depth1:
                        dist2 += d2
                        match wt_parent2:
                            case WeightedTreeRoot(_):
                                raise Exception("Nodes constructed have wrong depth.")
                            case WeightedTreeChild( _, depth2, unique_id2, wt_parent2, d2):
                                pass
                    # Now we know that both nodes have the same height.
                    while unique_id1 != unique_id2:
                        dist1 += d1
                        dist2 += d2
                        match wt_parent1:
                            case WeightedTreeRoot(_):
                                assert(dist1 >= 0)
                                assert(dist2 >= 0)
                                assert isinstance(wt_parent2, WeightedTreeRoot)
                                return dist1 + dist2
                            case WeightedTreeChild(_, _, unique_id1, wt_parent1, d1):
                                pass
                        match wt_parent2:
                            case WeightedTreeRoot(_):
                                raise Exception("Nodes constructed have wrong depth.")
                            case WeightedTreeChild(_, _, unique_id2, wt_parent2, d2):
                                pass
                    return abs(dist1) + abs(dist2)

def _weighted_depth_wt(tree : WeightedTree) -> float:
    """
    Return the weighted depth/ weighted height of the tree,
    i.e., the maximal geodesic distance from the root to any other point.
    """
    treelist = [(tree,0.0)]
    max_depth = 0.0
    
    while bool(treelist):
        newlist : list[tuple[WeightedTree,float]] = []
        for tree0, depth in treelist:
            if depth > max_depth:
                max_depth = depth
            for child_tree in tree0.subtrees:
                newlist.append((child_tree,depth+child_tree.dist))
        treelist=newlist
    return max_depth

def depth_table(tree : NeuronTree) -> dict[int,int]:
    """
    Return a dictionary which associates to each node the unweighted depth of that node in the tree.
    """
    depth : int = 0
    table : dict[int,int] = {} 
    treelist = [tree]
    while bool(treelist):
        for tree in treelist:
            table[tree.root.sample_number]=depth
        treelist = [child_tree for tree in treelist for child_tree in tree.child_subgraphs]
        depth += 1
    return table

def get_sample_pts_geodesic(
        tree : NeuronTree,
        num_sample_pts : int) -> list[tuple[WeightedTree,float]]:
    wt = WeightedTree_of(tree)
    max_depth = _weighted_depth_wt(wt)
    max_reps = 50
    counter=0
    step_size=max_depth
    adjustment=step_size/2
    while(counter < max_reps):
        ell = _sample_at_given_stepsize_wt(wt,step_size)
        num_nodes_this_step_size = len(ell)
        if num_nodes_this_step_size < num_sample_pts:
            step_size -= adjustment
        elif num_nodes_this_step_size > num_sample_pts:
            step_size += adjustment
        else:
            return ell
        adjustment /= 2
    raise Exception("Binary search timed out.")

def icdm_geodesic(tree : NeuronTree, num_samples : int) -> npt.NDArray[np.float_]:
    pts_list = get_sample_pts_geodesic(tree,num_samples)
    dist_list = []
    for i in range(len(pts_list)):
        for j in range(i+1,len(pts_list)):
            wt1, h1 = pts_list[i]
            wt2, h2 = pts_list[j]
            dist_list.append(geodesic_distance(wt1,h1,wt2,h2))
            assert(len(dist_list) == math.comb(num_samples,2)-math.comb(num_samples-i,2) + j - i)
    return np.array(dist_list)

def _compute_intracell_one(
        cell : SWCForest,
        metric: Literal["euclidean"] | Literal["geodesic"],
        types_keep: list[int] | Literal["keep_all"],
        sample_pts: int,
        keep_disconnect: bool
) -> npt.NDArray[np.float_]:

    cell_1 = cell if (keep_disconnect and metric == "euclidean") else [cell[0]]
    cell_2 = cell_1 if types_keep == "keep_all" else _filter_forest(cell_1, types_keep)
    match metric:
        case "euclidean":
            return icdm_euclidean(cell_2, sample_pts)
        case "geodesic":
            return icdm_geodesic(cell_2[0], sample_pts)

# def _compute_intracell_all(
#         cell_iter : Iterator[SWCForest],
#         metric: Literal["euclidean"] | Literal["geodesic"],
#         types_keep: list[int] | Literal["keep_all"],
#         sample_pts: int = 50,
#         keep_disconnect: bool = False,
# ) -> Iterator[Tuple[npt.NDArray[np.float_]]]:
#     r"""
#     Compute intracell distances for all files in the given directory wrt the given metric.
#     Return an iterator over pairs (cell_name, maybe_cell_dists)

#     :param cell_iter: an iterator over SWC forests.
#     :param metric: Either "euclidean" or "geodesic" as appropriate.
#     :param pool: A pathos multiprocessing pool to do the work of sampling and computing distances.\
#         Assumed to be open.
#     :param types_keep: optional parameter, a list of node types to sample from.
#     :param sample_pts: How many points to sample from each cell.
#     :param keep_disconnect: If keep_disconnect is True, we sample from only the the nodes connected\
#           to the soma. If False, all nodes are sampled from. This flag is only relevant to the\
#           Euclidean distance metric, as the geodesic distance between points \
#           in different components is undefined.
#     """
#     cell_iter = cell_iterator(infolder)

#     # Replace the forest w a forest w a single tree - the first tree,
#     # which is understood to always contain the soma.
#     # Carry the cell names along.
#     strip_to_soma_component : Callable[[Tuple[str,SWCForest]],Tuple[str,SWCForest]]
#     strip_to_soma_component = lambda pair : (pair[0], [pair[1][0]])

    
#     filter_forest_in_place : Callable[[Tuple[str,SWCForest]],Tuple[str,SWCForest]]
#     filter_forest_in_place = lambda pair : (pair[0], _filter_forest(pair[1],types_keep))

#     match metric:
#         case "euclidean":
#             if not keep_disconnect:
               
#                 cell_iter = map(strip_to_soma_component, cell_iter)
#             if type(types_keep) == list:

#                 cell_iter = map(filter_forest_in_place , cell_iter)
#             icdm_euc : Callable[[Tuple[str,SWCForest]],Tuple[str,np.NDArray[np.float_]]]
#             icdm_euc = lambda pair : (pair[0], icdm_euclidean(pair[1],num_samples))
#             icdm_iter = map(icdm_euc,cell_iter)
#         case "geodesic":
#             cell_iter = map(strip_to_soma_component, cell_iter)
#             if type(types_keep) == list:
#                 filter_forest_in_place : Callable[[Tuple[str,SWCForest]],Tuple[str,SWCForest]]
#                 filter_forest_in_place = lambda pair : (pair[0], _filter_forest(pair[1],types_keep))
#                 cell_iter = map(filter_forest_in_place , cell_iter)        
#             compute_pt_cloud : Callable[[str],Optional[npt.NDArray[np.float_]]]
#             compute_pt_cloud = functools.partial(get_sample_pts,
#             compute_pt_cloud = \
#                 lambda file_name : get_sample_pts(
#                     file_name,
#                     infolder,
#                     types_keep,
#                     sample_pts)
#             maybe_pt_clouds = pool.imap(
#                 compute_pt_cloud,
#                 filenames,
#                 chunksize=5)
#             compute_dist_mat : Callable[[Optional[npt.NDArray[np.float_]]],\
#                                         Optional[npt.NDArray[np.float_]]]
#             compute_dist_mat =\
#                 lambda maybe_cloud: None if maybe_cloud is None else pdist(maybe_cloud)
#             return(zip(cell_names,pool.imap(
#                 compute_dist_mat,
#                 maybe_pt_clouds,
#                 chunksize=1000
#             )))
#         case "geodesic":
#             compute_geodesic : Callable[[str], Optional[npt.NDArray[np.float_]]]
#             compute_geodesic =\
#                 lambda file_name: get_geodesic(file_name, infolder, types_keep, sample_pts)
#             return(zip(cell_names,pool.imap(
#                 compute_geodesic,
#                 filenames,
#                 chunksize=1)))
#         case _:
#             raise Exception("Metric must be either Euclidean or geodesic.")

def _read_and_compute_intracell_one(
        fullpath : str,
        metric: Literal["euclidean"] | Literal["geodesic"],
        types_keep: list[int] | Literal["keep_all"],
        sample_pts: int,
        keep_disconnect: bool
) -> tuple[str,npt.NDArray[np.float_]]:
    forest, _ = _read_swc_typed(fullpath)
    cell_name = os.path.splitext(os.path.split(fullpath)[1])[0]
    return (cell_name,_compute_intracell_one(forest,metric,types_keep,sample_pts,keep_disconnect))

def compute_and_save_intracell_all_csv(
    infolder: str,
    out_csv: str,
    metric: Literal["euclidean"] | Literal["geodesic"],
    types_keep: list[int] | literal["keep_all"],
    n_sample: int = 50,
    num_cores: int = 8,
    keep_disconnect: bool = False
) -> List[str]:
    r"""
    For each swc file in infolder, sample n_sample many points from the\
    neuron, evenly spaced, and compute the Euclidean or geodesic intracell\
    matrix depending on the value of the argument `metric`. Write the \
    resulting intracell distance matrices to a database file called `db_name.json`.

    :param infolder: Directory of input \*.swc files.
    :param metric: Either "euclidean" or "geodesic"
    :param db_name: .json file to write the intracell distance matrices to. \
        It is assumed that db_name.json does not exist, or is empty.
    :param types_keep: optional parameter, a list of node types to sample from.
    :param n_sample: How many points to sample from each cell.
    :param num_cores: the intracell distance matrices will be computed in parallel processes,\
          num_cores is the number of processes to run simultaneously. Recommended to set\
          equal to the number of cores on your machine.
    :param keep_disconnect: If keep_disconnect is True, we sample from only the the nodes connected\
          to the soma. If False, all nodes are sampled from. This flag is only relevant to the\
          Euclidean distance metric, as the geodesic distance between points \
          in different components is undefined.
    """
    pool = ProcessPool(nodes=num_cores)
    
    file_names = [
        file_name for file_name in os.listdir(infolder)
        if os.path.splitext(file_name)[1] == ".swc"
        or os.path.splitext(file_name)[1] == ".SWC"
    ]
    if infolder[-1] == '/':
        all_files = [infolder + file_name for file_name in file_names]
    else:
        all_files = [infolder + '/' + file_name for file_name in file_names]

    # cell_iter = cell_iterator(infolder)
    compute_icdm_fn = partial(
        _read_and_compute_intracell_one,
        metric = metric,
        types_keep = types_keep,
        sample_pts = n_sample,
        keep_disconnect = keep_disconnect)
    name_distmat_pairs = pool.imap(
        compute_icdm_fn,
        all_files,chunksize=5)
    batch_size = 1000
    failed_cells = write_csv_block(out_csv, name_distmat_pairs, batch_size)
    pool.close()
    pool.join()
    pool.clear()
    return failed_cells

def compute_and_save_intracell_all(
    infolder: str,
    db_name: str,
    metric: Literal["euclidean"] | Literal["geodesic"],
    types_keep: list[int] | literal["keep_all"],
    n_sample: int = 50,
    num_cores: int = 8,
    keep_disconnect: bool = False
) -> List[str]:
    r"""
    For each swc file in infolder, sample n_sample many points from the\
    neuron, evenly spaced, and compute the Euclidean or geodesic intracell\
    matrix depending on the value of the argument `metric`. Write the \
    resulting intracell distance matrices to a database file called `db_name.json`.

    :param infolder: Directory of input \*.swc files.
    :param metric: Either "euclidean" or "geodesic"
    :param db_name: .json file to write the intracell distance matrices to. \
        It is assumed that db_name.json does not exist, or is empty.
    :param types_keep: optional parameter, a list of node types to sample from.
    :param n_sample: How many points to sample from each cell.
    :param num_cores: the intracell distance matrices will be computed in parallel processes,\
          num_cores is the number of processes to run simultaneously. Recommended to set\
          equal to the number of cores on your machine.
    :param keep_disconnect: If keep_disconnect is True, we sample from only the the nodes connected\
          to the soma. If False, all nodes are sampled from. This flag is only relevant to the\
          Euclidean distance metric, as the geodesic distance between points \
          in different components is undefined.
    """
    pool = ProcessPool(nodes=num_cores)
    output_db = TinyDB(db_name)
    file_names = [
        file_name for file_name in os.listdir(infolder)
        if os.path.splitext(file_name)[1] == ".swc"
        or os.path.splitext(file_name)[1] == ".SWC"
    ]
    if infolder[-1] == '/':
        all_files = [infolder + file_name for file_name in file_names]
    else:
        all_files = [infolder + '/' + file_name for file_name in file_names]

    # cell_iter = cell_iterator(infolder)
    compute_icdm_fn = partial(
        _read_and_compute_intracell_one,
        metric = metric,
        types_keep = types_keep,
        sample_pts = n_sample,
        keep_disconnect = keep_disconnect)
    name_distmat_pairs = pool.imap(
        compute_icdm_fn,
        all_files,chunksize=5)
    batch_size = 1000
    failed_cells = write_tinydb_block(output_db, name_distmat_pairs, batch_size)
    pool.close()
    pool.join()
    pool.clear()
    return failed_cells

def _read_swc(file_path: str) -> List[List[str]]:
    r"""
    Reads an SWC file and returns a list of the non-comment lines, split by spaces into tokens.

    The SWC file should conform to the documentation here: \
    http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html

    In particular, all rows should be either a comment starting with the character "#" or \
    should have at least seven strings separated by whitespace.

    read_swc(file_path)[i] is the i-th non-comment row, split into a list of strings by whitespace.

    If there are fewer than seven whitespace-separated tokens in the i-th row, an error is raised.

    If there are greater than seven whitespace-separated tokens in the i-th row, \
    the first seven tokens are kept and the rest discarded.

    The seventh token is assumed to be the parent node index and to be -1 if the node has no parent.

    read_swc expects the rows of the graph to be in topologically sorted \
    order (parents before children) If this is not satisfied, read_swc raises an exception.
    In particular, the first node must be the root of the tree, and its parent has index -1.

    Args:
        file_path (string): absolute path to SWC file

    Returns:
        list of vertex rows from SWC file, where each vertex row is a list of eight strings.
    """
    vertices = []
    ids: Set[str] = set()
    ids.add("-1")
    with open(file_path, "r", encoding="utf-8") as swcfile:
        for line in swcfile:
            if line[0] == "#":
                continue
            row = re.split("\s|\t", line.strip())[0:7]
            if len(row) < 7:
                raise TypeError(
                    "Row"
                    + line
                    + "in file"
                    + file_path
                    + "has fewer than eight whitespace-separated strings."
                )
            if row[6] not in ids:
                raise ValueError(
                    "SWC parent nodes must be listed before the \
                        child node that references them. The node with index "
                    + row[0]
                    + " was accessed before its parent "
                    + row[6]
                )
            ids.add(row[0])
            vertices.append(row)
    return vertices

def _prep_coord_dict(
    vertices: List[List[str]],
    types_keep: Optional[Iterable[int]] = None,
    keep_disconnect=False,
) -> Tuple[List[List[str]], Dict[int, np.ndarray], float]:
    """
    This function does three different things.

    1. It filters the list "vertices" to return a sublist "vertices_keep".

       Call a node v "acceptable" if types_keep is None, or the type of v\
          is in types_keep, or t=="1"

       vertices_keep is the smallest sub-forest of vertices satisfying the following:
       - every node in the soma is in vertices_keep
       - if a node v is acceptable and v's parent is in vertices_keep, v is in vertices_keep
       - (if keep_disconnected == True) if a node v is acceptable and v has no parent, v is\
         in vertices_keep
       vertices_keep always contains the first vertex of vertices,
       and all vertices which belong to the soma.

    :param vertices: list of vertex rows from an SWC file
    :param types_keep: list of SWC neuron part types to sample points from.
                     By default, uses all points.
        keep_disconnect (boolean): If False, will only keep branches connected to the soma.
            If True, will keep all branches, including free-floating ones

    Returns:
        vertices_keep: list of rows from SWC file that are connected to the soma
        vertex_coords: dictionary of xyz coordinates for the ID of each vertex in vertices_keep
        total_length: sum of segment lengths from branches of kept vertices
    """
    # in case types_keep are numbers
    types_keep_strings: Optional[List[str]] = None
    if types_keep is not None:
        types_keep_strings = [str(x) for x in types_keep]

    def type_is_ok(typeid : str) -> bool:
        return (types_keep_strings is None) or (typeid in types_keep_strings) or (typeid == "1")

    vertices_keep: List[List[str]] = []
    vertex_coords: Dict[int, np.ndarray] = {}
    total_length: float = 0
    for v in vertices:
        this_id = int(v[0])
        this_coord = np.array((float(v[2]), float(v[3]), float(v[4])))
        pid = int(v[-1])

        if pid < 0:
            # If not keeping disconnected parts, only keep vertex without parent
            # if it has soma type or is first vertex
            if (
                v[1] == "1"
                or len(vertices_keep) == 0
                or (keep_disconnect and type_is_ok(v[1]))
            ):
                vertex_coords[this_id] = this_coord
                vertices_keep.append(v)
        elif pid in vertex_coords.keys() and type_is_ok(v[1]):
            # keep branch vertex if connected to soma root
            vertex_coords[this_id] = this_coord
            vertices_keep.append(v)
            seg_len = euclidean(vertex_coords[pid], this_coord)
            total_length += seg_len
        elif types_keep_strings is not None and v[1] in types_keep_strings:
            raise ValueError(
                "Vertex "
                + v[0]
                + " is of type "
                + v[1]
                + " which is in the list of types to keep, but its\
                parent may not be. CAJAL does not currently have a\
                strategy to deal with such SWC files. \
                Suggest cleaning the data or setting \
                types_keep = None."
            )

    return vertices_keep, vertex_coords, total_length

def _sample_pts_step(
    vertices: List[List[str]],
    vertex_coords: Dict[int, np.ndarray],
    step_size: float,
    types_keep: Optional[Iterable[int]] = None,
) -> Tuple[List[np.ndarray], int]:
    """
    Sample points at every set interval (step_size) along branches of neuron

    Args:
        vertices (list): list of rows from SWC file that are connected to the soma
        vertex_coords (dict): dictionary of xyz coordinates for the ID of \
            each vertex in vertices_keep
        step_size (float): even distance to sample points radially from soma
        types_keep (tuple,list): list of SWC neuron part types to sample points from
            by default, uses only 1 (soma), 2 (axon), 3 (basal dendrite), 4 (apical dendrite)

    Returns:
        sampled_pts_list: list of xyz coordinates for sampled points
        num_roots: number of connected parts of neuron
    """

    # The implementation of this function constructs a new graph (forest) T on
    # top of the input graph (forest), G, where G is sampled from the SWC
    # file. The nodes of T lie on the geometric line segments connecting the
    # nodes of G. Every node of T is of the form ax+by, where x and y are nodes
    # of G with x the parent of y, and a, b are real coefficients with a, b >=0
    # and a+b == 1. The parent node of a node in T is the next nearest node
    # along the path to the root of that tree. T is constructed so that the
    # geodesic distance between a parent and child node is exactly step_size.

    # The keys of vertex_dist are nodes in G.  The value vertex_dist[v] is a
    # float, which is the geodesic distance between v and the nearest node
    # above it in T, say x. Note that if step_size < dist(v,parent(v)), then x
    # will lie on the line segment between v and parent(v); however, if
    # step_size >> dist(v, parent(v)), then there may be several nodes of G
    # between v and x.
    vertex_dist: Dict[int, float] = {}

    # The list of nodes of T, represented as numpy float arrays of length 3.
    sampled_pts_list: List[np.ndarray] = []

    # The number of connected components of the forest.
    num_roots: int = 0

    types_keep_strings: Optional[List[str]] = None
    # in case types_keep are numbers
    if types_keep is not None:
        types_keep_strings = (
            [str(x) for x in types_keep]
            if isinstance(types_keep, Iterable)
            else [str(types_keep)]
        )

    # loop through list of vertices, sampling points from edge of vertex to parent
    for v in vertices:
        this_id = int(v[0])
        this_coord = np.array((float(v[2]), float(v[3]), float(v[4])))
        pid = int(v[-1])
        if pid == -1:
            num_roots += 1
            vertex_dist[this_id] = 0
            sampled_pts_list.append(this_coord)
            continue
        seg_len = euclidean(vertex_coords[pid], this_coord)
        pts_dist = np.arange(step_size, seg_len + vertex_dist[pid], step_size)
        if (types_keep_strings is None or v[1] in types_keep_strings) and len(
            pts_dist
        ) > 0:
            pts_dist = pts_dist - vertex_dist[pid]
            new_dist = seg_len - pts_dist[-1]
            new_pts = [
                vertex_coords[pid] + (this_coord - vertex_coords[pid]) * x / seg_len
                for x in pts_dist
            ]
            if types_keep_strings is None or v[1] in types_keep_strings:
                sampled_pts_list.extend(new_pts)
            vertex_dist[this_id] = new_dist
        else:
            vertex_dist[this_id] = vertex_dist[pid] + seg_len
    return sampled_pts_list, num_roots


def _sample_n_pts(
    vertices: List[List[str]],
    vertex_coords: Dict[int, np.ndarray],
    total_length: float,
    types_keep: Optional[Iterable[int]] = None,
    goal_num_pts: int = 50,
    min_step_change: float = 1e-7,
    max_iters: int = 50,
    verbose: bool = False,
) -> Optional[Tuple[np.ndarray, float, int]]:
    """
    Use binary search to find step size between points that will sample the \
    required number of points.

    Args:
       * vertices (list): list of tokenized rows from SWC file that are connected to the soma
       * vertex_coords (dict): dictionary of xyz coordinates for the ID of each vertex in vertices
       * total_length (float): sum of segment lengths from branches of kept vertices
       * types_keep: list of SWC neuron part types to sample points from.
            By default, all points are kept. The standard structure identifiers are 1-4, \
            with 0 the key for "undefined";\
            indices greater than 5 are reserved for custom types. \
            types_keep = (0,1,2,3,4) should handle most files.
       * goal_num_pts (integer): number of points to sample
       * min_step_change (float): stops while loop from infinitely \
             trying closer and closer step sizes
       * max_iters (integer): maximum number of iterations of while loop
       * verbose (boolean): If true, will print step size information for each search iteration.

    Returns:
       * sampled_pts: array of xyz coordinates of sampled points
       * step_size: step size that samples required number of points
       * i: number of iterations to reach viable step size
    """

    num_pts = 0
    min_step_size = 0.0
    max_step_size = total_length
    prev_step_size = max_step_size
    step_size = (min_step_size + max_step_size) / 2.0
    i = 0
    while (
        num_pts != goal_num_pts
        and abs(step_size - prev_step_size) > min_step_change
        and i < max_iters
    ):
        i += 1
        sampled_pts_list, num_roots = _sample_pts_step(
            vertices, vertex_coords, step_size, types_keep
        )
        if num_roots > goal_num_pts:
            warnings.warn(
                "More connected components in neuron than points to sample, skipping"
            )
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
        raise Exception(
            "Sampled 0 points from neuron, could be too large of min_step_change, \
             or types_keep does not include values in second column of SWC files"
        )
    else:
        sampled_pts = np.array(sampled_pts_list)
        return sampled_pts, step_size, i


def _sample_network_step(
    vertices: List[List[str]],
    vertex_coords: Dict[int, np.ndarray],
    step_size: float,
    types_keep: Optional[Iterable[int]] = None,
):
    """
    Sample points at every set interval (step_size) along branches of neuron, return networkx

    Args:
       * vertices (list): list of rows from SWC file that are connected to the soma
       * vertex_coords (dict): dictionary of xyz coordinates for \
           the ID of each vertex in vertices_keep
       * step_size (float): even distance to sample points radially from soma
       * types_keep (tuple,list): list of SWC neuron part types to sample points from \
            by default, uses only 1 (soma), 2 (axon), 3 (basal dendrite), 4 (apical dendrite)

    Returns:
        graph: networkx graph of sampled points weighted by distance between points
    """
    vertex_dist: Dict[int, float] = {}
    num_roots = 0
    graph = nx.Graph()
    prev_pts: Dict[int, str] = {}  # Save last point before this one so can connect edge
    # pos = {}

    types_keep_strings: Optional[List[str]] = None
    # in case types_keep are numbers
    if types_keep is not None:
        types_keep_strings = (
            [str(x) for x in types_keep]
            if isinstance(types_keep, Iterable)
            else [str(types_keep)]
        )

    # loop through list of vertices, sampling points from edge of vertex to parent
    for v in vertices:
        this_id = int(v[0])
        this_coord = np.array((float(v[2]), float(v[3]), float(v[4])))
        pid = int(v[-1])
        if pid < 0:
            num_roots += 1
            vertex_dist[this_id] = 0
            graph.add_node(str(this_id))
            # pos[str(this_id)] = this_coord[:2]
            prev_pts[this_id] = str(this_id)
            continue
        seg_len = euclidean(vertex_coords[pid], this_coord)
        pts_dist = np.arange(step_size, seg_len + vertex_dist[pid], step_size)
        if (types_keep_strings is None or v[1] in types_keep_strings) and len(
            pts_dist
        ) > 0:
            pts_dist = pts_dist - vertex_dist[pid]
            new_dist = seg_len - pts_dist[-1]
            new_pts = [
                vertex_coords[pid] + (this_coord - vertex_coords[pid]) * x / seg_len
                for x in pts_dist
            ]
            new_pts_ids = [prev_pts[pid]] + [
                str(this_id) + "_" + str(x) for x in range(len(pts_dist))
            ]
            new_pts_len = [
                vertex_dist[pid] + euclidean(new_pts[0], vertex_coords[pid])
            ] + [euclidean(new_pts[i], new_pts[i - 1]) for i in range(1, len(new_pts))]
            # Add new points to graph, with edge weighted by euclidean to parent
            if types_keep_strings is None or v[1] in types_keep_strings:
                for i in range(1, len(new_pts_ids)):
                    graph.add_node(new_pts_ids[i])
                    # pos[new_pts_ids[i]] = new_pts[i - 1][:2]
                    graph.add_edge(
                        new_pts_ids[i - 1], new_pts_ids[i], weight=new_pts_len[i - 1]
                    )
            vertex_dist[this_id] = new_dist
            prev_pts[this_id] = new_pts_ids[-1]
        else:
            vertex_dist[this_id] = vertex_dist[pid] + seg_len
            prev_pts[this_id] = prev_pts[pid]
    return graph  # , pos


def get_sample_pts(
    file_name: str,
    infolder: str,
    types_keep: Optional[Iterable[int]] = None,
    goal_num_pts: int = 50,
    min_step_change: float = 1e-7,
    max_iters: int = 50,
    keep_disconnect: bool = False,
    verbose: bool = False,
) -> Optional[npt.NDArray[np.float_]]:
    """
    Given an SWC file, samples a given number of points from the body of the neuron and
    returns a point cloud of xyz coordinates.

    :param file_name: SWC file name (including ".swc" or ".SWC" extension)
    :param infolder: path to folder containing SWC file
    :param types_keep: list of SWC neuron part types to sample points from.\
          If types_keep is None, then all part types are sampled.
    :param goal_num_pts: number of points to sample.
    :param min_step_change: stops while loop from infinitely trying closer and \
          closer step sizes
    :param max_iters: maximum number of iterations of while loop
    :param keep_disconnect: If True, will keep all branches from SWC.\
              If False, will keep only connected to soma
    :param verbose: If True, will print step size information for each search iteration

    :return: :py:const:`None` , if either of these occur:

        * The file does not end with ".swc" or ".SWC".
        * There are more connected components in the sample than goal_num_pts.

        Otherwise, an array of xyz coordinates of sampled points.
    """

    if file_name[-4:] != ".SWC" and file_name[-4:] != ".swc":
        warnings.warn("Input file must be a .swc or .SWC file, skipping")
        return None

    # Read SWC file
    swc_list = _read_swc(pj(infolder, file_name))

    # Get total length of segment type (for max step size)
    coord_list_out = _prep_coord_dict(
        swc_list, types_keep, keep_disconnect=keep_disconnect
    )
    retval = _sample_n_pts(
        coord_list_out[0],
        coord_list_out[1],
        coord_list_out[2],
        types_keep,
        goal_num_pts,
        min_step_change,
        max_iters,
        verbose,
    )
    if retval is None:
        return None
    else:
        return retval[0]


# def _compute_and_save_sample_pts(file_name: str, infolder : str, outfolder : str,
#                     types_keep : Optional[Iterable[int]] = None,
#                     goal_num_pts : int = 50, min_step_change : float =1e-7,
#                     max_iters : int =50, keep_disconnect:bool=True,
#                     verbose:bool=False) -> bool:
#     """A wrapper function for get_sample_pts which saves the output as a
#         comma-separated-value text file.  The output filename is the same as
#         the input filename except that it ends in .csv instead of .swc.  The
#         output text file will contain goal_num_pts rows and three
#         comma-separated columns with the xyz values of the sampled points.  xyz
#         coordinates are specified to 16 places after the decimal.


#         Args:
#             * file_name (string): SWC file name (including ".swc" or ".SWC")
#             * infolder (string): path to folder containing SWC file
#             * outfolder (string): path to output folder to save CSVs
#             * types_keep (tuple,list): list of SWC neuron part types to sample points from.\
#                   If types_keep is None, all points are sampled.
#             * goal_num_pts (integer): number of points to sample
#             * min_step_change (float): stops while loop from infinitely trying closer and\
#                 closer step sizes
#             * max_iters (integer): maximum number of iterations of while loop
#             * keep_disconnect (boolean): if True, will keep all branches from SWC. \
#                      if False, will keep only those connected to soma
#             * verbose (boolean): If true, will print step size \
#                           information for each search iteration

#         Returns:
#             Boolean success of sampling points from this SWC file.

#     """

#     sample_pts_out = get_sample_pts(file_name, infolder, types_keep, goal_num_pts,
#                                     min_step_change, max_iters, keep_disconnect, verbose)

#     if sample_pts_out is None:
#         return False

#     if len(sample_pts_out) == goal_num_pts:
#         np.savetxt(pj(outfolder, file_name[:-4] + ".csv"),
#                    np.array(sample_pts_out), delimiter=",", fmt="%.16f")
#         return True
#     else:
#         return False


def get_geodesic(
    file_name: str,
    infolder: str,
    types_keep: Optional[Iterable[int]] = None,
    goal_num_pts: int = 50,
    min_step_change: float = 1e-7,
    max_iters: int = 50,
    verbose: bool = False,
) -> Optional[npt.NDArray[np.float_]]:
    """
    Sample points from a given SWC file, compute the geodesic distance (networkx graph distance) \
    between points, and return the matrix of pairwise geodesic distances between points in the cell.

    :param file_name: SWC file name (including ".swc" or ".SWC")
    :param infolder: path to folder containing SWC file
    :param types_keep: list of SWC neuron part types to sample points from. \
             By default, all points are kept.
    :param goal_num_pts: number of points to sample.
    :param min_step_change: stops while loop from infinitely trying closer and closer step sizes
    :param max_iters: maximum number of iterations of while loop
    :param verbose: If true, will print step size information for each search iteration
    :return: None, if either of these occur:

        * The file does not end with ".swc" or ".SWC".
        * There are more connected components in the sample than goal_num_pts.

        Otherwise, returns a numpy vector-form distance vector encoding the \
            intracell geodesic distance matrix.
    """

    if file_name[-4:] != ".SWC" and file_name[-4:] != ".swc":
        warnings.warn("Input file must be a .swc or .SWC file, skipping")
        return None

    # Read SWC file
    swc_list = _read_swc(pj(infolder, file_name))

    # Get total length of segment type (for max step size)
    coord_list_out = _prep_coord_dict(swc_list, types_keep, keep_disconnect=False)

    sample_pts_out = _sample_n_pts(
        coord_list_out[0],
        coord_list_out[1],
        coord_list_out[2],
        types_keep,
        goal_num_pts,
        min_step_change,
        max_iters,
        verbose,
    )
    if sample_pts_out is None:
        return None

    if goal_num_pts == sample_pts_out[0].shape[0]:
        sample_network = _sample_network_step(
            coord_list_out[0], coord_list_out[1], sample_pts_out[1], types_keep
        )
        geo_dist_mat = squareform(
            nx.algorithms.shortest_paths.dense.floyd_warshall_numpy(sample_network)
        )
        return geo_dist_mat
    else:
        return None


def compute_and_save_geodesic(
    file_name: str,
    infolder: str,
    outfolder: str,
    types_keep: Optional[Iterable[int]] = None,
    goal_num_pts: int = 50,
    min_step_change: float = 1e-7,
    max_iters: int = 50,
    verbose: bool = False,
) -> bool:
    """
    A wrapper for get_geodesic which writes the results to a text file. \
    If the input filename is "file_name.swc" then the output filename will \
    be "file_name_dist.txt".

    The file has a single column. The rows of the file are the distances \
    d(x_i,x_j) for x_i, x_j sample points and i < j.

    The distances are floating point real numbers specified to 8 places past the decimal.

    :param file_name: SWC file name (including .swc)
    :param infolder: (string): path to folder containing SWC file
    :param outfolder: path to output folder to save distance vectors
    :param types_keep: list of SWC neuron part types to sample points from. \
             By default, uses all.
    :param goal_num_pts: number of points to sample
    :param min_step_change: stops while loop from infinitely \
       trying closer and closer step sizes
    :param max_iters: maximum number of iterations of while loop
    :param verbose: If true, will print step size information for each search iteration

    :return: Boolean success of sampling points from this SWC file.
    """

    geo_dist_mat = get_geodesic(
        file_name,
        infolder,
        types_keep,
        goal_num_pts,
        min_step_change,
        max_iters,
        verbose,
    )

    if geo_dist_mat is not None:
        np.savetxt(
            pj(outfolder, file_name[:-4] + "_dist.txt"), geo_dist_mat, fmt="%.8f"
        )
        return True
    else:
        return False

# def compute_and_save_sample_pts_parallel(infolder : str, outfolder : str,
#                              types_keep: Optional[Iterable[int]]=None,
#                              goal_num_pts : int =50, min_step_change :float =1e-7,
#                              max_iters : int =50, num_cores :int =8,
#                              keep_disconnect: bool =True) -> Iterable[bool]:
#     """
#     Parallelize sampling the same number of points from all SWC files in a folder.

#     Args:

#         * infolder (string): path to folder containing SWC files.\
#           Only files ending in '.SWC' or '.swc' will be processed, \
#           other files will be ignored with a warning.
#         * outfolder (string): path to output folder to save .csv files.
#         * types_keep (tuple,list): list of SWC neuron part types to sample points from.
#         * goal_num_pts (integer): number of points to sample
#         * min_step_change (float): stops while loop from infinitely trying closer\
#               and closer step sizes
#         * max_iters (integer): maximum number of iterations of while loop
#         * num_cores (integer): number of processes to use for parallelization
#         * keep_disconnect (boolean): If True, will keep all branches from SWC.\
#               If False, will keep only connected to soma

#     Returns:
#         A lazy list of Booleans which describe the success or \
#           failure of sampling from each file.
#     """

#     if not os.path.exists(outfolder):
#         os.mkdir(outfolder)
#     arguments = [(file_name, infolder, outfolder, types_keep, goal_num_pts,
#                   min_step_change, max_iters, keep_disconnect, False)
#                  for file_name in os.listdir(infolder)]
#     # start = time.time()
#     with Pool(processes=num_cores) as pool:
#         return(pool.starmap(_compute_and_save_sample_pts, arguments))
#     # print(time.time() - start)


# def compute_and_save_geodesic_parallel(infolder : str,
#                     outfolder : str, types_keep : Optional[Iterable[int]]=None,
#                     goal_num_pts : int =50, min_step_change : float =1e-7,
#                     max_iters : int =50, num_cores: int =8) -> Iterable[bool]:
#     """
#     Parallelize sampling and computing geodesic distance for the same number of points \
#     from all SWC files in a folder

#     Args:
#         * infolder (string): path to folder containing SWC files
#         * outfolder (string): path to output folder to save distance vectors
#         * types_keep (tuple,list): list of SWC neuron part types to sample points from.\
#            By default, all parts will be used.
#         * goal_num_pts (integer): number of points to sample
#         * min_step_change (float): stops while loop from infinitely trying closer\
#              and closer step sizes
#         * max_iters (integer): maximum number of iterations of while loop
#         * num_cores (integer): number of processes to use for parallelization

#     Returns:
#         A lazy list of Booleans indicating the success or failure for each file in the folder
#     """

#     if not os.path.exists(outfolder):
#         os.mkdir(outfolder)
#     arguments = [(file_name, infolder, outfolder, types_keep,\
#                   goal_num_pts, min_step_change, max_iters, False)
#                  for file_name in os.listdir(infolder)]
#     # start = time.time()
#     with Pool(processes=num_cores) as pool:
#         return(pool.starmap(compute_and_save_geodesic, arguments))
#     # print(time.time() - start)

# def _euclidean_case(file_name) -> None:
#         sample =\
#             get_sample_pts(
#                 file_name,infolder,
#                 types_keep, num_sample_pts,
#                 keep_disconnect = keep_disconnect)
#         if sample is None:
#             dist_mats[file_name] = None
#         else:
#             dist_mats[file_name] = pdist(sample[0])

# def geodesic_case(file_name) -> None:
#     dist_mats[file_name] =\
#         get_geodesic(file_name, infolder, types_keep, num_sample_pts)


def _euclidean_helper(sample_pts: Optional[np.ndarray]):
    if sample_pts is None:
        return None
    else:
        return pdist(sample_pts)

def compute_intracell_parallel(
    infolder: str,
    metric: str,
    types_keep: Optional[Iterable[int]] = None,
    sample_pts: int = 50,
    num_cores: int = 8,
    keep_disconnect: bool = False,
) -> Dict[str, Optional[npt.NDArray[np.float_]]]:
    r"""
    For each swc file in `infolder`, sample `sample_pts` many points from the
    neuron, evenly spaced, and compute the Euclidean or geodesic intracell
    matrix depending on the value of the argument `metric`.

    :param infolder:Directory of input \*.swc files.
    :param metric: Either "euclidean" or "geodesic"
    :param types_keep: optional parameter, a list of node types to sample from
    :param num_cores: the intracell distance matrices will be computed in parallel processes,\
          num_cores is the number of processes to run simultaneously. Recommended to set\
          equal to the number of cores on your machine.
    :param keep_disconnect: If keep_disconnect is True, we sample from only the the nodes connected\
          to the soma. If False, all nodes are sampled from. This flag is only relevant to the\
          Euclidean distance metric, as the geodesic distance between points in \
          different components is undefined.

    :return: A dictionary `dist_mats`\
        mapping file names (strings) to their intracell distance matrices. If the\
        intracell distance matrix for file_name could not be computed, then\
        `dist_mats[file_name]` is 
    """

    file_names = os.listdir(infolder)
    dist_mats: Dict[str, Optional[npt.NDArray[np.float_]]] = {}

    match metric:
        case "euclidean":
            eu_arguments = [
                (file_name, infolder, types_keep, sample_pts, 1e-7, 50, False)
                for file_name in file_names
            ]
            with ProcessPool(processes=num_cores) as pool:
                samples = pool.starmap(get_sample_pts, eu_arguments)
                dist_mat_list = pool.map(_euclidean_helper, samples, 100)
        case "geodesic":
            ge_arguments = [
                (file_name, infolder, types_keep, sample_pts)
                for file_name in file_names
            ]
            with ProcessPool(processes=num_cores) as pool:
                dist_mat_list = list(pool.starmap(get_geodesic, ge_arguments))
    for i in range(len(file_names)):
        dist_mats[file_names[i]] = dist_mat_list[i]

    return dist_mats

# def _compute_intracell_all(
#     infolder: str,
#     metric: str,
#     pool: ProcessPool,
#     types_keep: Optional[Iterable[int]] = None,
#     sample_pts: int = 50,
#     keep_disconnect: bool = False,
# ) -> Iterator[Tuple[str,Optional[npt.NDArray[np.float_]]]]:
#     r"""
#     Compute intracell distances for all files in the given directory wrt the given metric.
#     Return an iterator over pairs (cell_name, maybe_cell_dists)

#     :param infolder: Directory of \*.swc files.
#     :param metric: Either "euclidean" or "geodesic" as appropriate.
#     :param pool: A pathos multiprocessing pool to do the work of sampling and computing distances.\
#         Assumed to be open.
#     :param types_keep: optional parameter, a list of node types to sample from.
#     :param sample_pts: How many points to sample from each cell.
#     :param keep_disconnect: If keep_disconnect is True, we sample from only the the nodes connected\
#           to the soma. If False, all nodes are sampled from. This flag is only relevant to the\
#           Euclidean distance metric, as the geodesic distance between points \
#           in different components is undefined.
#     """

#     filenames = [
#         file_name for file_name in os.listdir(infolder)
#         if os.path.splitext(file_name)[1] == ".swc"
#         or os.path.splitext(file_name)[1] == ".SWC"
#     ]
#     cell_names = [os.path.splitext(filename)[0] for filename in filenames]

#     match metric:
#         case "euclidean":
#             compute_pt_cloud : Callable[[str],Optional[npt.NDArray[np.float_]]]
#             compute_pt_cloud = \
#                 lambda file_name : get_sample_pts(
#                     file_name,
#                     infolder,
#                     types_keep,
#                     sample_pts)
#             maybe_pt_clouds = pool.imap(
#                 compute_pt_cloud,
#                 filenames,
#                 chunksize=5)
#             compute_dist_mat : Callable[[Optional[npt.NDArray[np.float_]]],\
#                                         Optional[npt.NDArray[np.float_]]]
#             compute_dist_mat =\
#                 lambda maybe_cloud: None if maybe_cloud is None else pdist(maybe_cloud)
#             return(zip(cell_names,pool.imap(
#                 compute_dist_mat,
#                 maybe_pt_clouds,
#                 chunksize=1000
#             )))
#         case "geodesic":
#             compute_geodesic : Callable[[str], Optional[npt.NDArray[np.float_]]]
#             compute_geodesic =\
#                 lambda file_name: get_geodesic(file_name, infolder, types_keep, sample_pts)
#             return(zip(cell_names,pool.imap(
#                 compute_geodesic,
#                 filenames,
#                 chunksize=1)))
#         case _:
#             raise Exception("Metric must be either Euclidean or geodesic.")

# def compute_and_save_intracell_all_backup(
#     infolder: str,
#     db_name: str,
#     metric: str,
#     n_sample: int = 50,
#     num_cores: int = 8,
#     types_keep: Optional[Iterable[int]] = None,
#     keep_disconnect: bool = False
# ) -> List[str]:

#     r"""
#     For each swc file in infolder, sample n_sample many points from the\
#     neuron, evenly spaced, and compute the Euclidean or geodesic intracell\
#     matrix depending on the value of the argument `metric`. Write the \
#     resulting intracell distance matrices to a database file called `db_name.json`.

#     :param infolder: Directory of input \*.swc files.
#     :param metric: Either "euclidean" or "geodesic"
#     :param db_name: .json file to write the intracell distance matrices to. \
#         It is assumed that db_name.json does not exist.
#     :param types_keep: optional parameter, a list of node types to sample from.
#     :param n_sample: How many points to sample from each cell.
#     :param num_cores: the intracell distance matrices will be computed in parallel processes,\
#           num_cores is the number of processes to run simultaneously. Recommended to set\
#           equal to the number of cores on your machine.
#     :param keep_disconnect: If keep_disconnect is True, we sample from only the the nodes connected\
#           to the soma. If False, all nodes are sampled from. This flag is only relevant to the\
#           Euclidean distance metric, as the geodesic distance between points \
#           in different components is undefined.

#     :return: a list of file names for which the sampling process failed.
#     """

#     pool = ProcessPool(nodes=num_cores)
#     output_db = TinyDB(db_name)
#     dist_mats = _compute_intracell_all(
#         infolder,
#         metric,
#         pool,
#         types_keep,
#         n_sample,
#         keep_disconnect)

#     batch_size = 1000
#     failed_cells = write_tinydb_block(output_db, dist_mats, batch_size)
#     pool.close()
#     pool.join()
#     pool.clear()
#     return failed_cells
