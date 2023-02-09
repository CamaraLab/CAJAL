"""
Functions for sampling points from an SWC reconstruction of a neuron.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import math
import copy
import re
import numpy as np
from functools import partial
import numpy.typing as npt
from scipy.spatial.distance import euclidean, squareform, pdist
import networkx as nx
import warnings
from typing import List, Dict, Tuple, Optional, Iterable, Iterator, Set, \
    NewType, Callable, Literal, TypeVar
# from multiprocessing import Pool
from pathos.pools import ProcessPool
import os
from tinydb import TinyDB
from cajal.utilities import pj, write_tinydb_block, write_csv_block, Err, T

# Warning: Of 509 neurons downloaded from the Allen Brain Initiative
# database, about 5 had a height of at least 1000 nodes. Therefore on
# about 1% of test cases, recursive graph traversal algorithms will
# fail. For this reason we have tried to write our functions in an
# iterative style when possible.

@dataclass
class NeuronNode:
    sample_number : int
    structure_id : int
    coord_triple : tuple[float,float,float]
    radius : float
    parent_sample_number : int

@dataclass(eq=False)
class NeuronTree:
    root : NeuronNode
    child_subgraphs : List[NeuronTree]

    def __eq__(self,other):
        treelist0=[self]
        treelist1=[other]
        while bool(treelist0):
            assert(len(treelist0) == len(treelist1))
            for tree0, tree1 in zip(treelist0,treelist1):
                if tree0.root != tree1.root:
                    return False
                if len(tree0.child_subgraphs) != len(tree1.child_subgraphs):
                    return False
            treelist0 = [tree for child_tree in treelist0 for tree in child_tree.child_subgraphs ]
            treelist1 = [tree for child_tree in treelist1 for tree in child_tree.child_subgraphs ]
        return (not bool(treelist1))

# Convention: The *first element* of the SWC forest is always the
# component containing the soma.
SWCForest = List[NeuronTree]

def read_swc_node_dict(file_path : str) -> dict[int,NeuronNode]:
    r"""
    Read the swc file at `file_path` and return a dictionary mapping sample numbers \
    to their associated nodes.

    :param file_path: A path to an \*.swc file. \
    The only validation performed on the file's contents is to ensure that each line has \
    at least seven whitespace-separated strings.
    :return: A dictionary whose keys are sample numbers taken from \
    the first column of an SWC file and whose values are NeuronNodes.
    """
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

def topological_sort(
        nodes : dict[int,NeuronNode]) -> tuple[SWCForest,dict[int,NeuronTree]]:
    """
    Given a dictionary mapping (integer) sample numbers to NeuronNodes,
    construct the SWCForest representing the graph.

    :param nodes: A dictionary which directly represents the original SWC file, \
    in the sense that it sends sample_number to the node associated to that sample_number. \

    :return:
    #. the SWCForest encoded by the nodes dictionary
    #. a dictionary which associates to each sample number `x` the NeuronTree `tree` in the \
    SWCForest such that `tree.root.sample_number == x`.
    """
    components : list[NeuronTree] = []
    placed_trees : dict[int, NeuronTree] = {}

    for key in nodes:
        stack : list[int] = []
        while ((current_node := nodes[key]).parent_sample_number != -1) and \
              (key not in placed_trees):
            stack.append(key)
            key = current_node.parent_sample_number
        # Exit condition: Either key is in placed_trees, or parent_sample_number is -1, or both.
        if (current_node.parent_sample_number == -1 and key not in placed_trees):
            new_child_tree = NeuronTree(root=current_node,child_subgraphs=[])
            components.append(new_child_tree)
            placed_trees[key]=new_child_tree
        # Loop invariant:
        # key is in placed_trees.
        # current_node is placed_trees[key].root.
        while bool(stack):
            parent_tree = placed_trees[key]
            assert current_node is parent_tree.root
            child_key= stack.pop()
            new_child_node = nodes[child_key]
            new_child_tree = NeuronTree(root=new_child_node,child_subgraphs=[])
            placed_trees[child_key]=new_child_tree
            parent_tree.child_subgraphs.append(new_child_tree)
            key = child_key
            current_node = nodes[key]
        # At the end of this loop, all keys in the stack have been added to
        # placed_trees.
    return components, placed_trees

def read_swc(file_path : str) -> Tuple[SWCForest,Dict[int,NeuronTree]]:
    r"""
    Construct the graph (forest) associated to an SWC file.
    If the SWC file contains a marked soma node, forest[0] is a component containing a soma.\
    If the SWC file does not contain any soma node, forest[0] is the largest \
    component of the graph by number of nodes.

    An exception is raised if any line has fewer than seven whitespace \
    separated strings.
    
    :param file_path: A path to an \*.swc file.
    :return: (forest, lookup_table), where lookup_table \
          maps sample numbers for nodes to their positions in the forest.
    """
    nodes = read_swc_node_dict(file_path)
    components, tree_index = topological_sort(nodes)
    i=0
    while(i < len(components)):
        if components[i].root.structure_id == 1:
            swap_tree = components[i]
            components[i] = components[0]
            components[0] = swap_tree
            return components, tree_index
        i += 1

    # Otherwise, there is no soma node.
    return sorted(components, key = _num_nodes), tree_index
    

def cell_iterator(infolder : str) -> Iterator[Tuple[str,SWCForest]]:
    """
    Construct an iterator over all SWCs in a directory (all files ending in \*.swc or \*.SWC).
    
    :param infolder: A path to a folder containing SWC files.
    :return: An iterator over pairs (name, forest), where "name" is \
         the file root (everything before the period in the file name) \
         and "forest" is the forest contained in the SWC file.
    """
    file_names = [
        file_name for file_name in os.listdir(infolder)
        if os.path.splitext(file_name)[1] == ".swc"
        or os.path.splitext(file_name)[1] == ".SWC"
    ]
    cell_names = [os.path.splitext(file_name)[0] for file_name in file_names]
    all_files = [infolder + file_name for file_name in file_names] if infolder[-1] == '/' \
        else [infolder + '/' + file_name for file_name in file_names]
    cell_stream = (read_swc(file_name)[0] for file_name in all_files)
    return zip(cell_names, cell_stream)

def _is_soma_node(node : NeuronNode) -> bool:
    return (node.structure_id == 1)
    
def _has_soma_node(tree : NeuronTree) -> bool:
    return (_is_soma_node(tree.root) or any(map(_has_soma_node,tree.child_subgraphs)))

def _validate_one_soma(forest : SWCForest) -> bool:
    return(list(map(_has_soma_node, forest)).count(True) == 1)

def _filter_forest_rec(
        forest : SWCForest,
        keep_only : list[int],
) -> Tuple[SWCForest, SWCForest]:
    """
    Given an SWCForest `forest` and a list keep_only of integers to keep, \
    returns the smallest sub-forest of `forest` containing all nodes with \
    structure_id in keep_only.

    This forest is returned as the union of two disjoint lists. \
    The first list contains trees whose root node coincides with the root node of a tree \
    in `forest`; the second list contains trees whose root node is a proper descendant of \
    the root of a tree in `forest`.
    """
    # First, separate the forest into two lists, depending on whether their roots are in keep_only.
    subforest_root_in_keep_only : SWCForest = []
    subforest_root_not_in_keep_only : SWCForest = []
    for tree in forest:
        if tree.root.structure_id in keep_only:
            subforest_root_in_keep_only.append(tree)
        else:
            subforest_root_not_in_keep_only.append(tree)

    # I will call a "strong" subtree a subtree sharing the same root node.
    # A "weak" subtree is a subtree which does not share the same root node.
    ret_strong_subtree_list : SWCForest = []
    ret_weak_subtree_list : SWCForest = []
    for tree in subforest_root_in_keep_only:
        new_tree = NeuronTree(
                root = tree.root,
                child_subgraphs = [])
        ret_strong_subtree_list.append(new_tree)
        # The case where the tree has exactly one child is taken care of separately
        # instead of recursing in order to cut down on the total recursion depth of this algorithm.
        while len(tree.child_subgraphs) == 1:
            child_subtree = tree.child_subgraphs[0]
            if child_subtree.root.structure_id in keep_only:
                new_child_tree=NeuronTree(
                        root = copy.copy(child_subtree.root),
                        child_subgraphs = [])
                new_tree.child_subgraphs.append(new_child_tree)
                new_tree=new_child_tree    
            else:
                subforest_root_not_in_keep_only.append(child_subtree)
            tree=child_subtree
        if tree.root.structure_id not in keep_only or (not bool(tree.child_subgraphs)):
            continue
        strong_subtrees, weak_subtrees=\
            _filter_forest_rec(tree.child_subgraphs,keep_only)
        # Inductive assumption: Each tree in weak_subtrees has had parent_sample_number
        # changed to value -1; so we don't do that here.
        new_tree.child_subgraphs = strong_subtrees
        ret_weak_subtree_list += weak_subtrees
    # Weak subtrees with "good roots" are weak subtrees whose root node is in keep_only.
    weak_subtrees_good_roots : SWCForest = []
    # We will search through the elements with bad roots to find the good ones.
    stack = subforest_root_not_in_keep_only
    while(len(stack) > 0):
        # Inductive invariant: All elements in the stack have bad roots.
        last_tree = stack.pop()
        for child_tree in last_tree.child_subgraphs:
            if child_tree.root.structure_id in keep_only:
                weak_subtrees_good_roots.append(child_tree)
            else:
                stack.append(child_tree)
    # stack is empty.
    # Now we must turn to the problem of recursively filtering the subtrees of \
    #    weak_children_good_roots.
    for tree in weak_subtrees_good_roots:
        strong_subtrees, even_weaker_subtrees=\
            _filter_forest_rec(tree.child_subgraphs, keep_only)
        new_tree = NeuronTree(
                root = copy.copy(tree.root),
                child_subgraphs = strong_subtrees)
        new_tree.root.parent_sample_number = -1
        ret_weak_subtree_list.append(new_tree)
        ret_weak_subtree_list += even_weaker_subtrees
    return ret_strong_subtree_list, ret_weak_subtree_list

def filter_forest(forest : SWCForest, keep_only : list[int]) -> SWCForest:
    """
    Given an SWCForest `forest` and a list keep_only of integers to keep, \
    returns the smallest sub-forest of `forest` containing all nodes with \
    structure_id in keep_only.
    """
    a, b = _filter_forest_rec(forest, keep_only)
    return a + b

def _total_length(tree : NeuronTree) -> float:
    """
    Return the sum of lengths of all edges in the graph.
    """
    acc_length = 0.0
    treelist = [tree]
    while bool(treelist):
        for tree0 in treelist:
            for child_tree in tree0.child_subgraphs:
                acc_length += euclidean(
                    np.array(tree0.root.coord_triple),
                    np.array(child_tree.root.coord_triple))
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
                newlist.append((child_tree,
                                depth+\
                                euclidean(
                                    np.array(tree0.root.coord_triple),
                                    np.array(child_tree.root.coord_triple))))
        treelist=newlist
    return max_depth

def _discrete_depth(tree : NeuronTree) -> int:
    """
    Get the height of the tree in the unweighted or discrete sense, i.e. the \
    longest path from the root to any leaf measured in the number of edges.
    """
    depth : int = 0
    treelist = tree.child_subgraphs
    while bool(treelist):
        treelist = [child_tree for tree in treelist for child_tree in tree.child_subgraphs]
        depth +=1
    return depth

def _count_nodes_helper(node_a : NeuronNode, node_b: NeuronNode,
                        stepsize: float, offset : float) -> tuple[int,float]:
    """
    :param node_a: A node in the graph being sampled; the parent of node_b.
    :param node_b: A node in the graph being sampled; the child of node_a.
    :param stepsize: The sampling parameter controlling the distance between points \
    sampled from along the line segments of the graph.
    :param offset: The height above `node_a` of the last point which was sampled from the graph. \
    It is assumed that offset < stepsize; otherwise, it would be possible to choose a lower point \
    above node_a which is at least stepsize away from the one above it.
    
    :return: `num_intermediary nodes`, the number of points on the line segment from `a` to `b`
    which would be returned if one sampled uniformly from the least point already sampled above `a`
    to `b` at a distance of `stepsize` between sampled points.
    :return: `leftover`, the height of the least sampled point on the line segment from `a` to `b`
    after sampling according to this procedure.
    """
    cumulative = euclidean(np.array(node_a.coord_triple),
                           np.array(node_b.coord_triple))+offset
    num_intermediary_nodes = math.floor(cumulative / stepsize)
    leftover = cumulative - (num_intermediary_nodes * stepsize)
    return num_intermediary_nodes, leftover

def _count_nodes_at_given_stepsize(tree : NeuronTree, stepsize : float) -> int:
    """
    Count how many nodes will be returned if the user samples points uniformly
    from `tree`, \ starting at the root and adding all points at (geodesic)
    depth stepsize, 2 \* stepsize, 3 \* stepsize, and so on until we reach the
    end of the graph.

    :return: the number of points which would be sampled at this stepsize.
    """
    treelist = [(tree,0.0)]
    acc : int = 1
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
    """
    Returns the epsilon which will cause exactly `num_samples` points to be sampled, if
    the forest is sampled at `stepsize` epsilon.

    The user should ensure that len(forest) <= num_samples.
    """
    if len(forest) > num_samples:
        raise Exception("More trees in the forest than num_samples. \
        All root nodes of all connected components are returned as sample points, \
        so given this input it is impossible to get `num_samples` sample points. \
        Recommend discarding smaller trees. \
        ")

    max_depth = max([_weighted_depth(tree) for tree in forest])
    max_reps = 50
    counter = 0
    step_size = max_depth
    adjustment = step_size / 2
    while(counter < max_reps):
        num_nodes_this_step_size =\
            sum(map(lambda tree : _count_nodes_at_given_stepsize(tree, step_size), forest))
        if num_nodes_this_step_size < num_samples:
            step_size -= adjustment
        elif num_nodes_this_step_size > num_samples:
            step_size += adjustment
        else:
            return step_size
        adjustment /= 2
    raise Exception("Binary search timed out.")

def _branching_degree (forest : SWCForest) -> list[int]:
    """
    Compute the branching degrees of nodes in `forest`.
    The nodes are not indexed in any particular order, only by a breadth-first search,
    so it is primarily useful for computing summary statistics.

    :return: a list of integers containing the branching degree of each node in `forest`.
    """
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
    """
    Sample points uniformly throughout the forest, starting at the roots, \
     at the given step size.

    :return: a list of (x,y,z) coordinate triples, \
    represented as numpy floating point \
    arrays of shape (3,). The list length depends (inversely) \
    on the value of `step_size`.
    """
    sample_pts_list : list[npt.NDArray[np.float_]] = []
    for tree in forest:
        sample_pts_list.append(np.array(tree.root.coord_triple))
    treelist = [(tree, 0.0) for tree in forest]
    while bool(treelist):
        new_treelist : list[tuple[NeuronTree,float]] = []
        for tree, offset in treelist:
            root_triple = np.array(tree.root.coord_triple)
            for child_tree in tree.child_subgraphs:
                child_triple = np.array(
                    child_tree.root.coord_triple)
                dist = euclidean(root_triple, child_triple)
                assert(step_size >= offset)
                num_nodes,leftover = _count_nodes_helper(tree.root,child_tree.root,step_size,offset)
                spacing = np.linspace(start = step_size - offset,
                                      stop = dist - leftover,
                                      num= num_nodes,
                                      endpoint=True
                                      )
                assert(spacing.shape[0]==num_nodes)
                for x in spacing:
                    sample_pts_list.append(
                        (root_triple * x)+(child_triple *(1-x)))
                assert(leftover >= 0)
                assert(leftover < step_size)
                new_treelist.append((child_tree, leftover))
        treelist=new_treelist
    return sample_pts_list

def icdm_euclidean(forest : SWCForest, num_samples : int) -> npt.NDArray[np.float_]:
    """
    Compute the (Euclidean) intracell distance matrix for the forest, \
    with n sample points.
    :param forest: The cell to be sampled.
    :param num_samples: How many points to be sampled.
    :return: A condensed distance matrix of length n\* (n-1)/2.
    """
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
    """
    Convert a NeuronTree to a WeightedTree. A node in a WeightedTree does not contain \
    a coordinate triple, a radius, a structure_id, or a parent sample number.

    Instead, it contains a direct pointer to its parent, a list of its children,
    and (if it is a child node) the weight of the edge between the child and its parent.

    In forming the WeightedTree, any node with both a parent and exactly one child is eliminated, \
    and the parent and the child are joined directly by a single edge whose weight is the sum of \
    the two original edge weights. This reduces the number of nodes without affecting the \
    geodesic distances between points in the graph.

    :param tree: A NeuronTree to be converted into a WeightedTree.
    :return: The WeightedTree corresponding to the original NeuronTree.
    """
    
    treelist = [tree]
    depth :int = 0
    wt = WeightedTreeRoot(subtrees=[])
    correspondence_dict : dict[int,WeightedTree] = { tree.root.sample_number : wt }
    while bool(treelist):
        depth +=1
        new_treelist : list[NeuronTree] = []
        for tree in treelist:
            wt_parent = correspondence_dict[tree.root.sample_number]
            root_triple = np.array(tree.root.coord_triple)
            for child_tree in tree.child_subgraphs:
                child_triple = np.array(child_tree.root.coord_triple)
                dist = euclidean(child_triple,root_triple)
                while len(child_tree.child_subgraphs) == 1:
                    child_tree=child_tree.child_subgraphs[0]
                    new_triple=np.array(child_tree.root.coord_triple)
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

def node_type_counts_tree(tree : NeuronTree) -> dict[int,int]:
    """
    Return a dictionary whose keys are all structure_id's in `tree` and whose values are \
    the multiplicities with which that node type occurs.
    """
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

def node_type_counts_forest(forest : SWCForest) -> dict[int,int]:
    """
    Return a dictionary whose keys are all structure_id's in `forest` and whose values are \
    the multiplicities with which that node type occurs.
    """

    node_counts : dict[int,int]= {}
    for tree in forest:
        tree_node_counts = _node_type_counts_tree(tree)
        for key in tree_node_counts:
            if key in node_counts:
                node_counts[key]+=tree_node_counts[key]
            else:
                node_counts[key]=tree_node_counts[key]                
    return node_counts

def num_nodes(tree : NeuronTree) -> int:
    """
    Count the nodes in `tree.`
    """
    type_count_dict = _node_type_counts_tree(tree)
    return sum(type_count_dict[key] for key in type_count_dict)

def _sample_at_given_stepsize_wt(
        tree : WeightedTreeRoot,
        stepsize : float) -> list[tuple[WeightedTree,float]]:
    """
    Starting from the root of `tree`, sample points along `tree` at a geodesic distance \
    of `stepsize` \
    from the root, 2 \* `stepsize` from the root, and so on until the end of the graph. \

    In our formulation, a point `p` lying on a line segment from `a` to `b` \
    (where `a` is the parent node and `b` is the child node) is represented by a pair \
    `(p_dist, b)`, where `p_dist` is the distance from `p` to `b`, or the height of `p` above `b` \
    in the graph.

    :return: A list of sample points `(h, b)`, where `b` is a node in `tree` and `h` is the \
    distance of the sample point above `b`. `h` is guaranteed to be less than the distance between \
    `a` and `b`. If `b` is the root node of its tree, `h` is guaranteed to be 0.
    """
    treelist : list[tuple[WeightedTree,float]] = [(tree,0.0)]
    master_list : list[tuple[WeightedTree,float]] = [(tree,0.0)]
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
    """
    :param wt: A node in a weighted tree.
    :return: The weighted distance between wt and the root of the tree.
    """
    
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

    :param wt1: A node in a weighted tree.
    :param h1: Represents a point `p1` which lies `h1` above `wt1` in the tree, along \
    the line segment connecting `wt1` to its parent. `h1` is assumed to be less than the \
    distance between `wt1` and `wt2`.
    :param wt2: A node in a weighted tree.
    :param h2: Represents a point `p2` which lies `h2` above `wt2` in the tree, along \
    the line segment connecting `wt2` to its parent.
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

def _depth_table(tree : NeuronTree) -> dict[int,int]:
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
    """
    Sample points uniformly throughout the body of `tree`, starting at \
    the root, returning a list of length `num_sample_pts`.

    :return: a list of pairs (wt, h), where `wt` is a node of `tree`, \
    and `h` is a floating point real number representing a point \
    `p` which lies a distance of `h` above `wt` on the line \
    segment between `wt` and its parent. If `wt` is a child node, \
    `h` is guaranteed to be less than the distance between `wt` \
    and its parent. If `wt` is a root, `h` is guaranteed to be zero.

    "Sample points uniformly" means that there is some scalar `step_size` \    such that a point `p` on a line segment of `tree` will be in the \
    return list iff its geodesic distance from the origin is an integer \
    multiple of `step_size.`.
    """
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
    """
    Compute the intracell distance matrix for `tree` using the geodesic metric.
    
    Sample `num_samples` many points uniformly throughout the body of `tree`, compute the \
    pairwise geodesic distance between all sampled points, and return the matrix of distances.
    
    :return: A numpy array, a "condensed distance matrix" in the sense of \
    :func:`scipy.spatial.distance.squareform`, i.e., an array of shape \
    (num_samples \* num_samples - 1/2, ). Contains the entries in the intracell geodesic distance \
    matrix for `tree` lying strictly above the diagonal.
    """
    pts_list = get_sample_pts_geodesic(tree,num_samples)
    dist_list = []
    for i in range(len(pts_list)):
        for j in range(i+1,len(pts_list)):
            wt1, h1 = pts_list[i]
            wt2, h2 = pts_list[j]
            dist_list.append(geodesic_distance(wt1,h1,wt2,h2))
            assert(len(dist_list) == math.comb(num_samples,2)-math.comb(num_samples-i,2) + j - i)
    return np.array(dist_list)

def compute_intracell_one(
        cell : SWCForest,
        metric: Literal["euclidean"] | Literal["geodesic"],
        types_keep: list[int] | Literal["keep_all"],
        sample_pts: int,
        keep_disconnect: bool
) -> npt.NDArray[np.float_]:
    """
    Compute the intracell distance matrix for `tree` using either the Euclidean or geodesic \
    metric, as appropriate.
    
    Sample `num_samples` many points uniformly throughout the body of `tree`, compute the \
    pairwise distance between all sampled points, and return the matrix of distances.

    :param cell: An SWCForest to sample points and compute an intracell distance matrix from.
    :param metric: Controls which notion of distance between points of the cell to use.
    :param types_keep: If a list of integers is given, filter the cell first to discard \
    all cells except those of the given type. Otherwise, if types_keep is "keep_all", no \
    filtering is performed.
    :param sample_pts: How many points to sample from the cell.
    :param keep_disconnect: If keep_disconnect is true, only the component of the forest \
    connected to the soma will be sampled from. Otherwise, if \keep_disconnect is false, \
    points will be sampled uniformly from all components of the cell. The flag `keep_disconnect` \
    is ignored if the user selects the geodesic metric, as there is not a notion of geodesic \
    distance between points lying in two distinct connected components.

    If the user selects the geodesic distance, and supplies a list of integers "types_keep" \
    which does not contain the soma, an exception will be raised, as in this case \
    it is not clear which connected component should be sampled from.
    
    :return: A numpy array, a "condensed distance matrix" in the sense of \
    :func:`scipy.spatial.distance.squareform`, i.e., an array of shape \
    (num_samples \* num_samples - 1/2, ). Contains the entries in the intracell geodesic distance \
    matrix for `tree` lying strictly above the diagonal.
    """

    cell_1 = cell if (keep_disconnect and metric == "euclidean") else [cell[0]]
    cell_2 = cell_1 if types_keep == "keep_all" else filter_forest(cell_1, types_keep)
    match metric:
        case "euclidean":
            return icdm_euclidean(cell_2, sample_pts)
        case "geodesic":
            assert cell_2[0].root.structure_id == 1
            if (type(types_keep) == list and 1 not in types_keep):
                raise Exception("Soma was filtered out. \
                Not clear which connected component should be sampled.")
            return icdm_geodesic(cell_2[0], sample_pts)

def _read_and_compute_intracell_one(
        fullpath : str,
        metric: Literal["euclidean"] | Literal["geodesic"],
        types_keep: list[int] | Literal["keep_all"],
        sample_pts: int,
        keep_disconnect: bool
) -> tuple[str,npt.NDArray[np.float_]]:    
    forest, _ = read_swc(fullpath)
    cell_name = os.path.splitext(os.path.split(fullpath)[1])[0]
    return (cell_name,compute_intracell_one(forest,metric,types_keep,sample_pts,keep_disconnect))

def read_preprocess_compute_euclidean(
        file_name : str,
        n_sample : int,
        preprocess : Callable[ [SWCForest], Err[T] | SWCForest]
)-> Err[T] | npt.NDArray[np.float_]:
    
    loaded_forest, _ = read_swc(file_name)
    forest= preprocess(loaded_forest)
    if isinstance(forest, Err):
        return forest
    return icdm_euclidean(forest,n_sample)
    
def read_preprocess_compute_geodesic(
        file_name : str,
        n_sample : int,
        preprocess : Callable[ [SWCForest], Err[T] | NeuronTree]
)-> Err[T] | npt.NDArray[np.float_]:
    loaded_forest, _ = read_swc(file_name)
    tree= preprocess(loaded_forest)
    if isinstance(tree,Err):
        return tree
    return icdm_geodesic(tree,n_sample)

def compute_and_save_intracell_all_euclidean(
        infolder: str,
        out_csv: str,
        preprocess : Callable[[SWCForest], Err[T] | SWCForest],
        n_sample: int,
        num_cores: int=8
) -> list[tuple[str,Err[T]]]:

    file_names = [
        file_name for file_name in os.listdir(infolder)
        if os.path.splitext(file_name)[1] == ".swc"
        or os.path.splitext(file_name)[1] == ".SWC"
    ]
    if infolder[-1] == '/':
        all_files = [infolder + file_name for file_name in file_names]
    else:
        all_files = [infolder + '/' + file_name for file_name in file_names]
    cell_names = [ os.path.splitext(file_name)[0] for file_name in file_names]
    # Takes a pair of strings (cell_name, file_path), and 
    # returns (cell_name, read_preprocess_compute_euclidean(file_path, n_sample, preprocess))
    # This is a bit convoluted but it is just meant to give a way of keeping the cell's name
    # or identifier hanging around with it as it passes through the system
    rpce : Callable[ [tuple[str,str]], tuple[str,Err[T] | npt.NDArray[np.float_]]]
    rpce =\
        lambda str_pair : (str_pair[0],
                           read_preprocess_compute_euclidean(
                               str_pair[1],
                               n_sample,
                               preprocess))
    pool = ProcessPool(nodes=num_cores)
    icdms : Iterator[tuple[str,Err[T] | npt.NDArray[np.float_]]]
    assert(len(cell_names)==len(all_files))
    icdms = pool.imap(rpce,zip(cell_names,all_files))
    failed_cells : list[tuple[str,Err[T]]]
    failed_cells = write_csv_block(out_csv, n_sample, icdms, 10)
    pool.close()
    pool.join()
    pool.clear()
    return failed_cells

def compute_and_save_intracell_all_geodesic(
        infolder: str,
        out_csv: str,
        preprocess : Callable[[SWCForest], Err[T] | NeuronTree],
        n_sample: int,
        num_cores: int=8
) -> list[tuple[str,Err[T]]]:

    file_names = [
        file_name for file_name in os.listdir(infolder)
        if os.path.splitext(file_name)[1] == ".swc"
        or os.path.splitext(file_name)[1] == ".SWC"
    ]
    if infolder[-1] == '/':
        all_files = [infolder + file_name for file_name in file_names]
    else:
        all_files = [infolder + '/' + file_name for file_name in file_names]
    cell_names = [ os.path.splitext(file_name)[0] for file_name in file_names]
    # Takes a pair of strings (cell_name, file_path), and 
    # returns (cell_name, read_preprocess_compute_geodesic(file_path, n_sample, preprocess))
    # This is a bit convoluted but it is just meant to give a way of keeping the cell's name
    # or identifier hanging around with it as it passes through the system
    rpce : Callable[ [tuple[str,str]], tuple[str,Err[T] | npt.NDArray[np.float_]]]
    rpce =\
        lambda str_pair : (str_pair[0],
                           read_preprocess_compute_geodesic(
                               str_pair[1],
                               n_sample,
                               preprocess))
    pool = ProcessPool(nodes=num_cores)
    icdms : Iterator[tuple[str,Err[T] | npt.NDArray[np.float_]]]
    assert(len(cell_names)==len(all_files))
    icdms = pool.imap(rpce,zip(cell_names,all_files))
    failed_cells : list[tuple[str,Err[T]]]
    failed_cells = write_csv_block(out_csv, n_sample, icdms, 10)
    pool.close()
    pool.join()
    pool.clear()
    return failed_cells


# def compute_and_save_intracell_all(
#         infolder: str,
#         out_csv: str,
#         metric: Literal["euclidean"] | Literal["geodesic"],
#     types_keep: list[int] | Literal["keep_all"],
#     n_sample: int = 50,
#     num_cores: int = 8,
#     keep_disconnect: bool = False
# ) -> List[str]:
#     r"""
#     For each swc file in infolder, sample n_sample many points from the\
#     neuron, evenly spaced, and compute the Euclidean or geodesic intracell\
#     matrix depending on the value of the argument `metric`. Write the \
#     resulting intracell distance matrices to a database file called `db_name.json`.

#     :param infolder: Directory of input \*.swc files.
#     :param out_csv: Output file to write to.
#     :param metric: Either "euclidean" or "geodesic"
#     :param types_keep: optional parameter, a list of node types to sample from.
#     :param n_sample: How many points to sample from each cell.
#     :param num_cores: the intracell distance matrices will be computed in parallel processes,\
#           num_cores is the number of processes to run simultaneously. Recommended to set\
#           equal to the number of cores on your machine.
#     :param keep_disconnect: If keep_disconnect is True, we sample from only the the nodes connected\
#           to the soma. If False, all nodes are sampled from. This flag is only relevant to the\
#           Euclidean distance metric, as the geodesic distance between points \
#           in different components is undefined.
#     :return: List of cell names for which sampling failed.
#     """
#     pool = ProcessPool(nodes=num_cores)
    
#     file_names = [
#         file_name for file_name in os.listdir(infolder)
#         if os.path.splitext(file_name)[1] == ".swc"
#         or os.path.splitext(file_name)[1] == ".SWC"
#     ]
#     if infolder[-1] == '/':
#         all_files = [infolder + file_name for file_name in file_names]
#     else:
#         all_files = [infolder + '/' + file_name for file_name in file_names]

#     # cell_iter = cell_iterator(infolder)
#     compute_icdm_fn = partial(
#         _read_and_compute_intracell_one,
#         metric = metric,
#         types_keep = types_keep,
#         sample_pts = n_sample,
#         keep_disconnect = keep_disconnect)
#     name_distmat_pairs = pool.imap(
#         compute_icdm_fn,
#         all_files,chunksize=5)
#     batch_size = 1000
#     failed_cells = write_csv_block(out_csv, name_distmat_pairs, batch_size)
#     pool.close()
#     pool.join()
#     pool.clear()
#     return failed_cells
