"""
Defines a WeightedTree class, to represent the information relevant in an SWC from the \
geodesic point of view. Defines functions for manipulating and processing WeightedTrees.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Union

import numpy as np
from scipy.spatial.distance import euclidean

from .swc import NeuronTree


@dataclass
class WeightedTreeRoot:
    subtrees: list[WeightedTreeChild]
    structure_id: int


@dataclass
class WeightedTreeChild:
    subtrees: list[WeightedTreeChild]
    depth: int
    unique_id: int
    parent: WeightedTree
    dist: float
    structure_id: int


WeightedTree = Union[WeightedTreeRoot, WeightedTreeChild]


def WeightedTree_of(tree: NeuronTree) -> WeightedTreeRoot:
    """
    Convert a NeuronTree to a WeightedTree. A node in a WeightedTree does not contain \
    a coordinate triple, a radius, or a parent sample number.

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
    depth: int = 0
    wt_root = WeightedTreeRoot(subtrees=[], structure_id=tree.root.structure_id)
    correspondence_dict: dict[int, WeightedTree] = {tree.root.sample_number: wt_root}
    while bool(treelist):
        depth += 1
        new_treelist: list[NeuronTree] = []
        for tree0 in treelist:
            wt_parent = correspondence_dict[tree0.root.sample_number]
            root_triple = np.array(tree0.root.coord_triple)
            for child_tree in tree0.child_subgraphs:
                child_triple = np.array(child_tree.root.coord_triple)
                dist = euclidean(child_triple, root_triple)
                while len(child_tree.child_subgraphs) == 1:
                    child_tree = child_tree.child_subgraphs[0]
                    new_triple = np.array(child_tree.root.coord_triple)
                    dist += euclidean(child_triple, new_triple)
                    child_triple = new_triple
                new_wt = WeightedTreeChild(
                    subtrees=[],
                    depth=depth,
                    unique_id=child_tree.root.sample_number,
                    parent=wt_parent,
                    dist=dist,
                    structure_id=child_tree.root.structure_id,
                )
                correspondence_dict[child_tree.root.sample_number] = new_wt
                wt_parent.subtrees.append(new_wt)
                new_treelist.append(child_tree)
        treelist = new_treelist
    return wt_root


def weighted_dist_from_root(wt: WeightedTree) -> float:
    """
    :param wt: A node in a weighted tree.
    :return: The weighted distance between wt and the root of the tree.
    """

    x: float = 0.0
    while isinstance(wt, WeightedTreeChild):
        x += wt.dist
        wt = wt.parent
    return x


def weighted_depth_wt(tree: WeightedTree) -> float:
    """
    Return the weighted depth/ weighted height of the tree,
    i.e., the maximal geodesic distance from the root to any other point.
    """
    treelist = [(tree, 0.0)]
    max_depth = 0.0

    while bool(treelist):
        newlist: list[tuple[WeightedTree, float]] = []
        for tree0, depth in treelist:
            if depth > max_depth:
                max_depth = depth
            for child_tree in tree0.subtrees:
                newlist.append((child_tree, depth + child_tree.dist))
        treelist = newlist
    return max_depth
