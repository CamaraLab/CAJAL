r"""
Definition of a NeuronNode, NeuronTree and SWCForest class for representing the internal contents \
of an \*.swc file. Basic functions for manipulating, examining, validating and \
filtering \*.swc files. A function for reading \*.swc files from memory.
"""
from __future__ import annotations

import os
import re
from copy import copy
from dataclasses import dataclass
from collections import deque
import csv

from typing import Callable, Iterator, Literal
import numpy as np
from scipy.spatial.distance import euclidean
from pathos.pools import ProcessPool

from .utilities import Err, T

@dataclass
class NeuronNode:
    r"""
    A NeuronNode represents the contents of a single line in an \*.swc file.
    """
    sample_number: int
    structure_id: int
    coord_triple: tuple[float, float, float]
    radius: float
    parent_sample_number: int


@dataclass(eq=False)
class NeuronTree:
    r"""
    A NeuronTree represents one connected component of the graph coded in an \*.swc file.
    """
    root: NeuronNode
    child_subgraphs: list[NeuronTree]

    def __eq__(self, other):
        treelist0 = [self]
        treelist1 = [other]
        while bool(treelist0):
            assert len(treelist0) == len(treelist1)
            for tree0, tree1 in zip(treelist0, treelist1):
                if tree0.root != tree1.root:
                    return False
                if len(tree0.child_subgraphs) != len(tree1.child_subgraphs):
                    return False
            treelist0 = [
                tree for child_tree in treelist0 for tree in child_tree.child_subgraphs
            ]
            treelist1 = [
                tree for child_tree in treelist1 for tree in child_tree.child_subgraphs
            ]
        return not bool(treelist1)


# Convention: The *first element* of the SWC forest is always the
# component containing the soma.
SWCForest = list[NeuronTree]


def read_swc_node_dict(file_path: str) -> dict[int, NeuronNode]:
    r"""
    Read the swc file at `file_path` and return a dictionary mapping sample numbers \
    to their associated nodes.

    :param file_path: A path to an \*.swc file. \
    The only validation performed on the file's contents is to ensure that each line has \
    at least seven whitespace-separated strings.
    :return: A dictionary whose keys are sample numbers taken from \
    the first column of an SWC file and whose values are NeuronNodes.
    """
    nodes: dict[int, NeuronNode] = {}
    with open(file_path, "r") as file:
        for line in file:
            if line[0] == "#":
                continue
            row = re.split(r"\s|\t", line.strip())[0:7]
            if len(row) < 7:
                raise TypeError(
                    "Row"
                    + line
                    + "in file"
                    + file_path
                    + "has fewer than seven whitespace-separated strings."
                )
            nodes[int(row[0])] = NeuronNode(
                sample_number=int(row[0]),
                structure_id=int(row[1]),
                coord_triple=(float(row[2]), float(row[3]), float(row[4])),
                radius=float(row[5]),
                parent_sample_number=int(row[6]),
            )
    return nodes


def topological_sort(
    nodes: dict[int, NeuronNode]
) -> tuple[SWCForest, dict[int, NeuronTree]]:
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
    components: list[NeuronTree] = []
    placed_trees: dict[int, NeuronTree] = {}

    for key in nodes:
        stack: list[int] = []
        while ((current_node := nodes[key]).parent_sample_number != -1) and (
            key not in placed_trees
        ):
            stack.append(key)
            key = current_node.parent_sample_number
        # Exit condition: Either key is in placed_trees, or parent_sample_number is -1, or both.
        if current_node.parent_sample_number == -1 and key not in placed_trees:
            new_child_tree = NeuronTree(root=current_node, child_subgraphs=[])
            components.append(new_child_tree)
            placed_trees[key] = new_child_tree
        # Loop invariant:
        # key is in placed_trees.
        # current_node is placed_trees[key].root.
        while bool(stack):
            parent_tree = placed_trees[key]
            assert current_node is parent_tree.root
            child_key = stack.pop()
            new_child_node = nodes[child_key]
            new_child_tree = NeuronTree(root=new_child_node, child_subgraphs=[])
            placed_trees[child_key] = new_child_tree
            parent_tree.child_subgraphs.append(new_child_tree)
            key = child_key
            current_node = nodes[key]
        # At the end of this loop, all keys in the stack have been added to
        # placed_trees.
    return components, placed_trees


def read_swc(file_path: str) -> tuple[SWCForest, dict[int, NeuronTree]]:
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
    i = 0
    while i < len(components):
        if components[i].root.structure_id == 1:
            swap_tree = components[i]
            components[i] = components[0]
            components[0] = swap_tree
            return components, tree_index
        i += 1

    # Otherwise, there is no soma node.
    return sorted(components, key=num_nodes), tree_index


def linearize(forest: SWCForest) -> list[NeuronNode]:
    """
    Linearize the SWCForest into a list of NeuronNodes where the sample number of each node is just
    its index in the list plus 1.

    :param forest: An SWCForest to be linearized.

    :returns: A list `linear` of NeuronNodes. The list `linear` represents a directed graph \
    which is isomorphic to `forest`; under this graph isomorphism, the xyz coordinates, \
    radius, and structure identifier will be preserved, but the fields `parent_sample_number` and \
    `sample_number` will not be. Instead, we will have `linear[k].sample_number==k+1` for \
    each index `k`. (This index shift is clearly error-prone with Python's zero-indexing \
    of lists, but it seems common in SWC files.)

    In addition to having "standardized" indices, this is a breadth-first linearization algorithm. \
    It is guaranteed that:

    #. The graph is topologically sorted in that parent nodes come before child nodes.
    #. Each component is contained in a contiguous region of the list, whose first element is \
    of course the root by (1.)
    #. Within each component, the nodes are organized by level, so that the first element is the \
    root, indices 2..n are the nodes at depth 1, indices n+1 .. m are the nodes at depth 2, \
    and so on.
    """

    ell: list[NeuronNode] = []
    counter: int = 0
    for top_level_tree in forest:
        new_node = NeuronNode(
            sample_number=counter + 1,
            structure_id=top_level_tree.root.structure_id,
            coord_triple=top_level_tree.root.coord_triple,
            radius=top_level_tree.root.radius,
            parent_sample_number=-1,
        )
        ell.append(new_node)
        child_trees = [
            (counter, child_tree) for child_tree in top_level_tree.child_subgraphs
        ]
        # queue is a queue of ordered pairs (index, tree) : tuple[int,NeuronTree]
        # where ell[index] is the parent node of tree.
        queue = deque(child_trees)
        counter += 1
        while bool(queue):
            parent_index: int
            tree: NeuronTree
            parent_index, tree = queue.popleft()
            node = tree.root
            ell.append(
                NeuronNode(
                    sample_number=counter + 1,
                    structure_id=node.structure_id,
                    coord_triple=node.coord_triple,
                    radius=node.radius,
                    parent_sample_number=parent_index + 1,
                )
            )
            child_trees = [(counter, child_tree) for child_tree in tree.child_subgraphs]
            queue.extend(child_trees)
            counter += 1
    return ell


def forest_from_linear(ell: list[NeuronNode]) -> SWCForest:
    """
    Convert a list of :class:`swc.NeuronNode`'s to a graph.

    :param ell: A list of :class:`swc.NeuronNode`'s where \
    ell[i].sample_number == i+1 for all i. It is assumed that `ell` is topologically sorted, \
    i.e., that parents are listed before their children, and that roots are marked by -1.

    :return: An SWCForest containing the contents of the graph.
    """
    treedict: dict[int, NeuronTree] = {}
    components: SWCForest = []
    for node in ell:
        new_tree = NeuronTree(root=node, child_subgraphs=[])
        treedict[node.sample_number - 1] = new_tree
        if node.parent_sample_number == -1:
            components.append(new_tree)
        else:
            parent_index = node.parent_sample_number - 1
            treedict[parent_index].child_subgraphs.append(new_tree)
    return components


def write_swc(outfile: str, forest: SWCForest) -> None:
    """
    Write `forest` to `outfile`. Overwrite whatever is in `outfile`.

    This function does not respect the sample numbers and parent sample numbers
    in `forest`. They will be renumbered so that the indices are contiguous and
    start at 1.

    :param outfile: An absolute path to the output file.
    """
    lin = linearize(forest)
    rows = (
        [
            node.sample_number,
            node.structure_id,
            node.coord_triple[0],
            node.coord_triple[1],
            node.coord_triple[2],
            node.radius,
            node.parent_sample_number,
        ]
        for node in lin
    )
    with open(outfile, "w", newline="") as out_swc:
        csvwriter = csv.writer(out_swc, delimiter=" ")
        csvwriter.writerow(
            [
                "#",
                "sample_number",
                "structure_id",
                "x",
                "y",
                "z",
                "radius",
                "parent_sample_number",
            ]
        )
        csvwriter.writerows(rows)


def cell_iterator(infolder: str) -> Iterator[tuple[str, SWCForest]]:
    r"""
    Construct an iterator over all SWCs in a directory (all files ending in \*.swc or \*.SWC).

    :param infolder: A path to a folder containing SWC files.
    :return: An iterator over pairs (name, forest), where "name" is \
         the file root (everything before the period in the file name) \
         and "forest" is the forest contained in the SWC file.
    """
    file_names = [
        file_name
        for file_name in os.listdir(infolder)
        if os.path.splitext(file_name)[1] == ".swc"
        or os.path.splitext(file_name)[1] == ".SWC"
    ]
    cell_names = [os.path.splitext(file_name)[0] for file_name in file_names]
    all_files = (
        [infolder + file_name for file_name in file_names]
        if infolder[-1] == "/"
        else [infolder + "/" + file_name for file_name in file_names]
    )
    cell_stream = (read_swc(file_name)[0] for file_name in all_files)
    return zip(cell_names, cell_stream)


def _is_soma_node(node: NeuronNode) -> bool:
    return node.structure_id == 1


def _has_soma_node(tree: NeuronTree) -> bool:
    return _is_soma_node(tree.root) or any(map(_has_soma_node, tree.child_subgraphs))


def _validate_one_soma(forest: SWCForest) -> bool:
    return list(map(_has_soma_node, forest)).count(True) == 1


def _filter_forest_to_good_roots(
    forest: SWCForest, test: Callable[[NeuronNode], bool]
) -> tuple[SWCForest, SWCForest]:
    """
    Terminological convention:
    If t is a tree, call a "strong subtree" of t any subtree which shares the
    same root node.  Otherwise t is a "weak subtree."

    Given an SWCForest `forest` and a boolean criterion `test`, return two lists, \
    together they contain the largest subforest of
    `forest` such that the roots of each tree in `forest` satisfy `test(tree.root)==True` or \
    otherwise `test(tree.root)` has a truthy value. The first list contains the strong subtrees,
    and the second list contains the weak subtrees.

    :return: Two lists, `strong_subtrees` and `weak_subtrees`, \
    of all subtrees of trees in `forest` satisfying the following criteria:
    - each tree `t` satisfies `test(t.root)==True`
    - no strict ancestor `t1` of `t` has `test(t1.root)==True`

    We do not guarantee that the descendants of each tree `t` satisfy the property `test`.
    Trees in `strong_subtrees` have the same root node as some tree in `forest`; trees in \
    `weak_subtrees` do not.
    """
    subforest_good_strong: list[NeuronTree] = []
    subforest_good_weak: list[NeuronTree] = []
    subforest_bad_root: list[NeuronTree] = []
    for tree in forest:
        if test(tree.root):
            subforest_good_strong.append(tree)
        else:
            subforest_bad_root.append(tree)
    while bool(subforest_bad_root):
        # Inductive invariant: All elements in subforest_bad_root have bad roots.
        last_tree = subforest_bad_root.pop()
        for child_tree in last_tree.child_subgraphs:
            if test(child_tree.root):
                subforest_good_weak.append(child_tree)
            else:
                subforest_bad_root.append(child_tree)
    return subforest_good_strong, subforest_good_weak


def _filter_tree_while_nonbranching(
    tree: NeuronTree, test: Callable[[NeuronNode], bool]
) -> tuple[NeuronTree, NeuronTree, NeuronTree]:
    """
    :param tree: A NeuronTree such that test(tree.root) is True.
    :param test: A filtering criterion for membership.

    :return:
    #. the first descendant `desc` of `tree` such that one of the following conditions holds:
    - `desc` does not have exactly one child
    - `desc` has one child, and `test(desc.child_subgraphs[0].root)` is False.
    `desc` is not necessarily a strict descendant of `tree`; we may have `desc == tree`.
    #. `treecopy`, which is a partial copy of the graph `tree` from the root down to and \
    including `desc`, but no further. `treecopy` has the graph structure of a linked list. All \
    nodes in `treecopy` satisfy `test`.
    #. `last`, which is a pointer to the last node in `treecopy`. \
    We have `last.root == desc.root`.

    This function is not recursive.
    """
    treecopy = NeuronTree(root=tree.root, child_subgraphs=[])
    desc = tree
    last = treecopy
    while len(desc.child_subgraphs) == 1 and test(
        (child_subtree := desc.child_subgraphs[0]).root
    ):
        desc = child_subtree
        new_child_tree = NeuronTree(root=copy(desc.root), child_subgraphs=[])
        last.child_subgraphs.append(new_child_tree)
        last = new_child_tree
    return desc, treecopy, last


def _filter_forest_rec(
    forest: SWCForest,
    test: Callable[[NeuronNode], bool],
) -> tuple[SWCForest, SWCForest]:
    """
    Given an SWCForest `forest` and a boolean-valued test, \
    returns the smallest sub-forest of `forest` containing all nodes passing the test. \

    This forest is returned as the union of two disjoint lists. \
    The first list contains trees whose root node coincides with the root node of a tree \
    in `forest`; the second list contains trees whose root node is a proper descendant of \
    the root of a tree in `forest`.
    """
    # First, separate the forest into two lists,
    # depending on whether their roots are in keep_only.
    (
        subforest_good_roots_strong,
        subforest_good_roots_weak,
    ) = _filter_forest_to_good_roots(forest, test)
    # I will call a "strong" subtree a subtree sharing the same root node.
    # A "weak" subtree is a subtree which does not share the same root node.
    ret_strong_subtree_list: SWCForest = []
    ret_weak_subtree_list: SWCForest = []
    for tree in subforest_good_roots_strong:
        # The case where the tree has exactly one child is taken care of separately
        # instead of recursing in order to cut down on the total recursion depth of this algorithm.
        desc, treecopy, last = _filter_tree_while_nonbranching(tree, test)
        ret_strong_subtree_list.append(treecopy)
        strong_subtrees, weak_subtrees = _filter_forest_rec(desc.child_subgraphs, test)
        # Inductive assumption: Each tree in weak_subtrees has had parent_sample_number
        # changed to value -1; so we don't do that here.
        last.child_subgraphs = strong_subtrees
        ret_weak_subtree_list += weak_subtrees

    # Now we must turn to the problem of recursively filtering the subtrees of \
    #    subforest_good_roots_weak.
    for tree in subforest_good_roots_weak:
        strong_subtrees, even_weaker_subtrees = _filter_forest_rec(
            tree.child_subgraphs, test
        )
        new_tree = NeuronTree(root=copy(tree.root), child_subgraphs=strong_subtrees)
        new_tree.root.parent_sample_number = -1
        ret_weak_subtree_list.append(new_tree)
        ret_weak_subtree_list += even_weaker_subtrees
    return ret_strong_subtree_list, ret_weak_subtree_list


def filter_forest(forest: SWCForest, test: Callable[[NeuronNode], bool]) -> SWCForest:
    """
    Given an SWCForest `forest` and a criterion for membership, returns the `filtered_forest` \
    of all nodes in the SWCForest passing `test` (i.e. with a truthy value for test)
    """
    list_a, list_b = _filter_forest_rec(forest, test)
    return list_a + list_b


def keep_only_eu(structure_ids : Container[int]) -> Callable[[SWCForest],SWCForest]:
    def filter_by_structure_ids(forest : SWCForest) -> SWCForest:
        return filter_forest(
            forest,
            lambda node : operator.contains(structure_ids, node.structure_id))
    
    return filter_by_structure_ids


def keep_only_geo(structure_ids : Container[int]) -> Callable[[SWCForest], NeuronTree]:
    def filter_by_structure_ids(forest : SWCForest) -> Err[T] | NeuronTree:
        return filter_forest(
            forest,
            lambda node : operator.contains(structure_ids, node.structure_id))[0]
    return filter_by_structure_ids


def keep_only_geodesic(structure_ids : Container[int]) -> Callable[[NeuronNode],bool]:
    return lambda node : operator.contains(structure_ids, node.structure_i)d


def total_length(tree: NeuronTree) -> float:
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
                    np.array(child_tree.root.coord_triple),
                )
        treelist = [
            child_tree for tree0 in treelist for child_tree in tree0.child_subgraphs
        ]
    return acc_length


def weighted_depth(tree: NeuronTree) -> float:
    """
    Return the weighted depth/ weighted height of the tree,
    i.e., the maximal geodesic distance from the root to any other point.
    """
    treelist = [(tree, 0.0)]
    max_depth = 0.0

    while bool(treelist):
        newlist: list[tuple[NeuronTree, float]] = []
        for tree0, depth in treelist:
            if depth > max_depth:
                max_depth = depth
            for child_tree in tree0.child_subgraphs:
                newlist.append(
                    (
                        child_tree,
                        depth
                        + euclidean(
                            np.array(tree0.root.coord_triple),
                            np.array(child_tree.root.coord_triple),
                        ),
                    )
                )
        treelist = newlist
    return max_depth


def _discrete_depth(tree: NeuronTree) -> int:
    """
    Get the height of the tree in the unweighted or discrete sense, i.e. the \
    longest path from the root to any leaf measured in the number of edges.
    """
    depth: int = 0
    treelist = tree.child_subgraphs
    while bool(treelist):
        treelist = [
            child_tree for tree in treelist for child_tree in tree.child_subgraphs
        ]
        depth += 1
    return depth


def node_type_counts_tree(tree: NeuronTree) -> dict[int, int]:
    """
    Return a dictionary whose keys are all structure_id's in `tree` and whose values are \
    the multiplicities with which that node type occurs.
    """
    treelist = [tree]
    node_counts: dict[int, int] = {}
    while bool(treelist):
        for tree0 in treelist:
            if tree0.root.structure_id in node_counts:
                node_counts[tree0.root.structure_id] += 1
            else:
                node_counts[tree0.root.structure_id] = 1
        treelist = [
            child_tree for tree0 in treelist for child_tree in tree0.child_subgraphs
        ]
    return node_counts


def node_type_counts_forest(forest: SWCForest) -> dict[int, int]:
    """
    Return a dictionary whose keys are all structure_id's in `forest` and whose values are \
    the multiplicities with which that node type occurs.
    """

    node_counts: dict[int, int] = {}
    for tree in forest:
        tree_node_counts = node_type_counts_tree(tree)
        for key in tree_node_counts:
            if key in node_counts:
                node_counts[key] += tree_node_counts[key]
            else:
                node_counts[key] = tree_node_counts[key]
    return node_counts


def num_nodes(tree: NeuronTree) -> int:
    """
    Count the nodes in `tree.`
    """
    type_count_dict = node_type_counts_tree(tree)
    return sum(type_count_dict[key] for key in type_count_dict)


def _branching_degree(forest: SWCForest) -> list[int]:
    """
    Compute the branching degrees of nodes in `forest`.
    The nodes are not indexed in any particular order, only by a breadth-first search,
    so it is primarily useful for computing summary statistics.

    :return: a list of integers containing the branching degree of each node in `forest`.
    """
    treelist = forest
    branching_list: list[int] = []
    while bool(treelist):
        for tree in treelist:
            branching_list.append(len(tree.child_subgraphs))
        treelist = [
            child_tree for tree in treelist for child_tree in tree.child_subgraphs
        ]
    return branching_list


def _depth_table(tree: NeuronTree) -> dict[int, int]:
    """
    Return a dictionary which associates to each node the unweighted depth of that node in the tree.
    """
    depth: int = 0
    table: dict[int, int] = {}
    treelist = [tree]
    while bool(treelist):
        for tree0 in treelist:
            table[tree0.root.sample_number] = depth
        treelist = [
            child_tree for tree0 in treelist for child_tree in tree0.child_subgraphs
        ]
        depth += 1
    return table


def default_name_validate(filename: str) -> bool:
    if filename[0] == '.':
        return False
    return os.path.splitext(filename)[1].casefold() == ".swc".casefold()

def read_preprocess_save(
        infile_name: str,
        outfile_name: str,
        preprocess: Callable[[SWCForest], Err[T] | SWCForest],
) -> Err[T] | Literal["success"]:
    r"""
    Read the \*.swc file `file_name` from disk as an `SWCForest`.
    Apply the function `preprocess` to the forest. If preprocessing returns an error,\
    return that error. \
    Otherwise, write the preprocessed swc to outfile and return the string "success".
    """
    loaded_forest, _ = read_swc(infile_name)
    tree = preprocess(loaded_forest)
    if isinstance(tree, Err):
        return tree
    write_swc(outfile_name,tree)
    return "success"

def get_filenames(
        infolder : str,
        name_validate : Callable[[str], bool]) -> tuple[list[str],list[str]]:
    """
    Get a list of all files in infolder. Filter the list by name_validate. \

    :return: a pair of lists (cell_names, file_paths), where `file_paths` are the paths  \
    to cells we want to sample from, and `cell_names[i]` is the substring of `file_paths[i]` \
    containing only the file name, minus the extension; i.e., if  file_paths[i] is
    "/home/jovyan/files/abc.swc" then cell_names[i] is "abc".
    """

    file_names = [file_name for file_name in os.listdir(infolder) if name_validate(file_name)]
    file_paths = [os.path.join(infolder, file_name) for file_name in file_names]
    cell_names = [os.path.splitext(file_name)[0] for file_name in file_names]
    return (cell_names,file_paths)

def batch_filter_and_preprocess(
        infolder: str,
        outfolder : str,
        preprocess: Callable[[SWCForest], Err[T] | SWCForest],
        err_log : str,
        parallel_processes: int,
        suffix : str,
        name_validate : Callable[[str], bool] = default_name_validate
) -> None:
    r"""
    Get the set of files in infolder. Filter down to the filenames which pass the test
    `name_validate`, which is responsible for filtering out any non-swc files.

    For the files in this filtered list, read them into memory as :class:`swc.SWCForest`'s.
    Apply the function `preprocess` to each forest. `preprocess` may return an error \
    (essentially just a message contained in an error wrapper) or a modified/transformed \
    SWCForest, i.e., certain nodes have been filtered out, or certain components of the graph \
    deleted. If `preprocess` returns an error, write the error to the given log file `err_log` \
    together with the name of the cell that caused the error. \
    
    Otherwise, if `preprocess` returns an SWCForest, write this SWCForest into the folder \
    `outfolder` with filename == cellname + suffix + '.swc'.

    :param infolder: Folder containing SWC files to process.
    :param outfolder: Folder where the results of the filtering will be written.
    :param err_log: A file name for a (currently nonexistent) \*.csv file. This file will \
    be written to with a list of all the cells which were rejected by \
    `preprocess` together with an explanation of why these cells could not be processed.

    :param parallel_processes: Run this many Python processes in parallel. \
    :param suffix: If a file in infolder has the name "abc.swc" then the corresponding file \
    written to outfolder will have the name "abc" + suffix + ".swc".
    :param name_validate: A function which identifies the files in `infolder` which \
    are \*.swc files. The default argument, `default_name_validate`, checks to see \
    whether the filename has file extension ".swc", case insensitive, and discards files \
    starting with '.', the marker for hidden files on Linux. The user may need \
    to write their own function to ensure that various kinds of backup /autosave files \
    and metadata files are not read into memory.
    """

    try:
        os.mkdir(outfolder)
    except OSError as error:
        # print(error)
        pass
    cell_names, file_paths = get_filenames(infolder, default_name_validate)
    
    def rps(str_pair : tuple[str,str]) -> Err[T] | Literal["success"]:
        cell_name, file_path = str_pair
        outpath = os.path.join(outfolder,cell_name+suffix+".swc")
        return read_preprocess_save(file_path,outpath,preprocess)
        
    pool = ProcessPool(nodes=parallel_processes)
    results = pool.imap(rps,zip(cell_names,file_paths))
    
    with open(err_log, 'w', newline='') as outfile:
        for cell_name, result in zip(cell_names,results):
            match result:
                case "success":
                    pass
                case Err(code):
                    outfile.write(cell_name + " " + str(code)+"\n")

    pool.close()
    pool.join()
    pool.clear()





