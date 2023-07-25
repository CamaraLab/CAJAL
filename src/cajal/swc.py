r"""
Definition of a NeuronNode, NeuronTree and SWCForest class for representing the internal contents \
of an \*.swc file. Basic functions for manipulating, examining, validating and \
filtering \*.swc files. A function for reading \*.swc files from memory.
"""
from __future__ import annotations

import os
import sys
import operator
from copy import copy
from dataclasses import dataclass
from collections import deque
import csv
from typing import Callable, Iterator, Literal, Container, Optional

import numpy as np
from scipy.spatial.distance import euclidean
from pathos.pools import ProcessPool
import dill

from .utilities import Err, T

dill.settings["recurse"] = True


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

    def is_soma_node(self) -> bool:
        return self.structure_id == 1


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

    def __iter__(self) -> Iterator[NeuronTree]:
        """
        Iterate over all descendants of this tree in breadth-first order. \
        The first element of the iterator is just this tree. \
        Recommended to use these methods for large SWC files as naive recursion on \
        the graph structure may cause a stack overflow.
        """
        treelist = [self]
        while bool(treelist):
            new_treelist = []
            for tree in treelist:
                yield tree
                new_treelist += tree.child_subgraphs
            treelist = new_treelist

    def dfs(self) -> Iterator[NeuronTree]:
        """
        Iterate over all descendants of this tree in preorder traversal order. \
        The first element of the iterator is just this tree. \
        Recommended to use these methods for large SWC files as naive recursion on \
        the graph structure may cause a stack overflow.
        """
        stack = [self]
        while bool(stack):
            current_tree = stack.pop()
            for tree in reversed(current_tree.child_subgraphs):
                stack.append(tree)
            yield current_tree


if sys.version_info >= (3, 10):
    from typing import TypeAlias

    SWCForest: TypeAlias = list[NeuronTree]  # noqa: F821
else:
    SWCForest = list[NeuronTree]  # noqa: F821


def read_swc_node_dict(file_path: str) -> dict[int, NeuronNode]:
    r"""
    Read the swc file at `file_path` and return a dictionary mapping sample numbers \
    to their associated nodes.

    :param file_path: A full path to an \*.swc file. \
    The only validation performed on the file's contents is to ensure that each line has \
    at least seven whitespace-separated strings.

    :return: A dictionary whose keys are sample numbers taken from \
    the first column of an SWC file and whose values are NeuronNodes.
    """
    nodes: dict[int, NeuronNode] = {}
    with open(file_path, "r") as file:
        for n, line in enumerate(file):
            if line[0] == "#" or len(line.strip()) < 2:
                continue
            row = line.strip().split()[0:7]
            if len(row) < 7:
                raise TypeError(
                    "Row "
                    + str(n)
                    + " in file "
                    + file_path
                    + " has fewer than seven whitespace-separated strings."
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
    The forest is sorted by the number of nodes of the components

    An exception is raised if any line has fewer than seven whitespace \
    separated strings.

    :param file_path: A path to an \*.swc file.
    :return: (forest, lookup_table), where lookup_table \
          maps sample numbers for nodes to their positions in the forest.
    """
    nodes = read_swc_node_dict(file_path)
    components, tree_index = topological_sort(nodes)
    return sorted(components, key=num_nodes, reverse=True), tree_index


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
    if not (isinstance(forest, list)):
        raise TypeError("forest should be a list.")
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

    :return: An :class:`swc.SWCForest` containing the contents of the graph.
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


def default_name_validate(filename: str) -> bool:
    """
    If the file name starts with a period '.', the standard hidden-file marker on Linux,
    return False.
    Otherwise, return True if and only if the file ends in ".swc" (case-insensitive).
    """
    if filename[0] == ".":
        return False
    return os.path.splitext(filename)[1].casefold() == ".swc".casefold()


def cell_iterator(
    infolder: str, name_validate: Callable[[str], bool] = default_name_validate
) -> Iterator[tuple[str, SWCForest]]:
    r"""
    Construct an iterator over all SWCs in a directory (all files ending in \*.swc or \*.SWC).

    :param infolder: A path to a folder containing SWC files.
    :return: An iterator over pairs (name, forest), where "name" is \
         the file root (everything before the period in the file name) \
         and "forest" is the forest contained in the SWC file.
    """
    file_names = [
        file_name for file_name in os.listdir(infolder) if name_validate(file_name)
    ]
    cell_names = [os.path.splitext(file_name)[0] for file_name in file_names]
    all_files = (
        [infolder + file_name for file_name in file_names]
        if infolder[-1] == "/"
        else [infolder + "/" + file_name for file_name in file_names]
    )
    cell_stream = (read_swc(file_name)[0] for file_name in all_files)
    return zip(cell_names, cell_stream)


def has_soma_node(tree: NeuronTree) -> bool:
    """
    Returns true if any node is a soma node; else returns false.
    """

    def f(t: NeuronTree) -> bool:
        return t.root.is_soma_node()

    return any(map(f, tree))


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


def keep_only_eu(structure_ids: Container[int]) -> Callable[[SWCForest], SWCForest]:
    """
    Given `structure_ids`, a (list, set, tuple, etc.) of integers, \
    return a filtering function which accepts an :class:`swc.SWCForest` and \
    returns the subforest containing only the node types in `structure_ids`.

    Example: `keep_only([1,3,4])(forest)` is the subforest of `forest` containing only \
    the soma, the basal dendrites and the apical dendrites, but not the axon.

    The intended use is to generate a preprocessing function for `swc.read_preprocess_save`, \
    `swc.batch_filter_and_preprocess`, or `sample_swc.compute_and_save_intracell_all_euclidean`, \
    see the documentation for those functions for more information.

    :param structure_ids: A container of integers representing types of neuron nodes.
    :return: A filtering function taking as an argument an SWCForest `forest` and \
    returning the subforest of `forest` containing only the node types in `structure_ids`.
    """

    def filter_by_structure_ids(forest: SWCForest) -> SWCForest:
        return filter_forest(
            forest, lambda node: operator.contains(structure_ids, node.structure_id)
        )

    return filter_by_structure_ids


def preprocessor_eu(
    structure_ids: Container[int] | Literal["keep_all_types"], soma_component_only: bool
) -> Callable[[SWCForest], Err[str] | SWCForest]:
    """
    :param structure_ids: Either a collection of integers corresponding to structure ids in the \
    SWC spec, or the literal string 'keep_all_types'.
    :param soma_component_only: Indicate whether to sample from the whole SWC file, or only \
    from the connected component containing the soma. Whether this flag is appropriate \
    depends on the technology used to construct the SWC files. Some technologies generate \
    SWC files in which there are many unrelated connected components which are "noise" \
    contributed by other overlapping neurons. In other technologies, all components are \
    significant and the authors of the SWC file were simply unable to determine exactly where \
    the branch should be connected to the main tree. In order to get sensible results from \
    the data, the user should visually inspect neurons with multiple connected components \
    using a tool such as Vaa3D https://github.com/Vaa3D/release/releases/tag/v1.1.2 to \
    determine whether the extra components should be regarded as signal or noise.
    :return: A preprocessing function which accepts as argument an SWCForest `forest` and \
    returns a filtered forest containing only the nodes listed in `structure_ids`. If \
    `soma_component_only` is True, only nodes from the component containing the soma will be \
    returned; otherwise nodes will be drawn from across the whole forest.\
    If `soma_component_only` is True and there is not a unique connected component whose \
    root is a soma node, the function will return an error.
    """
    if soma_component_only:
        if structure_ids == "keep_all_types":

            def filter1(forest: SWCForest) -> Err[str] | SWCForest:
                soma_root_nodes = sum(
                    ((1 if tree.root.structure_id == 1 else 0) for tree in forest)
                )
                if soma_root_nodes != 1:
                    return Err(
                        "Found "
                        + str(soma_root_nodes)
                        + " many soma root nodes, not 1."
                    )
                return forest

            return filter1

        # This point in the code is reached only if structure_ids is not 'keep_all_types'.
        def filter2(forest: SWCForest) -> Err[str] | SWCForest:
            soma_root_nodes = sum(
                ((1 if tree.root.structure_id == 1 else 0) for tree in forest)
            )
            if soma_root_nodes != 1:
                return Err(
                    "Found " + str(soma_root_nodes) + " many soma root nodes, not 1."
                )
            soma_tree = next(tree for tree in forest if tree.root.structure_id == 1)
            return filter_forest(
                [soma_tree],
                lambda node: operator.contains(structure_ids, node.structure_id),
            )

        return filter2
    # soma_component_only is False.
    if structure_ids == "keep_all_types":
        return lambda forest: forest

    def filter3(forest: SWCForest) -> SWCForest:
        return filter_forest(
            forest, lambda node: operator.contains(structure_ids, node.structure_id)
        )

    return filter3


def preprocessor_geo(
    structure_ids: Container[int] | Literal["keep_all_types"],
) -> Callable[[SWCForest], NeuronTree]:
    """
    This preprocessor strips the tree down to only the components listed in `structure_ids` and \
    also trims the tree down to a single connected component.
    This is similar to :func:`swc.keep_only_eu` and the user should consult the documentation \
    for that function. Observe that the type signature is also different. The callable \
    returned by this function is suitable as a preprocessing function for \
    :func:`sample_swc.read_preprocess_compute_geodesic` or \
    :func:`sample_swc.compute_and_save_intracell_all_geodesic`.
    """

    if structure_ids == "keep_all_types":
        return lambda forest: forest[0]

    def filter_by_structure_ids(forest: SWCForest) -> NeuronTree:
        return filter_forest(
            forest, lambda node: operator.contains(structure_ids, node.structure_id)
        )[0]

    return filter_by_structure_ids


def total_length(tree: NeuronTree) -> float:
    """
    Return the sum of lengths of all edges in the graph.
    """
    acc_length = 0.0
    for tree0 in tree:
        for child_tree in tree0.child_subgraphs:
            acc_length += euclidean(
                np.array(tree0.root.coord_triple),
                np.array(child_tree.root.coord_triple),
            )
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


def discrete_depth(tree: NeuronTree) -> int:
    """
    :return: The height of the tree in the unweighted or discrete sense, i.e. the \
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
    :return: A dictionary whose keys are all structure_id's in `tree` and whose values are \
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
    :return: a dictionary whose keys are all structure_id's in `forest` and whose values are \
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
    :return: The number of nodes in `tree`.
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


def diagnostics(
    infolder: str,
    test: Callable[[SWCForest], Optional[Err[str]]],
    parallel_processes: int,
    name_validate: Callable[[str], bool] = default_name_validate,
) -> None:
    """
    Go through every SWC in infolder and apply `test` to the forest. \
    Print the names of cells failing the tests.
    """

    cell_names, file_paths = get_filenames(infolder, name_validate)

    def check_errs(file_path: str) -> Optional[Err[str]]:
        loaded_forest, _ = read_swc(file_path)
        return test(loaded_forest)

    pool = ProcessPool(nodes=parallel_processes)
    results = pool.imap(check_errs, file_paths)

    for cell_name, result in zip(cell_names, results):
        if isinstance(result, Err):
            print(cell_name + " " + str(result.code))

    pool.close()
    pool.join()
    pool.clear()


def read_preprocess_save(
    infile_name: str,
    outfile_name: str,
    preprocess: Callable[[SWCForest], Err[T] | SWCForest | NeuronTree],
) -> Err[T] | Literal["success"]:
    r"""
    Read the \*.swc file `file_name` from disk as an `SWCForest`.
    Apply the function `preprocess` to the forest. If preprocessing returns an error,\
    return that error. \
    Otherwise, write the preprocessed swc to outfile and return the string "success".

    This function exists mostly for convenience, as it can be called in parallel on \
    several files at once without requiring a large amount of data to be \
    communicated between processes.
    """
    loaded_forest, _ = read_swc(infile_name)
    tree = preprocess(loaded_forest)
    if isinstance(tree, Err):
        return tree
    if isinstance(tree, list):
        write_swc(outfile_name, tree)
    if isinstance(tree, NeuronTree):
        write_swc(outfile_name, [tree])
    return "success"


def get_filenames(
    infolder: str, name_validate: Callable[[str], bool] = default_name_validate
) -> tuple[list[str], list[str]]:
    """
    Get a list of all files in infolder. Filter the list by name_validate. \

    :return: a pair of lists (cell_names, file_paths), where `file_paths` are the paths  \
    to cells we want to sample from, and `cell_names[i]` is the substring of `file_paths[i]` \
    containing only the file name, minus the extension; i.e., if  file_paths[i] is
    "/home/jovyan/files/abc.swc" then cell_names[i] is "abc".

    See :func:`swc.default_name_validate` for an example of a name validation function.
    """

    file_names = [
        file_name for file_name in os.listdir(infolder) if name_validate(file_name)
    ]
    file_paths = [os.path.join(infolder, file_name) for file_name in file_names]
    cell_names = [os.path.splitext(file_name)[0] for file_name in file_names]
    return (cell_names, file_paths)


_empty_str = ""


def batch_filter_and_preprocess(
    infolder: str,
    outfolder: str,
    preprocess: Callable[[SWCForest], Err[T] | SWCForest | NeuronTree],
    parallel_processes: int,
    err_log: Optional[str],
    suffix: Optional[str] = None,
    name_validate: Callable[[str], bool] = default_name_validate,
) -> None:
    r"""

    Get the set of files in infolder. Filter down to the filenames which pass the test
    `name_validate`, which is responsible for filtering out any non-swc files.\
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
    :param preprocess: A function to filter out bad SWC forests or transform them into a more \
        manageable form.
    :param parallel_processes: Run this many Python processes in parallel.
    :param suffix: If a file in infolder has the name "abc.swc" then the corresponding file \
        written to outfolder will have the name "abc" + suffix + ".swc".
    :param name_validate: A function which identifies the files in `infolder` which \
        are \*.swc files. The default argument, :func:`swc.default_name_validate`, checks to see \
        whether the filename has file extension ".swc", case insensitive, and discards files \
        starting with '.', the marker for hidden files on Linux. The user may need \
        to write their own function to ensure that various kinds of backup /autosave files \
        and metadata files are not read into memory.
    """

    if suffix is None:
        suffix = ""
    try:
        os.mkdir(outfolder)
    except OSError:
        # print(error)
        pass
    cell_names, file_paths = get_filenames(infolder, default_name_validate)

    def rps(str_pair: tuple[str, str]) -> Err[T] | Literal["success"]:
        cell_name, file_path = str_pair
        outpath = os.path.join(outfolder, cell_name + suffix + ".swc")
        return read_preprocess_save(file_path, outpath, preprocess)

    pool = ProcessPool(nodes=parallel_processes)
    results = pool.imap(rps, zip(cell_names, file_paths))

    if err_log is not None:
        with open(err_log, "w", newline="") as outfile:
            for cell_name, result in zip(cell_names, results):
                if result == "success":
                    pass
                elif isinstance(result, Err):
                    outfile.write(cell_name + " " + str(result.code) + "\n")
                else:
                    raise ValueError(
                        "Should be error result or 'success' string literal."
                    )

    pool.close()
    pool.join()
    pool.clear()
