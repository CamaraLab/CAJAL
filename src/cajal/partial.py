"""For testing partial matching."""

import itertools as it

import numpy as np
from typing import Optional, Any

# import numpy.typing as npt
from copy import copy
from scipy.spatial.distance import euclidean

from .swc import NeuronNode, NeuronTree, preprocessor_eu, cell_iterator, total_length
from .sample_swc import icdm_euclidean
from .run_gw import uniform, gw_pairwise_parallel, DistanceMatrix


def dist(n1: NeuronNode, n2: NeuronNode):
    """Compute the Euclidean distance between two NeuronNodes."""
    return euclidean(
        np.array(n1.coord_triple),
        np.array(n2.coord_triple),
    )


def lengthtree(tree: NeuronTree):
    """Compute the length of the tree."""
    ell = list([t for t in tree])
    ell.reverse()
    # output_list = []
    length_tree_dict = dict()
    for tree in ell:
        child_ids = [ct.root.sample_number for ct in tree.child_subgraphs]
        children = [length_tree_dict[i] for i in child_ids]
        p1 = tree.root.coord_triple
        length = sum(
            [euclidean(p1, ct.root.coord_triple) for ct in tree.child_subgraphs]
        )
        length += sum([child["length"] for child in children])
        new_dict = {"length": length, "children": children}
        length_tree_dict[tree.root.sample_number] = new_dict
    return new_dict


def lengthtree_test(tree: NeuronTree):
    """Test lengthtree."""
    ell = [(tree, lengthtree(tree))]
    while ell:
        t, lt = ell.pop()
        assert len(t.child_subgraphs) == len(lt["children"])
        assert (total_length(t) - lt["length"]) < 0.01
        ell += list(zip(t.child_subgraphs, lt["children"]))


def newNeuronNode(n1: NeuronNode, n2: NeuronNode, d: float):
    """Create a new neuron node between n1 and n2, at distance d away from n1."""
    ct1 = np.array(n1.coord_triple)
    ct2 = np.array(n2.coord_triple)
    e = euclidean(ct1, ct2)
    ct3 = ct1 + ((ct2 - ct1) * d / e)
    return NeuronNode(
        sample_number=99999,
        structure_id=n1.structure_id,
        coord_triple=(ct3[0], ct3[1], ct3[2]),
        radius=n1.radius,
        parent_sample_number=n1.sample_number,
    )


def trim_swc_no_mutate(
    t: NeuronTree, lt: dict[str, Any], p: float, rng: np.random.Generator
) -> NeuronTree:
    """Randomly cut off proportion p of the tree t."""
    cut = p * lt["length"]
    ltchildren = copy(lt["children"])
    t0 = NeuronTree(root=copy(t.root), child_subgraphs=copy(t.child_subgraphs))
    children_copied = True
    t1 = t0  # t1 *should* be mutated. This is what we want to return.

    while cut > 0.0:
        num_children = len(t0.child_subgraphs)
        length_of_children = [
            dist(t0.root, t0.child_subgraphs[i].root) + ltchildren[i]["length"]
            for i in range(num_children)
        ]

        assert abs(sum(length_of_children) - total_length(t0)) < 0.01
        x = rng.uniform(low=0.0, high=sum(length_of_children))
        i = 0
        thres = 0.0
        while x >= thres + length_of_children[i]:
            thres += length_of_children[i]
            i += 1
        if length_of_children[i] <= cut:
            if not children_copied:
                t0.child_subgraphs = copy(t0.child_subgraphs)
                children_copied = True
            t0.child_subgraphs.pop(i)
            cut -= length_of_children[i]

            ltchildren.pop(i)
        elif cut >= ltchildren[i]["length"]:
            if not children_copied:
                t0.child_subgraphs = copy(t0.child_subgraphs)
                children_copied = True
            t0.child_subgraphs[i] = NeuronTree(
                root=newNeuronNode(
                    t0.root, t0.child_subgraphs[i].root, cut - ltchildren[i]["length"]
                ),
                child_subgraphs=[],
            )
            cut = 0.0
        else:
            if not children_copied:
                t0.child_subgraphs = copy(t0.child_subgraphs)
                children_copied = True
            t0.child_subgraphs[i] = copy(t0.child_subgraphs[i])
            t0 = t0.child_subgraphs[i]
            ltchildren = copy(ltchildren[i]["children"])
            children_copied = False
    return t1


def trim_swc(t: NeuronTree, params: list[float], rng: np.random.Generator):
    """
    Make trimmed copies of t.

    :return: a list [t0,... tn] where t_i is a modified copy of t with
        proportion p_i of the branches randomly deleted.
    """
    lt = lengthtree(t)
    ell = []
    for p in params:
        ell.append(trim_swc_no_mutate(t, lt, p, rng))
    return ell


def test_trim_nt(nt: NeuronTree):
    """Test trim_swc."""
    params = [0.1, 0.2, 0.3, 0.4]
    rng = np.random.default_rng(seed=0)
    trees = trim_swc(nt, params, rng)
    nt_len = total_length(nt)
    nt_trim_lens = list(map(total_length, trees))
    for i in range(4):
        if (nt_len * (1 - (params[i]))) < 0.99 * nt_trim_lens[i] or (
            nt_len * (1 - (params[i]))
        ) > 1.01 * nt_trim_lens[i]:
            print("nt_len", nt_len)
            print("i", i)
            print("nt_trim_len", nt_trim_lens[i])


def partial_matching_analysis(
    infolder: str,
    firstkcells: int,
    samplepts: int,
    parameters: list[float],
    gw_dist_csv: str,
    seed: Optional[int] = 0,
):
    """Carries out a preset analysis routine to assess partial matching on neurons."""
    rng = np.random.default_rng(seed)
    num_params = len(parameters)
    pe = preprocessor_eu(structure_ids=[1, 3, 4], soma_component_only=True)
    ci = it.islice(cell_iterator(infolder), firstkcells)
    dms: list[DistanceMatrix] = []
    names = []
    for cell_name, cell in ci:
        pe_cell = pe(cell)
        if isinstance(pe_cell, list):
            f = pe_cell[0]
        swcs = trim_swc(f, parameters, rng)
        dms += [icdm_euclidean([t], samplepts) for t in swcs]
        names += [cell_name + "_" + str(p) for p in parameters]
    u = uniform(samplepts)
    cells = [(dm, u) for dm in dms]
    assert len(cells) == num_params * firstkcells
    a = gw_pairwise_parallel(
        cells=cells, num_processes=14, names=names, gw_dist_csv=gw_dist_csv
    )
    return a[0]
