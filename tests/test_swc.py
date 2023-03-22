import os

from cajal.swc import (
    cell_iterator,
    default_name_validate,
    read_swc_node_dict,
    node_type_counts_forest,
    NeuronNode,
    NeuronTree,
    SWCForest,
)


def count_nodes1(d: dict[int, NeuronNode]) -> dict[int, int]:
    count_dict = {}
    for node in d.values():
        id = node.structure_id
        if id in count_dict:
            count_dict[id] += 1
        else:
            count_dict[id] = 1
    return count_dict


def count_nodes2(forest: SWCForest) -> dict[int, int]:
    count_dict = {}
    for tree in forest:
        for node in tree:
            id = node.root.structure_id
            if id in count_dict:
                count_dict[id] += 1
            else:
                count_dict[id] = 1
    return count_dict


def count_nodes3(forest: SWCForest) -> dict[int, int]:
    count_dict = {}
    for tree in forest:
        for node in tree.dfs():
            id = node.root.structure_id
            if id in count_dict:
                count_dict[id] += 1
            else:
                count_dict[id] = 1
    return count_dict


def test_1():
    swcdir = "CAJAL/data/swc"
    cell_names, forests = zip(*list(cell_iterator(swcdir)))
    swc_file_names = [
        os.path.join(swcdir, f) for f in os.listdir(swcdir) if default_name_validate(f)
    ]
    node_counts_main = [node_type_counts_forest(forest) for forest in forests]
    raw_dicts = [read_swc_node_dict(f) for f in swc_file_names]
    node_counts_1 = [count_nodes1(d) for d in raw_dicts]
    assert len(cell_names) == 100
    assert node_counts_main == node_counts_1
    del node_counts_1
    node_counts_2 = [count_nodes2(forest) for forest in forests]
    assert node_counts_main == node_counts_2
    del node_counts_2
    node_counts_3 = [count_nodes3(forest) for forest in forests]
    assert node_counts_main == node_counts_3
    del node_counts_3
