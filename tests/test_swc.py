import os
import math

from src.cajal.swc import (
    cell_iterator,
    default_name_validate,
    linearize,
    forest_from_linear,
    read_swc_node_dict,
    read_swc,
    node_type_counts_forest,
    num_nodes,
    filter_forest,
    write_swc,
    NeuronNode,
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


def is_prime(k: int):
    if k < 2:
        return False
    for i in range(2, math.ceil(math.sqrt(k))):
        if k % i == 0:
            return False
    return True


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
    del node_counts_main
    for swc_file in swc_file_names[:10]:
        forest, tree_index = read_swc(swc_file)
        roots = 0
        for key in tree_index:
            parent = tree_index[key].root.parent_sample_number
            assert parent == -1 or parent == tree_index[parent].root.sample_number
            if parent == -1:
                roots += 1

        assert len(forest) == roots
        lin = linearize(forest)
        linear_forest = forest_from_linear(lin)
        for i in range(len(lin)):
            assert lin[i].parent_sample_number - 1 < i
        all_sample_numbers = []
        for tree in linear_forest:
            sample_numbers_list = [subtree.root.sample_number for subtree in tree]
            assert sample_numbers_list == sorted(sample_numbers_list)
            # sample_numbers_list1 = sorted(set(sample_numbers_list))
            # assert len(sample_numbers_list1) == len(sample_numbers_list)
            if len(sample_numbers_list) > 1:
                for k in range(len(sample_numbers_list) - 1):
                    assert sample_numbers_list[k] + 1 == sample_numbers_list[k + 1]
            min_index = tree.root.sample_number
            max_index = tree.root.sample_number - 1
            for subtree in tree:
                assert subtree.root.sample_number >= min_index
                assert subtree.root.sample_number > max_index
                max_index = subtree.root.sample_number
            assert min_index == sample_numbers_list[0]
            assert max_index == sample_numbers_list[-1]
            all_sample_numbers += sample_numbers_list
        out_swc, ext = os.path.split(swc_file)
        outfile = out_swc + "_OUT" + ext
        write_swc(outfile, forest)
        read_forest, _ = read_swc(outfile)
        assert sorted(linear_forest, key=num_nodes) == read_forest
        filtered_forest = filter_forest(
            linear_forest, lambda node: is_prime(node.sample_number)
        )
        filtered_forest_samples = []
        for tree in filtered_forest:
            filtered_forest_samples += [subtree.root.sample_number for subtree in tree]
        prime_sample_numbers = list(filter(is_prime, all_sample_numbers))
        assert set(prime_sample_numbers) == set(filtered_forest_samples)
