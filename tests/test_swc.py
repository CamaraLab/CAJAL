import os
import math
from shutil import rmtree
from typing import Union

from cajal.utilities import Err
from cajal.swc import (
    batch_filter_and_preprocess,
    cell_iterator,
    default_name_validate,
    linearize,
    forest_from_linear,
    read_swc_node_dict,
    read_swc,
    node_type_counts_forest,
    num_nodes,
    has_soma_node,
    filter_forest,
    SWCForest,
    preprocessor_eu,
    write_swc,
    NeuronNode,
    total_length,
    discrete_depth,
    diagnostics,
    _depth_table,
    _branching_degree,
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


def filter_tests(forest: SWCForest) -> None:
    node_counts = node_type_counts_forest(forest)
    filter_1 = preprocessor_eu([2, 3], False)
    forest_1 = filter_1(forest)
    preprocessor_eu("keep_all_types", True)
    preprocessor_eu([1, 2, 3, 4], True)
    node_counts_1 = node_type_counts_forest(forest_1)
    for key in node_counts_1:
        assert key in node_counts
    for key in node_counts:
        if key in [2, 3]:
            assert key in node_counts_1


def test_1():
    swcdir = "tests/swc"
    cell_names, forests = zip(*list(cell_iterator(swcdir)))
    swc_file_names = [
        os.path.join(swcdir, f) for f in os.listdir(swcdir) if default_name_validate(f)
    ]
    node_counts_main = [node_type_counts_forest(forest) for forest in forests]
    raw_dicts = [read_swc_node_dict(f) for f in swc_file_names]
    node_counts_1 = [count_nodes1(d) for d in raw_dicts]
    num_swc_files = len(os.listdir(swcdir)) - 2
    assert len(cell_names) == num_swc_files
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
        _branching_degree(forest)
        _depth_table(forest[0])
        for i in range(len(lin)):
            assert lin[i].parent_sample_number - 1 < i
        all_sample_numbers = []
        for tree in linear_forest:
            assert total_length(tree) >= 0
            assert discrete_depth(tree) >= 0
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
        os.remove(outfile)
        assert sorted(linear_forest, key=num_nodes, reverse=True) == read_forest
        filtered_forest = filter_forest(
            linear_forest, lambda node: is_prime(node.sample_number)
        )
        filtered_forest_samples = []
        for tree in filtered_forest:
            filtered_forest_samples += [subtree.root.sample_number for subtree in tree]
        prime_sample_numbers = list(filter(is_prime, all_sample_numbers))
        assert set(prime_sample_numbers) == set(filtered_forest_samples)
        assert any(map(has_soma_node, forest))
        filter_tests(forest)


def only_even_nodes(forest: SWCForest) -> Union[Err[str], SWCForest]:
    node_cts = sum(node_type_counts_forest(forest).values())
    if node_cts % 2 == 0:
        return forest
    return Err(str(node_cts) + " many nodes.")


def test_2():
    swc_in_dir = "tests/swc"
    swc_out_dir = "tests/swc_test_out"
    batch_filter_and_preprocess(
        swc_in_dir,
        swc_out_dir,
        preprocess=only_even_nodes,
        parallel_processes=8,
        err_log="tests/swc_err_log.txt",
    )
    for _, forest in cell_iterator(swc_out_dir):
        assert sum(node_type_counts_forest(forest).values()) % 2 == 0
    with open("tests/swc_err_log.txt") as infile:
        for line in infile:
            filename = line.split()[0] + ".swc"
            forest, _ = read_swc(os.path.join(swc_in_dir, filename))
            assert sum(node_type_counts_forest(forest).values()) % 2 == 1
    os.remove("tests/swc_err_log.txt")
    rmtree(swc_out_dir)


def test_diagnostics():
    diagnostics("tests/swc", lambda forest: None, 1)
