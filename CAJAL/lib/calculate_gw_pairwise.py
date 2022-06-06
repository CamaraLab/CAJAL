import os
import time

from CAJAL.lib import run_gw


def run_euclidean(data_dir, gw_results_dir, data_prefix, num_cores, file_prefix):
    t1 = time.time()
    dist_mat_list = run_gw.get_distances_all(data_dir=data_dir, data_prefix=data_prefix)
    run_gw.save_dist_mat_preload_global(dist_mat_list, file_prefix, gw_results_dir,
                                        save_mat=False, num_cores=num_cores)
    t3 = time.time()
    return t3 - t1


def run_geodesic(distances_dir, gw_results_dir, data_prefix, num_cores, file_prefix):
    t1 = time.time()
    dist_mat_list = run_gw.load_distances_global(distances_dir=distances_dir, data_prefix=data_prefix)
    run_gw.save_dist_mat_preload_global(dist_mat_list, file_prefix, gw_results_dir, save_mat=False,
                                        num_cores=num_cores)
    t3 = time.time()
    return t3 - t1


def run_euclidean_example(file_prefix):
    run_euclidean(data_dir=os.path.abspath('../data/sampled_pts/example_sampled_50/'),
                  gw_results_dir=os.path.abspath('../data/gw_results'),
                  data_prefix="a10_full",
                  num_cores=12,
                  file_prefix=file_prefix)


def run_geodesic_example(file_prefix):
    run_geodesic(distances_dir=os.path.abspath('../data/sampled_pts/example_geodesic_50'),
                 gw_results_dir=os.path.abspath('../data/gw_results'),
                 data_prefix="a10_full",
                 num_cores=12,
                 file_prefix=file_prefix)


if __name__ == "__main__":
    run_euclidean_example("example_euclidean")
    run_geodesic_example("example_geodesic")
