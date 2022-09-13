# Provides command line interface for saving geodesic distance computed on sampled points from .obj files
import argparse
import pathlib
import sys
sys.path.append('..')  # can probably be removed when push to package
from CAJAL.lib.sample_swc import save_sample_pts_parallel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Save geodesic distance computed on sampled points from .swc files')
    parser.add_argument('infolder', metavar='infolder', type=pathlib.Path,
                        help='folder containing .swc files')
    parser.add_argument('outfolder', metavar='outfolder', type=pathlib.Path,
                        help='folder to save sampled vertices csv')
    parser.add_argument('goal_num_pts', metavar='n_sample', type=int,
                        help='number of vertices to sample from each swc file')
    parser.add_argument('--keep_disconnect', action='store_true',
                        help='sample points from all branches in SWC. If omitted, will sample from only \
                        branches connected to soma')
    parser.add_argument('--num_cores', default=8,
                        help='number of processes to use for parallelization')
    parser.add_argument("--types_keep", nargs="+", default=[0, 1, 2, 3, 4],
                        help='list of SWC neuron part types to sample points from. \
                        By default, uses only 1 (soma), 2 (axon), 3 (basal dendrite), 4 (apical dendrite)')

    args = parser.parse_args()
    save_sample_pts_parallel(**vars(args))

    # python scripts/sample_pts_from_swc.py -h
    # python scripts/sample_pts_from_swc.py ../../data/swc_test ../../test 50
    # python scripts/sample_pts_from_swc.py ../../data/swc_test ../../test 50 --types_keep 1 2
