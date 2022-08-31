# Provides command line interface for saving geodesic distance computed on sampled points from .obj files
import argparse
import pathlib
import sys
sys.path.append('..') # can probably be removed when push to package
from CAJAL.lib.sample_mesh import save_sample_from_obj_parallel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Save geodesic distance computed on sampled points from .obj files')
    parser.add_argument('infolder', metavar='infolder', type=pathlib.Path,
                        help='folder containing .obj file')
    parser.add_argument('outfolder', metavar='outfolder', type=pathlib.Path,
                        help='folder to save sampled vertices csv')
    parser.add_argument('n_sample', metavar='n_sample', type=int,
                        help='number of vertices to sample from each mesh')
    parser.add_argument('--disconnect', action='store_true',
                        help='separate mesh into disconnected components. If omitted, will sample from whole mesh')
    parser.add_argument('--num_cores', default=8,
                        help='number of processes to use for parallelization')

    args = parser.parse_args()
    save_sample_from_obj_parallel(**vars(args))

    # python scripts/sample_pts_from_obj.py -h
    # python scripts/sample_pts_from_obj.py data/obj_files ../../test_obj 50 --disconnect