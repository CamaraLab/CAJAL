# Provides command line interface for saving sampled points from .obj files
import argparse
import pathlib
import sys
sys.path.append('..') # can probably be removed when push to package
from CAJAL.lib.sample_mesh import save_geodesic_from_obj_parallel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evenly sample n vertices from .obj and save in csv')
    parser.add_argument('infolder', metavar='infolder', type=pathlib.Path,
                        help='folder containing .obj file')
    parser.add_argument('outfolder', metavar='outfolder', type=pathlib.Path,
                        help='folder to save sampled vertices csv')
    parser.add_argument('n_sample', metavar='n_sample', type=int,
                        help='number of vertices to sample from each mesh')
    parser.add_argument('--method', default="networkx",
                        help="one of 'networxk' or 'heat', how to compute geodesic distance. \
                        Networkx is slower but more exact for non-watertight methods, \
                        heat is a faster approximation")
    parser.add_argument('--connect', action='store_true',
                        help='check for disconnected meshes and connect them simply by adding faces. \
                        If omitted, will save sampled geodesic for each disconnected mesh in .obj')
    parser.add_argument('--num_cores', default=8,
                        help='number of processes to use for parallelization')

    args = parser.parse_args()
    save_geodesic_from_obj_parallel(**vars(args))

    # python scripts/sample_geodesic_obj.py -h
    # python scripts/sample_geodesic_obj.py data/obj_files ../../test_obj 50 --method heat