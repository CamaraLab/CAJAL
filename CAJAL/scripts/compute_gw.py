# Command line interface for computing and saving GW distance matrix
import argparse
import pathlib
import time
import sys
sys.path.append('..') # can probably be removed when push to package
from CAJAL.lib.run_gw import get_distances_all, load_distances_global, save_dist_mat_preload_global


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute and save GW distance matrix')
    parser.add_argument('infolder', metavar='infolder', type=pathlib.Path,
                        help='folder containing files of sampled points or distance matrices per cell')
    parser.add_argument('outfile_prefix', metavar='outfile_prefix',
                        help='name of output file to write GW distance matrix to, without extension')
    parser.add_argument('outfolder', metavar='outfolder', type=pathlib.Path,
                        help='path to directory to write output file to')
    parser.add_argument('--data_prefix',
                        help='only read files from infolder starting with this string')
    parser.add_argument('--geodesic', action='store_true',
                        help='assume input folder contains files with distance matrices in vector form. \
                             If omitted, assumes input folder contains csv files of sampled point coordinates')
    parser.add_argument('--save_mat', action='store_true',
                        help='return coupling matrix (matching) between points, which takes much longer on large runs. \
                            If omitted, only returns GW distance.')
    parser.add_argument('--verbose', action='store_true',
                        help='print how much time the GW computation takes')
    parser.add_argument('--num_cores', metavar="N",default=8,
                        help='number of processes to use for parallelization')

    args = parser.parse_args()

    if args.geodesic:
        dist_mat_list = load_distances_global(distances_dir=args.infolder, data_prefix=args.data_prefix)
        start = time.time()
        save_dist_mat_preload_global(dist_mat_list, args.outfile_prefix, args.outfolder, save_mat=args.save_mat,
                                     num_cores=args.num_cores)
        end = time.time()
        if args.verbose:
            print('Time in GW calculation: {:.3f}s'.format(end - start))
    else:
        dist_mat_list = get_distances_all(data_dir=args.infolder, data_prefix=args.data_prefix)
        start = time.time()
        save_dist_mat_preload_global(dist_mat_list, args.outfile_prefix, args.outfolder, save_mat=args.save_mat,
                                     num_cores=args.num_cores)
        end = time.time()
        if args.verbose:
            print('Time in GW calculation: {:.3f}s'.format(end - start))

# python scripts/compute_gw.py -h
# python scripts/compute_gw.py data/sampled_pts/swc_sampled_50/ test ../../test --data_prefix a10_full