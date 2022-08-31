# Command line interface for plotting UMAP visualization of GW distance
import argparse
import pathlib
import time
import sys
sys.path.append('..') # can probably be removed when push to package
from CAJAL.lib import visualization as vis, utilities

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot UMAP visualization of GW distance')
    parser.add_argument('gw_dist_file', type=pathlib.Path,
                        help='file with GW distance matrix in vector form')
    parser.add_argument('--cluster', default="",
                        help='Clustering method, either "louvain" or "leiden". If omitted, will not cluster.')
    parser.add_argument('--cluster_nn', default=5, type=int,
                        help='Number of nearest neighbors for clustering method')

    args = parser.parse_args()

    # Read in GW distance matrix
    gw_mat = utilities.load_dist_mat(args.gw_dist_file)

    # Create 2-dimensional UMAP embedding to visualize the morphology space
    umap_emb = vis.get_umap(gw_mat)

    if args.cluster == "":
        # Scatter plot of UMAP embedding (each point is a cell)
        vis.plot_umap(umap_emb)
    elif args.cluster == "louvain":
        # Louvain clustering of morphology space (cells in the same cluster have more similar morphologies)
        louvain_clus = vis.louvain_clustering(gw_mat, nn=args.cluster_nn)
        vis.plot_umap(umap_emb, c = louvain_clus, cmap="Set1")
    elif args.cluster == "leiden":
        # Leiden clustering of morphology space (cells in the same cluster have more similar morphologies)
        leiden_clus = vis.leiden_clustering(gw_mat, nn=args.cluster_nn)
        vis.plot_umap(umap_emb, c = leiden_clus, cmap="Set1")
    else:
        print("Invalid --cluster parameter, must be louvain, leiden, or left blank")


# python scripts/plot_umap.py -h
# python scripts/plot_umap.py ../../test/test_gw_dist_mat.txt