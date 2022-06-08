import unittest
from CAJAL.lib import visualization as vis
from CAJAL.lib.utilities import load_dist_mat


class TestErrorsClass(unittest.TestCase):

    def test_cluster_umap(self):
        gw_mat = load_dist_mat("../data/gw_results/example_euclidean_gw_dist_mat.txt")
        embedding = vis.get_umap(gw_mat, n_neighbors=5, min_dist=0.1, negative_sample_rate=5, n_components=2)
        cluster_louvain = vis.louvain_clustering(gw_mat, 5)
        plot = vis.plot_umap(embedding, c=cluster_louvain, cmap="Set1")

    def test_all_cluster(self):
        gw_mat = load_dist_mat("../data/gw_results/example_euclidean_gw_dist_mat.txt")
        cluster_louvain = vis.louvain_clustering(gw_mat, 5)
        cluster_leiden = vis.leiden_clustering(gw_mat, 5)
        cluster_leiden_res = vis.leiden_clustering(gw_mat, 5, 6e-5)


if __name__ == '__main__':
    unittest.main()
