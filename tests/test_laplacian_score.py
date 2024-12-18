from cajal.laplacian_score import laplacian_score_no_covariates, laplacian_scores
from scipy.spatial.distance import squareform
import numpy as np
import statistics


def test():
    # with open("CAJAL/data/c_elegans_features.csv") as infile:
    #     csv_reader = csv.reader(infile)
    #     w
    data = np.loadtxt("CAJAL/data/c_elegans_features.csv", delimiter=",", dtype=str)
    # header = data[0] (Currently unused.)
    # feature_names = header[1:-1] (Currently unused.)
    cell_names = data[1:, 0]
    days = data[1:, -1].astype(int)

    feature_matrix = data[1:, 1:-1].astype(int)
    num_features = feature_matrix.shape[1]

    day1_indices = np.nonzero(
        np.array(["day1" in cell_name for cell_name in cell_names])
    )[0]
    day2_indices = np.nonzero(
        np.array(["day2" in cell_name for cell_name in cell_names])
    )[0]
    day3_indices = np.nonzero(
        np.array(["day3" in cell_name for cell_name in cell_names])
    )[0]
    day5_indices = np.nonzero(
        np.array(["day5" in cell_name for cell_name in cell_names])
    )[0]

    fmat_day1 = feature_matrix[day1_indices]
    fmat_day2 = feature_matrix[day2_indices]
    fmat_day3 = feature_matrix[day3_indices]
    fmat_day5 = feature_matrix[day5_indices]
    nontriv_day1 = np.nonzero(np.any(fmat_day1, axis=0))[0]
    nontriv_day2 = np.nonzero(np.any(fmat_day2, axis=0))[0]
    nontriv_day3 = np.nonzero(np.any(fmat_day3, axis=0))[0]
    nontriv_day5 = np.nonzero(np.any(fmat_day5, axis=0))[0]

    fmat_day1 = fmat_day1[:, nontriv_day1]
    fmat_day2 = fmat_day2[:, nontriv_day2]
    fmat_day3 = fmat_day3[:, nontriv_day3]
    fmat_day5 = fmat_day5[:, nontriv_day5]

    gw_csv_loc = "tests/c_elegans_gw_mat.csv"
    gw_dist_mat = squareform(np.loadtxt(gw_csv_loc))

    gw_dists_day1 = gw_dist_mat[day1_indices][:, day1_indices]
    gw_dists_day2 = gw_dist_mat[day2_indices][:, day2_indices]
    gw_dists_day3 = gw_dist_mat[day3_indices][:, day3_indices]
    gw_dists_day5 = gw_dist_mat[day5_indices][:, day5_indices]

    median1 = statistics.median(squareform(gw_dists_day1, force="tovector"))
    median2 = statistics.median(squareform(gw_dists_day2, force="tovector"))
    median3 = statistics.median(squareform(gw_dists_day3, force="tovector"))
    median5 = statistics.median(squareform(gw_dists_day5, force="tovector"))

    num_permutations = 20
    ls_day1 = laplacian_score_no_covariates(
        fmat_day1, gw_dists_day1, median1, num_permutations, True
    )
    laplacian_score_no_covariates(
        fmat_day2, gw_dists_day2, median2, num_permutations, False
    )
    laplacian_score_no_covariates(
        fmat_day3, gw_dists_day3, median3, num_permutations, False
    )
    laplacian_score_no_covariates(
        fmat_day5, gw_dists_day5, median5, num_permutations, False
    )

    feature_data, other = ls_day1
    assert feature_data["feature_laplacians"].shape == nontriv_day1.shape
    assert feature_data["laplacian_p_values"].shape == nontriv_day1.shape
    assert feature_data["laplacian_q_values"].shape == nontriv_day1.shape
    assert other["random_feature_laplacians"].shape == (
        num_permutations,
        nontriv_day1.shape[0],
    )

    med = statistics.median(squareform(gw_dist_mat))
    covariate = days
    feature_data, other = laplacian_scores(
        feature_matrix, gw_dist_mat, med, num_permutations, covariate, True
    )
    assert feature_data["feature_laplacians"].shape == (num_features,)
    assert feature_data["laplacian_p_values"].shape == (num_features,)
    assert feature_data["laplacian_q_values"].shape == (num_features,)
    assert feature_data["beta_0"].shape == (num_features,)
    assert feature_data["beta_1"].shape == (num_features,)
    assert feature_data["regression_coefficients_fstat_p_values"].shape == (
        num_features,
    )
    assert feature_data["laplacian_p_values_post_regression"].shape == (num_features,)
    num_covariates = 1
    assert other["covariate_laplacians"].shape == (num_covariates,)
    assert other["random_feature_laplacians"].shape == (num_permutations, num_features)
    assert other["random_covariate_laplacians"].shape == (
        num_permutations,
        num_covariates,
    )
