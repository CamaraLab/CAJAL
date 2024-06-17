import numpy as np
import numpy.typing as npt
from scipy.stats import rankdata, t, f
from typing import Optional


def pearson_coefficient(
    feature_arr: npt.NDArray[np.float64],
    joint_dist: npt.NDArray[np.float64],
    permutations: int,
) -> npt.NDArray[np.float64]:
    r"""
    Let X, Y be two random variables on the finite set {0,...,N-1}, jointly distributed with \
    distribution described in the matrix joint_dist; joint_dist[i][j] = p(X=i,Y=j). \
    Let f be a function from {0,...,N-1} to |R. In what follows we call f a "feature." \
    Then f(X) and f(Y) are random variables. \
    Here we compute the Pearson correlation coefficient of f(X) and f(Y), and return it. \
    We also compute the Pearson correlation coefficient of \
    :math:`f(\\sigma \circ X)`, :math:`f(\\sigma \circ Y)` \
    for `permutations` many randomly selected permutations :math:`\\sigma` of the \
    set {0,\dots,n-1}. \
    (It it is assumed that X and Y are identically distributed.) \

    :param feature_arr: array of shape (N ,num_features), where N is the \
    number of possible values the random variables X, Y can take on, and num_features is \
    the number of features. Each column represents a feature on N elements.
    :param joint_dist: an (N,N) matrix whose entries sum to 1, \
    where the sum of entries in the i-th row is equal to the sum of entries in the i-th column. \
    :permutations: Recompute the Pearson correlation coefficient for
    `permutations` many randomly chosen permutations \ of the underlying set
    {0,\dots, n-1}.
    :return: A matrix `A` of shape (permutations+1,num_features) where A[0,j] is the Pearson \
    correlation coefficient of f_j(X) with f_j(Y) (where f_j is the j-th feature \
    in `feature_arr` ) and A[i,j] (for i > 0) is the Pearson correlation coefficient of \
    f_j(sigma_i \circ X) with f_j(sigma_i \circ X), where sigma_i is a randomly chosen permutation \
    of the set {0,\dots, n-1}.
    """

    # The probabilities for the random variable X.
    # The number of different values X and Y can take on.
    num_features = feature_arr.shape[1]
    N = feature_arr.shape[0]
    pearson_coefficient_list = []

    # Allocate space for temporary storage matrices
    means = np.zeros((num_features,))
    f_tildes = np.zeros((N, num_features), dtype=np.float64)
    tempvar1 = np.zeros((num_features, N), dtype=np.float64)
    tempvar2 = np.zeros((num_features, N), dtype=np.float64)
    rng = np.random.default_rng()
    covariance = np.zeros((num_features,), dtype=np.float64)
    variance = np.zeros((num_features,), dtype=np.float64)

    for i in range(permutations + 1):
        if i > 0:
            new_index_list = rng.permutation(N)
            joint_dist = joint_dist[new_index_list, :]
            joint_dist = joint_dist[:, new_index_list]
        X = np.sum(joint_dist, axis=0)

        # Compute the means of the features wrt the given distribution.
        np.dot(feature_arr.T, X, out=means)
        # Normalize the features by subtracting the means.
        np.subtract(feature_arr, means[np.newaxis, :], out=f_tildes)
        # Compute the covariance of f(X1) with f(X2).
        np.matmul(f_tildes.T, joint_dist, out=tempvar1)
        np.multiply(tempvar1, f_tildes.T, out=tempvar2).sum(axis=1, out=covariance)

        # Compute var(f(X1))=var(f(X2)).
        np.multiply(f_tildes.T, f_tildes.T, out=tempvar1)
        tempvar1.dot(X, out=variance)
        assert np.all(variance != 0.0)
        pearson_coefficient_list.append(covariance / variance)
    return np.stack(pearson_coefficient_list, axis=0)


def _validate(feature_arr: npt.NDArray[np.float64]):
    for i in range(feature_arr.shape[1]):
        v = feature_arr[:, i]
        if np.all(v == v[0]):
            raise Exception(
                "feature_arr[:,"
                + str(i)
                + "] is constant. \
                            This leads to divide-by-zero errors. \
                            Please clean data by removing constant columns."
            )


def _to_distribution(
    dist_mat: npt.NDArray[np.float64], epsilon: float
) -> npt.NDArray[np.float64]:
    """
    Convert a distance matrix on N elements to a probability distribution on
    N x N elements by a two step process:

    * create a graph adjacency matrix where two edges are connected iff their
      distance is less than epsilon
    * normalize the graph adjacency matrix so that it is a probability distribution.

    :param dist_mat: squareform distance matrix
    :param epsilon: threshold to connect two nodes in the graph
    :return: an n x n probability matrix representing a joint distribution on
    two variables.
    """
    adjacency_matrix = (dist_mat < epsilon).astype(np.int_)
    np.fill_diagonal(adjacency_matrix, 0)
    return adjacency_matrix / np.sum(adjacency_matrix)


def multilinear_regression(
    X: npt.NDArray[np.float64], Y: npt.NDArray[np.float64]
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """
    Compute the least-squares solution b to Y = Xb
    :param X: shape (n, p)
    :param Y: shape (n, m)

    :return: A tuple (b, e, SSE, SSR, s2b), where
    * b is coefficicient matrix minimizing the sum of squared errors ||Y - Xb||, shape (p,m)
    * e, the matrix of residuals, shape (n,m)
    * SSE, the sum of squared errors (residuals), shape (m,)
    * SSR, the sum of squares of the regression, shape (m,)
    * s2b, the sample variance-covariance matrices of the observed coefficient vectors b.
      Shape (p,p,m), where s2b[i,j,k] is the estimated covariance of b[i,k] with b[j,k].
    """

    b, SSE = np.linalg.lstsq(X, Y, rcond=None)[0:2]
    hat_Y = np.matmul(X, b)
    Y_means = np.sum(Y, axis=0) / Y.shape[0]
    Ya = np.subtract(hat_Y, Y_means[np.newaxis, :])
    SSR = np.multiply(Ya, Ya).sum(axis=0)
    assert SSR.shape == (Y.shape[1],)
    XTX_inv = np.linalg.inv(np.matmul(X.T, X))
    n = Y.shape[0]
    p = X.shape[1]
    s2b = np.tensordot(XTX_inv, SSE, axes=0) / (n - p)
    return b, np.subtract(Y, hat_Y), SSE, SSR, s2b


def benjamini_hochberg(p_values: npt.NDArray) -> npt.NDArray:
    """
    Takes an array of p-values, and returns an array of q-values.
    """
    ranked_p_values = rankdata(p_values)
    q_values = p_values * len(p_values) / ranked_p_values
    q_values[q_values >= 1] = 1
    return q_values


def permutation_pvalue(
    X: npt.NDArray[np.float64], A: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    p-value is computed from Phipson and Smyth, "Permutation P-values should never be zero."

    :param X: shape (k,)
    :param A: shape (n,k)

    :return: a vector V of shape (k,) where V[i] tells the p-value that the given
    position of X[i] relative to the elements of A[i] would arise from chance.

    """
    return (np.greater(X[np.newaxis, :], A).astype(np.int_).sum(axis=0) + 1) / (
        (A.shape[0] + 1)
    )


def laplacian_score_w_covariates(
    feature_arr: npt.NDArray[np.float64],
    distance_matrix: npt.NDArray[np.float64],
    epsilon: float,
    permutations: int,
    covariates: npt.NDArray[np.float64],
    return_random_laplacians: bool,
) -> tuple[dict[str, npt.NDArray[np.float64]], dict[str, npt.NDArray[np.float64]]]:
    """
    :param feature_arr: An array of shape (N ,num_features), where N is the \
    number of nodes in the graph, and num_features is \
    the number of features. Each column represents a feature on N elements. \
    Columns should be preprocessed to remove constant features.
    :param distance_matrix: squareform distance matrix of size (N,N)
    :param epsilon: connect nodes of graph if their distance is less than epsilon
    :param permutations: how many permutations should be run
    :param covariates: array of shape (N, num_covariates), where N is the number \
    of nodes in the graph, and num_covariates is the number of covariates
    :param return_random_laplacians: if True, the output dictionary will contain \
    of all the generated laplacians. This will likely be the largest object in the \
    dictionary.

    :return: A tuple of two dictionaries, "feature_data" and "other".
    All values in feature_data are arrays of shape (num_features,).

    * feature_data['feature_laplacians'] := the laplacian scores of f
    * feature_data['laplacian_p_values'] := the p-values from the permutation test
    * feature_data['laplacian_q_values'] := the q-values from the permutation test
    * (for i in range(1,covariates)) feature_data['beta_i'] := the p-value that beta_i is not zero \
      for that feature; see p. 228, 'Applied Linear Statistical Models', \
      Nachtsheim, Kutner, Neter, Li. Shape (num_features,)
    * feature_data['regression_coefficients_fstat_p_values'] := the p-value that not all beta_i \
          are zero, using the F-statistic, \
          see p. 226, 'Applied Linear Statistical Models', Nachtsheim, \
      Kutner, Neter, Li.
    * feature_data['laplacian_p_values_post_regression'] := the p-value of the residual \
          laplacian of the feature once the \
      covariates have been regressed out.
    * feature_data['laplacian_q_values_post_regression'] := the q-values from the permutation test.
    * other['covariate_laplacians'] := the laplacian scores of the covariates, of shape \
          (num_covariates,)
    * (Optional, if `return_random_laplacians` is True) \
      other['random_feature_laplacians'] := the matrix of randomly generated feature laplacians, \
      shape (permutations,num_features).
    * (Optional, if `return_random_laplacians` is True) \
      other['random_covariate_laplacians'] := the matrix of randomly generated
          covariate laplacians, shape (permutations, num_covariates)
    """

    distribution = _to_distribution(distance_matrix, epsilon)
    N = feature_arr.shape[0]
    num_features: int = feature_arr.shape[1]
    if len(covariates.shape) == 1:
        covariates = covariates[:, np.newaxis]
    num_covariates = covariates.shape[1]
    A = np.concatenate((feature_arr, covariates), axis=1)
    assert A.shape == (N, num_features + num_covariates)
    laplacians: npt.NDArray[np.float64] = (
        np.negative(pearson_coefficient(A, distribution, permutations)) + 1.0
    )

    # Extract the four quadrants from the laplacians matrix.
    # true feature laplacians
    tfl = laplacians[0, :num_features]
    assert tfl.shape == (num_features,)
    # random feature laplacians
    rfl = laplacians[1:, :num_features]
    assert rfl.shape == (permutations, num_features)
    # true covariate laplacians
    tcl = np.concatenate((np.array([1.0]), laplacians[0, num_features:]), axis=0)
    assert tcl.shape == (num_covariates + 1,)
    # random covariate laplacians
    rcl = np.concatenate(
        (
            np.full((permutations, 1), fill_value=1, dtype=np.float64),
            laplacians[1:, num_features:],
        ),
        axis=1,
    )
    assert rcl.shape == (permutations, num_covariates + 1)

    b, rfl_resids, SSE, SSR, s2b = multilinear_regression(rcl, rfl)
    assert b.shape == (num_covariates + 1, num_features)
    assert s2b.shape == (num_covariates + 1, num_covariates + 1, num_features)
    MSR = SSR / (num_covariates)
    if permutations <= num_covariates + 1:
        raise Exception("Must be more permutations than covariates.")
    MSE = SSE / (permutations - (num_covariates + 1))
    sb = np.sqrt(np.diagonal(s2b)).T

    # Compute the t-statistic to decide whether beta is statistically significant
    t_stat = (b / sb)[1:, :]
    p_betas = t.sf(t_stat, permutations - num_covariates)
    f_stat = MSR / MSE
    p_all_betas = f.sf(f_stat, num_covariates, permutations - num_covariates)
    tfl_resid = tfl - np.matmul(tcl, b)

    data = {}
    data["feature_laplacians"] = tfl
    data["laplacian_p_values"] = permutation_pvalue(tfl, rfl)
    data["laplacian_q_values"] = benjamini_hochberg(data["laplacian_p_values"])
    data["beta_0"] = b[0]
    for i in range(1, b.shape[0]):
        data["beta_" + str(i)] = b[i]
        data["beta_" + str(i) + "_p_value"] = p_betas[i - 1]
    data["regression_coefficients_fstat_p_values"] = p_all_betas
    data["laplacian_p_values_post_regression"] = permutation_pvalue(
        tfl_resid, rfl_resids
    )
    data["laplacian_q_values_post_regression"] = benjamini_hochberg(
        data["laplacian_p_values_post_regression"]
    )
    other = {}
    other["covariate_laplacians"] = tcl[1:]
    if return_random_laplacians:
        other["random_feature_laplacians"] = rfl
        other["random_covariate_laplacians"] = rcl[:, 1:]
    return (data, other)


def laplacian_score_no_covariates(
    feature_arr: npt.NDArray[np.float64],
    distance_matrix: npt.NDArray[np.float64],
    epsilon: float,
    permutations: int,
    return_random_laplacians: bool,
) -> tuple[dict[str, npt.NDArray[np.float64]], dict[str, npt.NDArray[np.float64]]]:
    """
    :param feature_arr: An array of shape (N ,num_features), where N is the \
        number of nodes in the graph, and num_features is \
        the number of features. Each column represents a feature on N elements. \
        Columns should be preprocessed to remove constant features.
    :param distance_matrix: A squareform distance matrix containing pairwise distances \
        between points in a space. Should be of size (N,N).
    :param epsilon: From `distance_matrix` we will build an undirected graph G \
        such that nodes i,j are connected in G iff their distance in \
        `distance_matrix` is strictly less than `epsilon`, and \
        compute the laplacian score of features on `G`.
    :param permutations: Generate `permutations` many random permutations \
        :math:`\\sigma` of the set of nodes of `G`, and compute the laplacian scores \
        of the features :math:`f \\circ \\sigma` for each permutation :math:`\\sigma`. \
        These additional laplacian scores are used to perform a \
        non-parametric permutation test, returning a p-value representing the \
        chance that the Laplacian would be equally as high for a randomly \
        selected permutation of the feature.
    :param return_random_laplacians: Whether to return the randomly generated Laplacians.

    :return: a pair of dictionaries (feature_data,other), with

        * `feature_data`['feature_laplacians'] - Laplacian scores of the given features, \
          shape (num_features,)
        * `feature_data`['laplacian_p_values'] - p-value of observing \
          such an extreme Laplacian, at the given number of permutations; shape (num_features,)
        * `feature_data`['laplacian_q_values'] - p-values adjusted by the \
          Benjamini-Hochsberg method; shape (num_features,)
        * `other`['random_feature_laplacians'] - if `return_random_laplacians` is true, \
          this will contain all the randomly generated laplacians of all the features. \
          Shape (num_permutations,num_features)
    """

    distribution = _to_distribution(distance_matrix, epsilon)
    laplacians: npt.NDArray[np.float64] = (
        np.negative(pearson_coefficient(feature_arr, distribution, permutations)) + 1.0
    )
    true_laplacians = laplacians[0, :]
    random_laplacians = laplacians[1:, :]
    data = {}
    data["feature_laplacians"] = true_laplacians
    data["laplacian_p_values"] = permutation_pvalue(true_laplacians, random_laplacians)
    data["laplacian_q_values"] = benjamini_hochberg(data["laplacian_p_values"])
    other = {}
    if return_random_laplacians:
        other["random_feature_laplacians"] = random_laplacians
    return (data, other)


def laplacian_scores(
    feature_arr: npt.NDArray[np.float64],
    distance_matrix: npt.NDArray[np.float64],
    epsilon: float,
    permutations: int,
    covariates: Optional[npt.NDArray[np.float64]],
    return_random_laplacians: bool,
) -> dict[str, npt.NDArray[np.float64]]:
    """
    :param feature_arr: An array of shape (N, num_features), where N is the
        number of nodes in the graph, and num_features is
        the number of features. Each column represents a feature on N elements.
        Columns should be preprocessed to remove constant features.
    :param distance_matrix: vectorform distance matrix
    :param epsilon: connect nodes of graph if their distance is less than epsilon
    :param permutations: Generate `permutations` many random permutations \
        :math:`\\sigma` of the set of nodes of `G`, and compute the laplacian scores \
        of the features :math:`f \\circ \\sigma` for each permutation :math:`\\sigma`. \
        These additional laplacian scores are used to perform a
        non-parametric permutation test, returning a p-value representing the
        chance that the Laplacian would be equally as high for a randomly \
        selected permutation of the feature.
    :param covariates: (optional) array of shape (N, num_covariates),
        or simply (N,), where N is the number \
        of nodes in the graph, and num_covariates is the number of covariates
    :param return_random_laplacians: if True, the output dictionary will contain \
        all of the generated laplacians. This will likely be the largest object in the \
        dictionary.
    :return:
        A pair of dictionaries `(feature_data, other)`.
        All values in feature_data are of shape (num_features,).

        * feature_data['feature_laplacians'] := the laplacian scores of f, shape (num_features,)
        * feature_data['laplacian_p_values'] := the p-values from the permutation test, \
          shape (num_features,)
        * feature_data['laplacian_q_values'] := the q-values from the permutation test,
          shape (num_features,)
        * (Optional, if covariates is not None) (for i in range(1, covariates.shape[0]))
          feature_data['beta_i'] := the p-value that beta_i is not zero for that
          feature; see p. 228, 'Applied Linear Statistical Models', \
          Nachtsheim, Kutner, Neter, Li. Shape (num_features,)
        * (Optional, if covariates is not None)
          feature_data['regression_coefficients_fstat_p_values'] :=
          the p-value that not all beta_i are zero, using the F-statistic, \
          see p. 226, 'Applied Linear Statistical Models', Nachtsheim, \
          Kutner, Neter, Li. Shape (num_features,)
        * (Optional, if covariates is not None)
          feature_data['laplacian_p_values_post_regression'] :=
          the p-value of the residual laplacian of the feature once the \
          covariates have been regressed out.
        * (Optional, if covariates is not None)
          feature_data['laplacian_q_values_post_regression'] :=
          the q-values from the permutation test, shape (num_features,)
        * (Optional, if covariates is not None) other['covariate_laplacians'] := \
          the laplacian scores of the covariates, shape (num_covariates,)\
          (if a matrix of covariates was supplied, else this entry will be absent)
        * (Optional, if `return_random_laplacians` is True) \
          other['random_feature_laplacians'] := the matrix of randomly generated feature \
          laplacians, shape (permutations,num_features).
        * (Optional, if `covariates` is not None and `return_random_laplacians` is True)
          other['random_covariate_laplacians'] := the matrix of randomly generated covariate
          laplacians, shape (permutations, num_covariates)

    """

    _validate(feature_arr)
    if covariates is None:
        return laplacian_score_no_covariates(
            feature_arr,
            distance_matrix,
            epsilon,
            permutations,
            return_random_laplacians,
        )
    if len(covariates.shape) == 1:
        covariates = covariates[:, np.newaxis]

    return laplacian_score_w_covariates(
        feature_arr,
        distance_matrix,
        epsilon,
        permutations,
        covariates,
        return_random_laplacians,
    )
