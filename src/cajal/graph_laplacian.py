import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import squareform
from scipy.stats import rankdata, t, f
from typing import Optional

def pearson_coefficient(
        feature_arr : npt.NDArray[np.float_],
        joint_dist : npt.NDArray[np.float_],
        permutations : int
    ) -> npt.NDArray[np.float_]:

    """
    Let X, Y be two random variables on the finite set {0,...,N-1}, jointly distributed with \
    distribution described in the matrix joint_dist; joint_dist[i][j] = p(X=i,Y=j). \
    Let f be a function from {0,...,N-1} to |R. In what follows we call f a "feature." \
    Then f(X) and f(Y) are random variables. \
    Here we compute the Pearson correlation coefficient of f(X) and f(Y), and return it. \
    We also compute the Pearson correlation coefficient of f(sigma \circ X), f(sigma \circ Y) \
    for `permutations` many randomly selected permutations sigma of the set {0,\dots,n-1}. \
    (Caveat: It it is assumed that X and Y are identically distributed.) \

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
    pearson_coefficient_list=[]

    # Allocate space for temporary storage matrices
    means = np.zeros((num_features,))
    f_tildes=np.zeros((N,num_features),dtype=np.float_)
    tempvar1 = np.zeros((num_features,N),dtype=np.float_)
    tempvar2 = np.zeros((num_features,N),dtype=np.float_)
    rng= np.random.default_rng()
    covariance=np.zeros((num_features,),dtype=np.float_)
    variance=np.zeros((num_features,),dtype=np.float_)

    for i in range(permutations+1):
        if i > 0:
            new_index_list=rng.permutation(N)
            joint_dist = joint_dist[new_index_list,:]
            joint_dist = joint_dist[:,new_index_list]
        X = np.sum(joint_dist,axis=0)

        # Compute the means of the features wrt the given distribution.
        np.dot(feature_arr.T,X,out=means)
        # Normalize the features by subtracting the means.
        np.subtract(feature_arr,means[np.newaxis,:],out=f_tildes)
        # Compute the covariance of f(X1) with f(X2).
        np.matmul(f_tildes.T,joint_dist,out=tempvar1)
        np.multiply(tempvar1,f_tildes.T,out=tempvar2).sum(axis=1,out=covariance)

        # Compute var(f(X1))=var(f(X2)).
        np.multiply(f_tildes.T,f_tildes.T,out=tempvar1)
        tempvar1.dot(X,out=variance)
        pearson_coefficient_list.append(covariance/variance)

    return np.stack(pearson_coefficient_list,axis=0)

def validate(feature_arr : npt.NDArray[np.float_]):
    for i in range(feature_arr.shape[1]):
        v = feature_arr[:,i]
        if np.all(v == v[0]):
            raise Exception("feature_arr[:," + \
                            str(i) + \
                            "] is constant. \
                            This leads to divide-by-zero errors. \
                            Please clean data by removing constant columns.")

def to_distribution(
        dist_mat : npt.NDArray[np.float_],
        epsilon : float) -> npt.NDArray[np.float_]:
    """
    Convert a distance matrix on N elements to a probability distribution on
    N x N elements by a two step process:

    * create a graph adjacency matrix where two edges are connected iff their
      distance is less than epsilon
    * normalize the graph adjacency matrix so that it is a probability distribution. 
    
    :param dist_mat: vectorform distance matrix
    :param epsilon: threshold to connect two nodes in the graph
    :return: an n x n probability matrix representing a joint distribution on
    two variables.
    """
    adjacency_matrix = (squareform(dist_mat) < epsilon).astype(np.int_)
    return adjacency_matrix / np.sum(adjacency_matrix)

def multilinear_regression(
        X: npt.NDArray[np.float_],
        Y: npt.NDArray[np.float_]
) -> tuple[npt.NDArray[np.float_],
           npt.NDArray[np.float_],
           npt.NDArray[np.float_],
           npt.NDArray[np.float_],
           npt.NDArray[np.float_]]:
    """
    Compute the least-squares solution b to Y = Xb
    :param X: shape (n, p)
    :param Y: shape (n, m)
    
    :return: A tuple (b, SSE, SSR), where
    * b is coefficicient matrix minimizing the sum of squared errors ||Y - Xb||, shape (p,m)
    * e, the matrix of residuals, shape (n,m)
    * SSE, the sum of squared errors (residuals), shape (m,)
    * SSR, the sum of squares of the regression, shape (m,)
    * s2b, the sample variance-covariance matrix of the observed coefficient vector b.
    """

    b, SSE = np.linalg.lstsq(X,Y,rcond=None)[0:2]
    # b = lstsq_results[0]
    # SSE = lstsq_results[1]
    hat_Y = np.matmul(X,b)
    Y_means = np.sum(Y,axis=0)/Y.shape[0]
    Ya = np.subtract(hat_Y, Y_means[np.newaxis,:])
    SSR = np.dot(Ya.T, Ya)
    assert SSR.shape == (Y.shape[1],)
    XTX_inv = np.linalg.inv(np.matmul(X.T,X))
    n = Y.shape[0]
    p = X.shape[1]
    s2b = XTX_inv/(n-p)
    return b, np.subtract(Y,hat_Y), SSE, SSR, s2b


def benjamini_hochberg(p_values : npt.NDArray) -> npt.NDArray:
    """
    Takes an array of p-values, and returns an array of q-values.
    """
    ranked_p_values = rankdata(p_values)
    q_values = p_values * len(p_values)/ranked_p_values
    q_values[q_values >= 1]=1
    return q_values


def percentile(
        X : npt.NDArray[np.float_],
        A: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    """
    :param X: shape (k,)
    :param A: shape (n,k)

    :return: a vector V of shape (k,) where V[i] tells the percentile of \
    X among elements of A (X[i] is greater than fraction V[i] of elements in A[i])
    """
    return np.greater(
        X[np.newaxis,:],
        A).astype(np.int_).sum(axis=0)/(A.shape[0]+1)


def graph_laplacian_w_covariates(
        feature_arr : npt.NDArray[np.float_],
        distance_matrix : npt.NDArray[np.float_],
        epsilon : float,
        permutations : int,
        covariates : npt.NDArray[np.float_],
        return_random_laplacians : bool
) -> dict[str,npt.NDArray[np.float_]]:
    """
    :param feature_arr: An array of shape (N ,num_features), where N is the \
    number of nodes in the graph, and num_features is \
    the number of features. Each column represents a feature on N elements. \
    Columns should be preprocessed to remove constant features.

    :return: A dictionary `data` with
    * data['feature_laplacians'] := the graph laplacians of f, shape (num_features,)
    * data['covariate_laplacians'] := the graph laplacians of the covariates, shape (num_covariates,)
    * data['laplacian_p_values'] := the p-values from the permutation test, shape (num_features,)
    * data['laplacian_q_values'] := the q-values from the permutation test, shape (num_features,)
    * data['regression_coefficients_beta_p_values'] := for i in range(1,covariates), the p-value that beta_i is not zero \
      for that feature; see p. 228, 'Applied Linear Statistical Models', \
      Nachtsheim, Kutner, Neter, Li. Shape (num_features,num_covariates)
    * data['regression_coefficients_fstat_p_values'] := the p-value that not all beta_i are zero, using the F-statistic, \
      see p. 226, 'Applied Linear Statistical Models', Nachtsheim, \
      Kutner, Neter, Li. Shape (num_features,)
    * data['laplacian_p_values_post_regression'] := the p-value of the residual laplacian of the feature once the \
      covariates have been regressed out.
    * data['laplacian_q_values_post_regression'] := the q-values from the permutation test, shape (num_features,)
    * (Optional, if `return_random_laplacians` is True) \
      data['random_feature_laplacians'] := the matrix of randomly generated feature laplacians, \
      shape (num_features, permutations).
    * (Optional, if `return_random_laplacians` is True) \
      data['random_covariate_laplacians'] := the matrix of randomly generated covariate laplacians, \
      shape (num_covariates, permutations)
    """

    validate(feature_arr)
    distribution = to_distribution(distance_matrix,epsilon)
    N = feature_arr.shape[0]
    num_features : int = feature_arr.shape[1]
    if len(covariates.shape)==1:
        covariates = covariates[:,np.newaxis]
    num_covariates = covariates.shape[1]
    A = np.concatenate((feature_arr,covariates),axis=1)
    assert A.shape == (N, num_features + num_covariates)
    laplacians : npt.NDArray[np.float_] =\
        np.negative(pearson_coefficient(A,distribution,permutations)) + 1.0

    # Extract the four quadrants from the laplacians matrix.
    # true feature laplacians
    tfl = laplacians[0,:num_features]
    assert tfl.shape == (num_features,)
    # random feature laplacians
    rfl = laplacians[1:,:num_features]
    assert rfl.shape == (permutations, num_features)
    # true covariate laplacians
    tcl  = np.concatenate((laplacians[0,num_features:], np.array([1.0])),
                          axis=0)
    assert tcl.shape == (num_covariates+1,)
    # random covariate laplacians
    rcl = np.concatenate(
        (laplacians[1:,num_features:],
         np.full((permutations,1), fill_value=1,dtype=np.float_)),
        axis=1)
    assert rcl.shape == (permutations,num_covariates+1)

    b, rfl_resids, SSE, SSR, s2b = multilinear_regression(rcl,rfl)
    assert b.shape == (num_covariates+1,num_features)

    MSR = SSR/(num_covariates - 1)
    if (permutations <= num_covariates):
        raise Exception("Must be more permutations than covariates.")
    MSE = SSE/(permutations - num_covariates)
    sb = np.sqrt(np.diag(s2b))
    # Compute the t-statistic to decide whether beta is 
    t_stat = (b/sb)[:-1]
    p_betas = t.sf(t_stat, permutations - num_covariates) * 2
    f_stat = MSR/MSE
    p_all_betas = f.sf(f_stat, num_covariates-1, permutations-num_covariates)
    tfl_resid = tfl - np.matmul(tcl,b)

    data = {}
    data['feature_laplacians'] = tfl
    data['covariate_laplacians'] = tcl
    data['laplacian_p_values'] = percentile(tfl,rfl)
    data['laplacian_q_values']= benjamini_hochberg(data['laplacian_p_values'])
    data['regression_coefficients_beta_p_values']=p_betas
    data['regression_coefficients_fstat_p_values']=p_all_betas
    data['laplacian_p_values_post_regression']=percentile(tfl_resid,rfl_resids)
    data['laplacian_q_values_post_regression']=\
        benjamini_hochberg(data['laplacian_p_values_post_regression'])
    return data

def graph_laplacians(
        feature_arr : npt.NDArray[np.float_],
        distance_matrix : npt.NDArray[np.float_],
        epsilon : float,
        permutations : int,
        return_random_laplacians : bool
) -> npt.NDArray[np.float_]:
    """
    :param feature_arr: An array of shape (N ,num_features), where N is the \
    number of nodes in the graph, and num_features is \
    the number of features. Each column represents a feature on N elements. \
    Columns should be preprocessed to remove constant features.

    :param distance_matrix: A vector-form distance matrix containing pairwise distances between \
    points in a space. Can be any vector of length N * (N-1)/2 for some N.
    :param epsilon: From `distance_matrix` we will build an undirected graph G such that nodes i,j \
    are connected in G iff their distance in `distance_matrix` is strictly less than `epsilon`, and \
    compute the graph laplacian of features on `G`.

    :param permutations: Generate `permutations - 1` many random permutations `sigma` of the set \
    of nodes of `G`, and \
    compute the graph laplacians of the features `f \circ \sigma` for each permutation `sigma`. \
    These additional laplacian scores are used to perform a non-parametric permutation test, \
    returning a p-value \
    representing the chance that the laplacian would be equally as high for a randomly \
    selected permutation of the feature.

    :covariates: An optional (N, k) matrix of covariate features, where N is the number of nodes \
    in the graph and k is the number of covariates.
    
    :return: A matrix of shape (3,num_features), where
    - the first row contains the graph laplacians for the features `f`
    - the second row contains the p-values from the permutation testing
    - the third row contains q-values which result from adjusting the p-values by a \
    Benjamini-Hochberg procedure
    """
    
    validate(feature_arr)
    N = feature_arr.shape[0]
    distribution = to_distribution(distance_matrix,epsilon)
    num_features : int = feature_arr.shape[1]
    laplacians : npt.NDArray[np.float_] =\
        np.negative(pearson_coefficient(feature_arr,distribution,permutations)) + 1.0
    true_laplacians = laplacians[0,:]
    random_laplacians = laplacians[1:,:]
    p_values = np.less(random_laplacians, laplacians).astype(np.int_).sum(axis=0)/permutations
    q_values = benjamini_hochberg(p_values)
    return np.stack((laplacians,p_values,q_values),axis=0)
