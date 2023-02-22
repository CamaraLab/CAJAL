import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

def pearson_coefficient(
        feature_arr : npt.NDArray[np.float_],
        joint_dist : npt.NDArray[np.float_],
        permutations : int
    ) -> npt.NDArray[np.float_]:

    """
    Let X, Y be two random variables on the finite set {0,...,N-1}, jointly distributed with \
    distribution described in the matrix joint_dist; joint_dist[i][j] = p(X=i,Y=j).

    Let f be a function from {0,...,N-1} to |R. In what follows we call f a "feature."

    Then f(X) and f(Y) are random variables.

    Here we compute the Pearson correlation coefficient of f(X) and f(Y), and return it.

    We also compute the Pearson correlation coefficient of f(sigma \circ X), f(sigma \circ Y) \
    for `permutations` many randomly selected permutations sigma of the set {0,\dots,n-1}.

    (Caveat: It it is assumed that X and Y are identically distributed.)
    
    :param feature_arr: array of shape (N ,num_features), where N is the \
    number of possible values the random variables X, Y can take on, and num_features is \
    the number of features. Each column represents a feature on N elements.

    :param joint_dist: an (N,N) matrix whose entries sum to 1, \
    where the sum of entries in the i-th row is equal to the sum of entries in the i-th column. \

    :permutations: Recompute the Pearson correlation coefficient for
    `permutations` many randomly chosen permutations \ of the underlying set
    {0,\dots, n-1}.

    :return: A matrix `A` of shape (permutations,num_features) where A[0,j] is the Pearson \
    correlation coefficient of f_j(X) with f_j(Y) (where f_j is the j-th feature \
    in `feature_arr` ) and A[i,j] (for i > 0) is the Pearson correlation coefficient of \
    f_j(sigma_i \circ X) with f_j(sigma_i \circ X), where sigma_i is a randomly chosen permutation \
    of the set {0,\dots, n-1}.
    """

    # The probabilities for the random variable X.
    X = np.sum(joint_dist,0)
    # The number of different values X and Y can take on.
    N = X.shape[0]
    num_features = feature_arr.shape[1]
    assert N == feature_arr.shape[0]
    pearson_coefficient_list=[]

    # Allocate space for temporary storage matrices
    means = np.zeros((num_features,))
    f_tildes=np.zeros((N,num_features),dtype=np.float_)
    tempvar1 = np.zeros((num_features,N),dtype=np.float_)
    tempvar2 = np.zeros((num_features,N),dtype=np.float_)
    rng= np.random.default_rng()
    covariance=np.zeros((num_features,),dtype=np.float_)
    variance=np.zeros((num_features,),dtype=np.float_)

    for i in range(permutations):
        if i > 0:
            new_index_list=rng.permutation(N)
            joint_dist = joint_dist[new_index_list,:]
            joint_dist = joint_dist[:,new_index_list]
            
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

def graph_laplacian(
        feature_arr : npt.NDArray[np.float_],
        distance_matrix : npt.NDArray[np.float_],
        epsilon : float,
        permutations : int
) -> npt.NDArray[np.float_]:
    """
    :param feature_arr: An array of shape (N ,num_features), where N is the \
    number of nodes in the graph, take on, and num_features is \
    the number of features. Each column represents a feature on N elements.

    :param distance_matrix: A vector-form distance matrix containing pairwise distances between \
    points in a space. Can be any vector of length N * (N-1)/2 for some N.
    :param epsilon: From `distance_matrix` we will build an undirected graph G such that nodes i,j \
    are connected in G iff their distance in `distance_matrix` is strictly less than `epsilon`, and compute \
    the graph laplacian of features on `G`.

    :param permutations: Generate `permutations - 1` many random permutations `sigma` of the set of nodes of `G`, and \
    compute the graph laplacians of the features `f \circ \sigma` for each permutation `sigma`. \
    These additional laplacian scores are used to perform a non-parametric permutation test, returning a p-value \
    representing the chance that the laplacian would be equally as high for a randomly selected permutation of the feature.
    
    :return: A matrix of shape (num_features,3), where
    - the first row contains the graph laplacians for the features `f`
    - the second row contains the p-values from the permutation testing
    - the third row contains q-values which result from adjusting the p-values by a Benjamini-Hochberg procedure
    """

    adjacency_matrix = (squareform(distance_matrix) < epsilon).astype(np.int_)
    distribution = adjacency_matrix / np.sum(adjacency_matrix)
    laplacians = np.negative(pearson_coefficient(feature_arr,adjacency_matrix,permutations)) + 1
    true_laplacians = laplacians[0,:]
    random_laplacians = laplacians[1:,:]
    num_features=feature_arr.shape[1]
    p_values = np.less(random_laplacians, true_laplacians).astype(np.int_).sum(axis=0)/permutations
    ranked_p_values = rankdata(p_values)
    q_values = p_values * len(p_values)/ranked_p_values
    return np.stack((true_laplacians,p_values,q_values),axis=0)
    
