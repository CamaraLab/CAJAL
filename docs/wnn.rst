Weighted Nearest Neighbors
==========================

This page documents an implementation of the weighted nearest neighbors algorithm described in
`Integrated analysis of multimodal single-cell data <https://www.sciencedirect.com/science/article/pii/S0092867421005833>`_.

Given a set of cells :math:`c_1,c_2,\dots,c_k` we may measure or observe the cells with different technologies, collecting (for example)

- electrophsyiological feature vectors :math:`e_1,\dots, e_k \in \mathbb{R}^n`
- transcriptomic feature vectors from a reduced :math:`e_1,\dots, e_k \in \mathbb{R}^m`
- pairwise Gromov-Wasserstein distances :math:`m_{ij}=d_GW(c_i,c_j)` for all :math:`i,j`
- (...)

Here we restrict ourselves to measurements which can be expressed as

- a list of :math:`k` vectors in some finite-dimensional Euclidean space
- a pairwise distance or dissimilarity matrix

The weighted nearest neighbor algorithm integrates measurements from
these different modalities into a single notion of similarity which
reflects the overall/combined similarity of the two cells with respect
to all of the constituent modalities. It does not assign equal weight
to each modality; rather, at each point in the space, each component
is weighted according to the information it contributes at that point.

We have added two novel aspects to the original algorithm in order to
incorporate distance matrices such as the Gromov-Wasserstein distance
matrix. Specifically, we have to

- estimate the dimensionality of the latent space from the distance matrix
- embed the distance matrix into a Euclidean space of the appropriate dimension.

To estimate the dimensionality of the latent space, we chose the MADA
algorithm, based on the general benchmarking paper
`Intrinsic Dimension Estimation: Relevant Techniques and a Benchmark Framework <https://onlinelibrary.wiley.com/doi/10.1155/2015/759567>`_
which notes that MADA performs well in applications such as
geophysical signal processing and that the MADA paper authors "prove
the consistency in probability of the presented estimators [and]
derive upper bounds on the probability of the estimation-error for
large enough values of :math:`N`."

To embed the distance matrix into Euclidean space, we use Isomap, which is a standard algorithm.

.. autoclass:: cajal.wnn.Modality
.. autofunction:: cajal.wnn.wnn
