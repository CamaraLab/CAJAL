Ternary Plots
=============

The central algorithm
Compute the weighted nearest neighbors pairing, following
[Integrated analysis of multimodal single-cell data](https://www.sciencedirect.com/science/article/pii/S0092867421005833.)
This algorithm differs from the published algorithm in the paper in a few ways. In particular we do not take the L2 normalization of columns of the matrix before we begin.



.. autofunction:: cajal.ternary.ternary_distance_clusters
