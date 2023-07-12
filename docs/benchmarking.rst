Inferring Associations with Cell Morphology
===========================================

The Laplacian Score is a statistical test implemented in CAJAL to determine whether
differences in a numerical feature assigned to cells, :math:`f : G\to \mathbb{R}`, such as the expression of a gene or the genotype
of the cell in a given locus, are related to differences in cell morphology. Specifically,
the Laplacian Score answers the question: if :math:`x` and :math:`y` are two cells
with similar morphology, are :math:`f(x)` and :math:`f(y)` closer on average than
if :math:`x` and :math:`y` were chosen randomly?

To perform this analysis, CAJAL uses the Gromov-Wasserstein distance between every pair
of cells to construct an undirected graph :math:`G` where nodes represent cells and edges
connect cells with distances less than :math:`\varepsilon`, a user-specified positive real
parameter. The Laplacian score of :math:`f` with respect to the graph :math:`G` is
positive number defined by

.. math::

		C_G(f) = \frac{\sum_{(i,j)\in E(G)} (f(i) - f(j))^2}{\operatorname{Var}_G(f)}


where :math:`E(G)` is the set of edges in the graph, :math:`i,j` range over
nodes of :math:`G`, and :math:`\operatorname{Var}_G(f)` is the weighted
variance of `f,` where the weight of node :math:`i` is proportional to
the number of neighbors of :math:`i` in :math:`G`. When the Laplacian Score is close to
zero, this indicates that the values of :math:`f` tend to be similar between
connected cells.

To test the significance of the Laplacian Score, CAJAL provides a permutation test
that shuffles the values of :math:`f` across the nodes of :math:`G` to generate a null
distribution, from which a p-value can be computed. Additionally, CAJAL supports
regression analysis to account for the influence of other covariates,
:math:`g_1,\dots,g_n`, defined on :math:`G`. Users can fit a multivariate linear
regression model to remove the dependence of :math:`C_G(f)` on
:math:`C_G(g_1),\dots, C_G(g_n)`, and evaluate whether the Laplacian Score of :math:`f`
is below what would be expected from the covariate features.

Overall, the Laplacian Score implemented in CAJAL provides a flexible approach
for analyzing the relationship between cell morphology and numerical features, with the
ability to account for other covariates and assess statistical significance.

More information about the theoretical foundations of the Laplacian score can be found at:

\- Govek, K. W., et al. `CAJAL enables analysis and integration of single-cell morphological data using metric geometry. <https://doi.org/10.1038/s41467-023-39424-2>`_ Nature Communications 14 (2023) 3672.

\- Govek, K. W., Yamajala, V. S., and Camara, P. G. `Clustering-Independent Analysis of Genomic Data using Spectral Simplicial Theory. <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007509>`_ PLOS Computational Biology 15 (2019) 11.

\- He, X., Cai, D., and  Niyogi, P. `Laplacian Score for Feature Selection <https://proceedings.neurips.cc/paper_files/paper/2005/file/b5b03f06271f8917685d14cea7c6c50a-Paper.pdf>`_ In Advances in neural information processing systems (2005) 507-514.
