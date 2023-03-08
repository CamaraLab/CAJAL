Use of the Graph Laplacian to identify features related to cell morphology
--------------------------------------------------------------------------

Suppose that we have a set of cells, :math:`G`, and a numerical feature
:math:`f : G\to \mathbb{R}`. Our question is whether the value of :math:`f`
is related to cell morphology. There are multiple ways to phrase the question
informally:
- If :math:`x` and :math:`y` are two cells in `G` with similar morphology, then
  are :math:`f(x)` and :math:`f(y)` closer on average then they would be if
  :math:`x` and :math:`y` were chosen randomly?
- If :math:`x` is a fixed cell in :math:`G`, and :math:`y` is a randomly chosen
  cell in :math:`G`, does :math:`f(y)` tend to be closer to :math:`f(x)` if
  :math:`y` is more morphologically similar to :math:`x`?
- Does cell morphology have explanatory power with regards to the value of :math:`f`?

We implement and offer a statistical test which represents one formalization of
these ideas, the *graph Laplacian*, building on work by He, Cai and Niyogi
(`Laplacian Score for Feature Selection
<https://proceedings.neurips.cc/paper/2005/hash/b5b03f06271f8917685d14cea7c6c50a-Abstract.html>`_);
see also `Multi-modal analysis and integration of single-cell morphological data <https://www.biorxiv.org/content/10.1101/2022.05.19.492525v3.full>`_.

We first use the Gromov-Wasserstein distance to equip the set of cells with an
(undirected) graph structure. Assume that we have already computed the GW-distance
between any two cells in :math:`G`. Choose also a positive real number
:math:`varepsilon` which is much less than the maximum distance between two
cells; for example, one could take the median observed difference. Now from
this point forward we will view :math:`G` as an undirected graph, where two
distinct nodes :math:`x,y` are connected if and only if
:math:`d_{GW}(x,y)<\varepsilon`.

 The graph Laplacian of :math:`f` with respect to the graph :math:`G` is defined by
.. math::

   C_G(f) = \frac{\sum_{(i,j)\in E(G)} (f(i) - f(j))^2}{\operatorname{Var}_G(f)}

where :math:`E(G)` is the set of edges in the graph `G`, :math:`i,j` range over
nodes of :math:`G`, `n(i)` is the number of neighbors of :math:`i` in
:math:`G`, and :math:`\operatorname{Var}_G(f)` is a weighted
variance of `f` where the weight of node :math:`i` is proportional to :math:`n(i)`.

The graph Laplacian takes values between 0 and 2. When the Laplacian is near
zero we interpret it as showing that the values of :math:`f(i)` and
:math:`f(j)` are linearly correlated when :math:`i,j` are connected by an edge
in the graph. If the Laplacian is close to 1, they are uncorrelated.

In cases where it is unreasonable to expect a strong linear correlation, (for
example, :math:`f` is not continuous or not modelled by any reasonable
distribution) the graph Laplacian cannot be interpreted directly. We provide a
permutation test which compares :math:`C_G(f)` to :math:`C_G(f\circ\pi)` for
many randomly chosen permutations :math:`\pi : G\to G` of the set of nodes of
the graph. If :math:`C_G(f) < C_G(f\circ\pi)` for all but a small fraction
:math:`\alpha` of the permutations, we conclude that it is unlikely that
:math:`C_G(f)` could have arisen by chance by choosing a random function with
the same values.

We also provide functionality to let the user regress out on covariates. If
:math:`g_1,\dots,g_n` are features on :math:`G`, and the user wants to know
whether :math:`C_G(f)` is below what would be expected from
:math:`C_G(g_1),\dots, C_G(g_n)`, they can fit a multilinear regression to
predict :math:`C_G(f\circ\pi)` as a sum :math:`\sum_i \beta_i
C_G(g_i\circ\pi)+\beta_0` plus a residual error term :math:`\varepsilon_i`. If
the true residual :math:`\varepsilon =C_G(f) - \widehat{C_G(f)}` is at the
lower tail end of the residuals, we conclude that :math:`f` respects the
morphology graph structure in excess of what would be predicted given with
:math:`C_G(g_1),\dots, C_G(g_n)`.


