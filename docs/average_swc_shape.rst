.. _cluster and visualize:

Clustering and Visualization
============================

Once the user has computed the Gromov-Wasserstein distances between cells it is
possible to cluster the cells using standard clustering techniques. The Louvain
clustering algorithm and Leiden clustering algorithm are two commonly used
algorithms to identify communities within large networks; they can be adapted
to finite metric spaces by constructing a `k`-nearest-neighbors graph on top of
the metric space. CAJAL provides access to both of these clustering
algorithms. When combined with a low-dimensional embedding tool such as the
UMAP algorithm, the user can plot the clusters in a 2-dimensional embedding and
visualize them.

(... sample code goes here ...)

Average Cell Shape
------------------

When computing the Gromov-Wasserstein distance between two cells :math:`X` and
:math:`Y`, the optimal transport algorithm returns two pieces of information:

#. a *coupling matrix* which represents the optimal way to map [#]_ :math:`X`
   onto :math:`Y` so that the distortion of :math:`X` is minimized
#. the distortion induced by this optimal coupling matrix, which we call the
   Gromov-Wasserstein distance.

Assuming that the mapping of :math:`X` onto :math:`Y` is a one-to-one
correspondence [#]_, we can then relabel the points of :math:`Y` according to this
correspondence, which amounts to a reorientation of :math:`Y` so that :math:`X`
and :math:`Y` are aligned as well as is possible. After such a relabelling, the intracell
distance matrices of :math:`X` and :math:`Y` are *directly comparable*.

Given a cluster of morphologically similar neurons, it is possible to use this
observation to construct a "morphological average" of cells in the cluster by
interpolating between them in the latent space. We implement an algorithm
:func:`cajal.utilities.avg_shape_spt` to construct this morphological average:

- Identify the *medoid cell* of the cluster, i.e., find that cell whose average
  distance to the other cells is minimal.
- Use the optimal coupling matrices to reorient every other cell with the
  medoid, so that they can be directly compared.
- Rescale all intracell distance matrices to unit step size, so that
  differences in overall size do not distort the comparison.
- Within each cell, cap the distance between points at 2. This destroys
  information about *global structure* of the geodesic distances and prevents
  very distant points from having an outsize effect.
- Take the ordinary arithmetic average of all distance matrices. The distance \
  between any two points in the averaged matrix can be thought of as the average distance \
  between corresponding pairs of points in each cell in the cluster.
- Construct a shortest-path tree through the weighted graph encoded by the average distance
  matrix. This tree is our proxy for a representative neuron of the cluster; its morphology \
  is the morphology of the average neuron in the cluster.

.. autofunction:: cajal.utilities.avg_shape
.. autofunction:: cajal.utilities.avg_shape_spt


.. [#] Speaking precisely, it is incorrect to say that the coupling matrix
       gives a map from :math:`X` to :math:`Y`. In fact, it gives a
       "probabilistic" or "nondeterministic" mapping of :math:`X` onto
       :math:`Y`; it associates to each :math:`x` in :math:`X` a *probability
       distribution* over the points of :math:`Y`. It is precisely this
       distinction which makes the computation of the Gromov-Wasserstein
       distance tractable by making it reducible to an optimal transport
       problem. For our purposes we will coerce this distribution to a function
       by associating to each point :math:`x` in :math:`X` the *mode* of the
       associated distribution over nodes of :math:`Y`.
.. [#] If the map is not injective, some nodes of :math:`Y` will be repeated in
       the reoriented cell. If the map is not surjective, some nodes of
       :math:`Y` will be omitted in the reoriented cell.
