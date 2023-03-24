Computing Average Cell Shapes
=============================

When computing the Gromov-Wasserstein distance between two cells :math:`X` and
:math:`Y`, the optimal transport algorithm returns two pieces of information:

#. A *coupling matrix*, which represents the optimal probabilistic mapping of :math:`X`
   onto :math:`Y` that minimizes the distortion.
#. The distortion induced by this optimal coupling matrix, which is the
   Gromov-Wasserstein distance.

We can utilize the coupling matrix to construct a morphological average of a group or
cluster of cells. CAJAL implements an algorithm called :func:`avg_shape_spt` to construct this
morphological average. In brief, the algorithm proceeds as follows:

- Identify the *medoid cell* of the cluster, which is the cell that has the lowest average
  distance to the other cells.
- Use the optimal coupling matrices to reorient every other cell with respect to the
  medoid, so that they can be directly compared.
- Rescale all intracellular distance matrices to unit step size to ensure that
  differences in overall size do not distort the comparison.
- Cap the distance between points within each cell at 2. This destroys
  information about the global structure of the geodesic distances, preventing
  very distant points from having an outsize effect.
- Compute the arithmetic mean of all distance matrices, where the distance
  between any two points in the averaged matrix is the average distance
  between the corresponding pairs of points in each cell in the cluster.
- For neurons, construct a shortest-path tree through the weighted graph encoded by the average distance
  matrix. This tree represents the average neuronal morphology of the cluster.


