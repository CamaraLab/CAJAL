Running Gromov-Wasserstein
==========================


The GW distance is calculated using the same function whether the distance matrices represent the Euclidean or geodesic metric.

We will agree upon the following two standard representations of a distance matrix. In this section, a "distance matrix" on :math:`n` objects :math:`x_0,\dots, x_{n-1}` refers to one of the following:

* A one-dimensional floating-point numpy array :math:`v` of length :math:`(n-1)\times n/ 2`, such that the distance between objects :math:`x_i` and :math:`x_j` (for :math:`i < j`) is stored in :math:`v` at index :math:`{n \choose 2} - {n - i \choose 2} + (j - i - 1)` (see :func:`scipy.spatial.distance.pdist` and footnote 2 of :func:`scipy.spatial.distance.squareform`).
* A text file with :math:`(n - 1) \times n/2` lines, with a single floating-point number on each line, with indices for the distance :math:`d(x_i,x_j)` computed in the same way.


.. autofunction:: run_gw.compute_GW_distance_matrix


