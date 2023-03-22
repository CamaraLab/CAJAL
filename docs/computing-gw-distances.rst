Computing GW Distances
======================

To compute the Gromov-Wasserstein (GW) distance between intracellular distance matrices,
users can employ the function :func:`cajal.run_gw.compute_gw_distance_matrix`.

This section assumes that the user has already obtained the intracellular
distance matrices for their cells. It is worth noting that the GW distance
can be calculated using the same function regardless of how the intracellular
distance matrices were computed and whether they represent the Euclidean or
geodesic metric.

To use the function, the user should provide the path to an input \*.csv
database containing the intracellular distance matrices through the argument
`intracell_db_loc`. The output GW distance matrix will be saved in a new \.csv
file specified by the argument `gw_csv`.

.. code-block:: python

        run_gw.compute_and_save_gw_distance_matrix(
            intracell_db_loc = "/home/jovyan/CAJAL/CAJAL/data/swc_icd.csv",
            gw_csv = "/home/jovyan/CAJAL/CAJAL/data/gw_dists.csv",
            save_mat = False)

By default, the coupling matrices which represent the best possible pairing
between two cells are not retained, as indicated by the argument `save_mat`.
CAJAL provides functionality to compute an "average cell shape" given a family
of cells. When the cells are neurons, CAJAL can display this average cell
shape, providing a visual representation of the prototypical features of
neurons in a given cluster. Currently, this average cell morphology is the only
part of CAJAL which depends on the coupling matrices.

Numpy should automatically parallelize the computation across multiple cores.
Users on Windows can check the process
manager, while those on Unix-based systems can use the "top" command to verify
that all cores are being utilized.

.. warning::

   Note that setting `save_mat` to True will generate a large amount of data,
   which scales quadratically with the number of input cells. For example, if
   there are 150 cells with 50 sampled points each, the resulting database size
   may be approximately 180MB. File IO may also become a bottleneck in the
   computation. Therefore, users should exercise caution when setting
   `save_mat` to True, especially when working with a large number of cells.

