Running Gromov-Wasserstein
==========================

.. py:module:: src.cajal.run_gw

.. class:: cajal.run_gw.Distribution

	   A :class:`run_gw.Distribution` is a numpy array of shape (n,), with values
	   nonnegative and summing to 1,
	   where n is the number of points in the set.
	   
	   :value: numpy.typing.NDArray[numpy.float\_]

.. class:: cajal.run_gw.DistanceMatrix

	   A DistanceMatrix is a numpy array of shape (n, n) where n is the
	   number of points in the space; it should be a symmetric nonnegative matrix
	   with zeros along the diagonal.

	  :value: numpy.typing.NDArray[numpy.float\_]

.. autofunction:: cajal.run_gw.icdm_csv_validate
.. autofunction:: cajal.run_gw.cell_iterator_csv
.. autofunction:: cajal.run_gw.cell_pair_iterator_csv
.. autofunction:: cajal.run_gw.gw_pairwise_parallel
.. autofunction:: cajal.run_gw.compute_gw_distance_matrix
