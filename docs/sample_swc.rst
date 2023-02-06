Processing SWC Files
====================

CAJAL supports neuronal tracing data in the SWC spec as specified here:
http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html

The sample_swc.py file contains functions to help the user sample points
from an \*.swc file.

.. autoclass:: sample_swc.NeuronNode
.. autoclass:: sample_swc.NeuronTree
.. class:: sample_mesh.SWCForest

	   A :class:`sample_swc.SWCForest` is a list of
	   :class:`sample_swc.NeuronTree`'s. It is intended to be used
	   to represent a list of all connected components from an SWC
	   file. An SWCForest represents all contents of one SWC file.


.. autofunction:: sample_swc.read_swc
.. autofunction:: sample_swc.cell_iterator
.. autofunction:: sample_swc.filter_forest
.. autofunction:: sample_swc.get_sample_pts_euclidean
.. autofunction:: sample_swc.icdm_euclidean

Some features of the sampling functionality regarding geodesic distance do not depend on the coordinates of the points in the neuron. It is more convenient in this case to convert the tree to a different kind of data structure where the weights are an explicit field and need not be recomputed every time they are needed.

.. autoclass:: sample_swc.WeightedTreeRoot
.. autoclass:: sample_swc.WeightedTreeChild
.. class:: sample_mesh.WeightedTree

	   A :class:`sample_swc.WeightedTree` is either a
	   :class:`sample_swc.WeightedTreeRoot` or a
	   :class:`sample_swc.WeightedTreeChild`.

.. autofunction:: sample_swc.WeightedTree_of
.. autofunction:: sample_swc.geodesic_distance
.. autofunction:: sample_swc.get_sample_pts_geodesic
.. autofunction:: sample_swc.icdm_geodesic
.. autofunction:: sample_swc.compute_intracell_one
.. autofunction:: sample_swc.compute_and_save_intracell_all_csv

