Sampling from SWC Files
=======================

CAJAL supports neuronal tracing data in the SWC spec as specified here: http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html

The sample_swc.py file contains functions to help the user sample points from an \*.swc file.

.. autofunction:: sample_swc.get_sample_pts

After sampling the points of the neuron via ``sample_swc.get_sample_pts``, the user can compute the intracell Euclidean distance matrix between those points.
		  
.. autofunction:: sample_swc.compute_and_save_sample_pts
.. autofunction:: sample_swc.get_geodesic
.. autofunction:: sample_swc.compute_and_save_geodesic
.. autofunction:: sample_swc.compute_and_save_sample_pts_parallel
.. autofunction:: sample_swc.compute_and_save_geodesic_parallel

