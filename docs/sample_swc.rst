Processing SWC Files
====================

CAJAL supports neuronal tracing data in the SWC spec as specified here: http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html

The sample_swc.py file contains functions to help the user sample points from an \*.swc file.

.. warning:: In the Allen Brain 


.. autofunction:: sample_swc.get_sample_pts
.. autofunction:: sample_swc.get_geodesic
		  
After sampling the points of the neuron via :func:`sample_swc.get_sample_pts`, the user can compute the intracell Euclidean distance matrix between those points.

.. autofunction:: sample_swc.compute_and_save_intracell_all
