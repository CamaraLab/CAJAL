Computing GW Distances
======================

Currently, CAJAL is equipped to deal with three kinds of input data files: neuronal tracing data (SWC files), 3D meshes (two-dimensional simplicial complexes) and 2D segmentation files (tiff files.)

Neuronal Tracing Data
---------------------

CAJAL supports neuronal tracing data in the SWC spec as specified here:
http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html

To analyze the similarity of two neurons, CAJAL first replaces each neuron by an intracell distance matrix, where the rows and columns correspond to points in the cell, and the entry at position (i, j) is the distance between points x_i and x_j. Experience has shown diminishing returns in predictive power past n = 50-100 points.

We offer some functions to help the user load and process SWC files.


.. autofunction:: sample_swc.get_sample_pts

                  After sampling the points of the neuron via ``sample_swc.get_sample_pts``, the user can compute the intracell Euclidean distance matrix between those points.
		  
.. autofunction:: sample_swc.compute_and_save_sample_pts
.. autofunction:: sample_swc.get_geodesic
.. autofunction:: sample_swc.compute_and_save_geodesic
.. autofunction:: sample_swc.compute_and_save_sample_pts_parallel
.. autofunction:: sample_swc.compute_and_save_geodesic_parallel
