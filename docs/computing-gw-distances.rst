Computing GW Distances
======================

Currently, Cajal is equipped to deal with three kinds of input data files: neuronal tracing data (SWC files), 3D meshes (two-dimensional simplicial complexes) and 2D segmentation files (tiff files.)

Neuronal Tracing Data
---------------------

Cajal supports neuronal tracing data in the SWC spec as specified here:
http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html

.. autofunction:: sample_swc.read_swc
