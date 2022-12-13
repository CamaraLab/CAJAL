Sampling from Meshes
====================

CAJAL supports cell morphology data in the form of Wavefront \*.obj files.

A \*.obj file should consist of a series of lines, either
- comments starting with "#" (discarded)
- a vertex line, starting with "v" and followed by three floating point xyz coordinates
- a face line, starting with f and followed by three integers which are indices for the vertices

All other lines will be ignored or discarded.

For examples of compatible mesh files see the folder /CAJAL/data/obj_files in the CAJAL Git repository.

The sample_mesh.py file contains functions to help the user sample points from
an \*.obj file and compute the geodesic distances between points.

.. autofunction:: sample_mesh.read_obj
