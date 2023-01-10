Processing Obj Meshes
=====================

CAJAL supports cell morphology data in the form of Wavefront \*.obj files.

A \*.obj file should consist of a series of lines, either
- comments starting with "#" (discarded)
- a vertex line, starting with "v" and followed by three floating point xyz coordinates
- a face line, starting with f and followed by three integers which are indices for the vertices

All other lines will be ignored or discarded.

For examples of compatible mesh files see the folder /CAJAL/data/obj_files in the CAJAL Git repository.

The sample_mesh.py file contains functions to help the user sample points from
an \*.obj file and compute the geodesic distances between points.

.. py:module:: CAJAL.lib.sample_mesh

.. class:: sample_mesh.VertexArray

	   A :class:`sample_mesh.VertexArray` is a numpy array of shape (n, 3),
	   where n is the number of vertices in the mesh.

	   Each row of a :class:`sample_mesh.VertexArray` is an XYZ coordinate triple for a point in the mesh.
	   
	   :value: numpy.typing.NDArray[numpy.float\_]

.. class:: sample_mesh.FaceArray

	   A FaceArray is a numpy array of shape (m, 3) where m is the number
	   of faces in the mesh.  Each row of a FaceArray is a list of three
	   natural numbers, corresponding to indices in the corresponding
	   VertexArray, representing triangular faces joining those three
	   points.

	  :value: numpy.typing.NDArray[numpy.int\_]

.. autofunction:: sample_mesh.read_obj


