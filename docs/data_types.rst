Data Types and File Formats
===========================

By a "point cloud", we mean a set of points in n-dimensional space. More
specifically, CAJAL represents a point cloud as a numpy float array of shape
(k, n), where k is the number of data points. For SWC files and \*.obj meshes,
n = 3. For tiff files, n = 2.

If point clouds are saved to a file for long-term storage or to free memory,
CAJAL's convention is to save them as comma-separated value files where each each point lies
on its own line and is represented as a triple of xyz coordinates.

You can use :func:`numpy.savetxt` to write a point cloud to file.
Here the format string fmt="%.16f" means that we retain 16 values after the decimal place.

.. code-block:: python
		
		import numpy as np
		np.savetxt("mycloud.csv", pt_cloud, delimiter=",", fmt="%.16f")
		
You can read this back with :func:`numpy.loadtxt` or :func:`pandas.read_csv`.

By an "intracell distance matrix", we mean a numpy floating point array
representing a matrix in "vector form" (see
:func:`scipy.spatial.distance.pdist` and footnote 2 of 
:func:`scipy.spatial.distance.squareform`)

A point cloud can be converted to a Euclidean distance matrix using
:func:`scipy.spatial.distance.pdist`.

.. code-block:: python
		
		from scipy.spatial.distance import pdist
		dist_mat = pdist(pt_cloud[0])
		
The function :func:`run_gw.compute_intracell_distances_one` reads a point cloud
\*.csv file into memory and returns an intracell distance matrix.

.. autofunction:: run_gw.compute_intracell_distances_one

The function :func:`run_gw.compute_intracell_distances_all` is the batch version,
operating on a directory of point cloud \*.csv files and returning a list of
intracell distance matrices.

.. autofunction:: run_gw.compute_intracell_distances_all
