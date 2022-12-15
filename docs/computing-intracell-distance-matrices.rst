Computing Intracell Distance Matrices and Point Clouds
======================================================

CAJAL represents a cell as a finite list of points, together with a distance
between each point. More specifically, this data is represented as a matrix
(the "intracell distance matrix") where the rows and columns correspond to
points in the cell, and the entry at position (i, j) is the distance between
points x_i and x_j. Experience has shown diminishing returns in predictive
power past n = 50-100 points. In order to compute the Gromov-Wasserstein
distance between two cells, the user must first convert their cell morphology
data into this format. This section discusses the functionality CAJAL provides
for this purpose.

Currently, CAJAL is equipped to deal with three kinds of input data files:
neuronal tracing data (SWC files), 3D meshes (two-dimensional simplicial
complexes) and 2D segmentation files (tiff files.)

Euclidean vs. geodesic distances
--------------------------------

CAJAL requires the user to encode a cell's morphology as an intracell distance
matrix. There are at least two reasonable ways to define the distance between
two points in a cell:

1. the ordinary, straight-line Euclidean
   distance through space ("as the crow flies")
2. the geodesic distance, the length of the shortest path
   *through the body of the cell.*

The choice between the Euclidean and geodesic distances will affect what kinds
of cell deformations CAJAL regards as significant or relevant when trying to
deform one cell into another. If the user
represents two cells A and B by Euclidean distance matrices, then CAJAL will
regard any translation, rotation or mirroring of A across a plane as
insignificant, in the sense that such operations on A are not considered a
"distortion" and these deformations do not increase the GW distance. However,
bending or flexing A is a distortion and increases the GW distance. One can
visualize the neuron in this case as made of a thick wire which takes some
effort to bend; the GW distance measures the cost of distorting or deforming
it.

On the other hand, if the user represents two neurons by their geodesic
distance matrices, then translation, rotation, mirroring, bending and flexing A
are all now "irrelevant" operations which do not increase the GW distance, but
stretching, elongating or compressing A will be considered a costly
deformation. One can visualize the neuron in this case as made of thin rubber,
which is easily bent but takes some work to stretch or compress along any
segment.

To illustrate the distinction, suppose we have two pieces of string, A
and B. Both A and B are twelve inches long. A is laid out in a straight line,
whereas B is tightly coiled. If A and B are represented by Euclidean distance
matrices, then the Gromov-Wasserstein distance between them will be nontrivial,
because one must bend B to straighten it out into a line segment. However, if
they are represented by their geodesic distance matrices, then the
Gromov-Wasserstein distance will be zero.  One can deform A into B
without any stretching or elongating, as they are the same length.


Basic Data and File Formats
---------------------------

By a "point cloud", we mean a set of points in n-dimensional space. More
specifically, CAJAL represents a point cloud as a numpy float array of shape
(k, n), where k is the number of data points. For SWC files and \*.obj meshes,
n = 3. For tiff files, n = 2.

If point clouds are saved to a file for long-term storage or to free memory,
they should be saved as comma-separated value files where each each point lies
on its own line and is represented as a triple of xyz coordinates.

You can use :func:`numpy.savetxt` to write a point cloud to file.
Here the format string fmt="%.16f" means that we retain 16 values after the decimal place.

.. code-block:: python

		import numpy as np
		np.savetxt("mycloud.csv", pt_cloud, delimiter=",", fmt="%.16f")

You can read this back with :func:`numpy.loadtxt` or :func:`pandas.read_csv`.

By an "intracell distance matrix", we mean either a numpy floating point array
representing a matrix in "vector form" (see
:func:`scipy.spatial.distance.pdist` and footnote 2 of 
:func:`scipy.spatial.distance.squareform`) or a Python
:func:`multiprocessing.Array`.

A point cloud can be converted to a Euclidean distance matrix using
:func:`scipy.spatial.distance.pdist`.

.. code-block:: python
		
		from scipy.spatial.distance import pdist
		dist_mat = pdist(pt_cloud[0])

The function `run_gw.compute_intracell_distances_one` reads a point cloud
\*.csv file into memory and returns an intracell distance matrix.

.. autofunction:: run_gw.compute_intracell_distances_one

The function `run_gw.compute_intracell_distances_all` is the batch version,
operating on a directory of point cloud \*.csv files and returning a list of
intracell distance matrices.

.. autofunction:: run_gw.compute_intracell_distances_all


Neuronal Tracing Data
---------------------

CAJAL supports neuronal tracing data in the SWC spec as specified here:
http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html

We offer some functions to help the user load and process SWC files, see
:doc:`sample_swc` for full documentation.

The function :func:`sample_swc.get_sample_pts` can be used to read an SWC file
and sample its contents. The following function call will return a point cloud
the of points spaced along branches of the neuron. The types_keep flag is
optional; if one gives a list of integers, only points in the neuron with that
SWC structure identifier will be sampled. The standard structure identifiers
are 1-4, with 0 the key for "undefined"; indices greater than 5 are reserved
for custom types. By default (types_keep = None) all nodes are eligible to be sampled.


.. code-block:: python
		
		from CAJAL import sample_swc
		pt_cloud = sample_swc.get_sample_pts(
		                          file_name="a10_full_Chat-IRES-Cre-neo_Ai14-280699.05.02.01_570681325_m.swc",
					  infolder="/CAJAL/data/swc_files",
					  types_keep=None,
					  goal_num_pts = 50)[0]

CAJAL attempts to sample points in an evenly spaced way along the branches of
the neuron, or if there are multiple components in the SWC file, in an evenly
spaced way along the branches of each component. :func:`sample_swc.get_sample_pts` will return
"None" and raise a warning if there are more components in the graph than
points to sample, as it is not clear how to choose the points in an evenly
spaced way.


Point clouds can be written to a local directory as csv files, where each line
contains three floating-point coordinates. Here the format string
fmt="%.16f" means that we retain 16 values after the decimal place.

.. code-block:: python

		import numpy as np
		np.savetxt("mycloud.csv", pt_cloud, delimiter=",", fmt="%.16f")


We walk through an example. Suppose the user has a folder
:code:`/CAJAL/data/swc_files` containing a number of swc files. The function
:func:`sample_swc.compute_and_save_sample_pts_parallel` will go through each swc file in
the input directory and randomly sample a given number of points from each
neuron - in this case, 50 points from each. The 50 points are stored in the
given directory :code:`/CAJAL/data/sampled_pds/swc_sampled_50` as a
CSV file with 50 lines, where each line contains one point as
a triple of (x, y, z) coordinates. :code:`num_cores` is best set to the number
of cores on your machine. 

.. code-block:: python
		
		from CAJAL import sample_swc
		swc_infolder = "/CAJAL/data/swc_files"
		sampled_csv_folder = "/CAJAL/data/sampled_pts/swc_sampled_50"
		sample_swc.compute_and_save_sample_pts_parallel(
		    swc_infolder, sampled_csv_folder, goal_num_pts=50, num_cores=8)

Next, the user should compute the pairwise Euclidean distances between the
sampled points of each SWC file. The function
`:func:compute_intracell_distances_all` returns a list of distance matrices,
one for each \*.csv file in the given folder, linearized as arrays.

.. code-block:: python

		from CAJAL import run_gw
		dist_mat_list = run_gw.compute_intracell_distances_all(data_dir=sampled_csv_folder)

The Euclidean distance is not the only way to do this. The
user can also represent a neuron in terms of the geodesic distances between
points through the graph coded by the SWC file.
