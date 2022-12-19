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
CAJAL's convention is to save them as comma-separated value files where each each point lies
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

The function :func:`run_gw.compute_intracell_distances_one` reads a point cloud
\*.csv file into memory and returns an intracell distance matrix.

.. autofunction:: run_gw.compute_intracell_distances_one

The function :func:`run_gw.compute_intracell_distances_all` is the batch version,
operating on a directory of point cloud \*.csv files and returning a list of
intracell distance matrices.

.. autofunction:: run_gw.compute_intracell_distances_all


Neuronal Tracing Data
---------------------

CAJAL supports neuronal tracing data in the SWC spec as specified here:
http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html

We offer some functions to help the user load and process SWC files, see
:doc:`sample_swc` for full documentation.

Sampling from SWC Files
^^^^^^^^^^^^^^^^^^^^^^^
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

One can then convert this to a Euclidean distance matrix with :func:`scipy.spatial.distance.pdist` or write it to a \*.csv file to be read later, as described in :ref:`Basic Data and File Formats`.

We walk through an example. Suppose the user has a folder
:code:`/CAJAL/data/swc_files` containing a number of swc files. The function
:func:`sample_swc.compute_and_save_sample_pts_parallel` will go through each swc file in
the input directory and randomly sample a given number of points from each
neuron - in this case, 50 points from each. :code:`num_cores` is best set to the number
of cores on your machine. 

.. code-block:: python
		
		from CAJAL import sample_swc
		swc_infolder = "/CAJAL/data/swc_files"
		sampled_csv_folder = "/CAJAL/data/sampled_pts/swc_sampled_50"
		sample_swc.compute_and_save_sample_pts_parallel(
		    swc_infolder, sampled_csv_folder, goal_num_pts=50, num_cores=8)

Next, the user should compute the pairwise Euclidean distances between the
sampled points of each SWC file. The function
:func:`compute_intracell_distances_all` returns a list of distance matrices,
one for each \*.csv file in the given folder, linearized as arrays.

.. code-block:: python

		from CAJAL import run_gw
		dist_mat_list = run_gw.compute_intracell_distances_all(data_dir=sampled_csv_folder)

Computing geodesic intracell distance matrices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
		
The user can also represent a neuron in terms of the geodesic distances between
points through the graph coded by the SWC file.
For full documentation, see `Geodesic Intracell Distances from SWC files`.

To load an \*.swc file into memory and compute its intracell geodesic distance matrix, use :func:`sample_swc.get_geodesic`.

The functions :code:`compute_and_save_sample_pts_parallel` and
:code:`get_intracell_distances_all` are only appropriate when the user wants to
represent a cell by its Euclidean distance matrix, as they forget the
topology. To convert a folder of SWC files to a folder of intracell geodesic
distance matrices, the user can run

.. code-block:: python

		infolder = "/CAJAL/data/swc_files"
		outfolder = "/CAJAL/data/sampled_pts/swc_geodesic_50"
		sample_swc.compute_and_save_geodesic_parallel(infolder, outfolder,
                                  goal_num_pts=50, num_cores=8)

3D meshes
---------

CAJAL supports Wavefront \*.obj 3D mesh files. The lines of a mesh file are
expected to be either

- a comment, marked with a "#"
- a vertex, written as `v float1 float2 float3`
- a face, written as `f linenum1 linenum2 linenum3`

Examples of \*.obj files compatible with CAJAL can be found in the CAJAL Github
repository in CAJAL/data/obj_files.

It is expected that a \*.obj file may contain several distinct connected
components. By default, these will be separated into individual cells.

However, the user may find themselves in a situation where each \*.obj file is
supposed to represent a single cell, but due to some measurement error, the
mesh given in the \*.obj file has multiple connected components - think of a
scan of a neuron where there are missing segments in a dendrite. In this case
CAJAL provides functionality to create a new mesh where all components will be
joined together by new faces so that one can sensibly compute a geodesic
distance between points in the mesh. (If the user wants to compute the
Euclidean distance between points, such repairs are unnecessary, as Euclidean
distance is insensitive to connectivity.)

Sampling from meshes
^^^^^^^^^^^^^^^^^^^^

The function :func:`sample_mesh.obj_sample_parallel` will go through all \*.obj files in
the given directory and sample a point cloud with n_sample points from each
component of each \*.obj file, and save these point clouds as \*.csv files in
the given output directory. (It is not necessary to write the point clouds to a
file, they can be kept in memory as numpy arrays.)

.. code-block:: python

		from CAJAL.lib import sample_mesh
		infolder = "/CAJAL/data/obj_files"
		outfolder = "/CAJAL/data/sampled_pts/obj_sampled_50"
		sample_mesh.obj_sample_parallel(infolder, outfolder, n_sample=50,
		disconnect=True, num_cores=8)

The user can then compute a Euclidean intracell distance matrix for each
connected component, and compute the GW distances between all component
cells.

Geodesic distances from meshes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CAJAL provides one batch-processing function which
goes through all \*.obj files in a given directory, separates them into
connected components, computes geodesic intracell distance matrices for each
component, and writes all these square matrices as files to a standard
output. (Bundling file I/O and math together in one function is less modular
but it makes it easier to parallelize and not fill the memory)

.. code-block:: python

		sample_mesh.compute_and_save_geodesic_from_obj_parallel(
		            infolder="/CAJAL/data/obj_files",
			    outfolder="CAJAL/data/sampled_pts/obj_geodesic_50",
			    n_sample=50,
			    method="heat",
			    connect=False,
			    num_cores=8)

Segmentation files 
-------------------

`Image segmentation <https://en.wikipedia.org/wiki/Image_segmentation>`_ is the
process of separating an image into distinct components to simplify
representations of objects. `Morphological segmentation
<https://www.sciencedirect.com/science/article/abs/pii/104732039090014M>`_
refers to image segmentation based on morphology.

There are existing tools available to the user to segment an image, see for
example the `ImageJ/Fiji Morphological Segmentation plugin
<https://www.youtube.com/watch?v=gF4nhq7I2Eo>`_. (If you are unfamiliar with
image segmentation, the linked YouTube video is only 6 minutes long and is a
helpful introduction.) CAJAL provides tools to sample from the cell boundaries
of segmented image files, such as the image provided at the
`5:20 mark of the above video <https://youtu.be/gF4nhq7I2Eo?t=320>`_.

Suppose that the user has a collection of \*.tiff files such as the following
(from CAJAL/data/tiff_images/epd210cmd1l3_1.tif)

.. image:: images/epd210cmd1l3_1.png

The user can use :func:`tifffile.imread` or :func:`cv.imread` to load \*.tiff
files into memory. CAJAL expects that an image is loaded as a Numpy integer array of
shape (n, m), where n x m is the dimension of the picture in pixels and the
value in image[n,m] codes the color of the image.

.. code-block:: python

		img=tifffile.imread(CAJAL/data/tiff_images/epd210cmd1l3_1.tif)
		im_array2=cv.imread(CAJAL/data/tiff_images/epd210cmd1l3_1.tif)

The OpenCV package provides some basic functionality to clean image data and
perform segmentation, as mentioned earlier you can also use ImageJ for this
task. We give an example to show how to segment `img`, an integer Numpy array
of shape (n,m).

.. code-block:: python

                # Collapse the grayscale image to black and white.
		# Everything with value below 100 gets mapped to white.
		# Everything above 100 gets mapped to black.
		_, thresh = cv.threshold(img,100,255,cv.THRESH_BINARY)
		# See this tutorial for explanation of cv.morphologyEx 
                # and the MORPH_OPEN and MORPH_CLOSED flags.
		# https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
		kernel = np.ones((5,5),np.uint8)
                closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
		closethenopen = cv.morphologyEx(closing, cv.MORPH_OPEN,kernel)
		# closethenopen is black-and-white, like thresh, but with some
		# noise removed.

		from skimage import measure
		# labeled_img is a numpy array of the same shape as closethenopen
                # but instead of being black and white, each connected region
		# of the image shares a unique common color.		
		labeled_img = measure.label(closethenopen)

		# The image is still somewhat noisy, with a few specks in it.
		# We despeckle it naively by removing all connected regions
		# with fewer than 1000 pixels by grouping these into the
		# background region, labelled with 0.
		labels = np.unique(labeled_img, return_counts=True)
		labels = (labels[0][1:],labels[1][1:])
		#remove specks
		remove = np.isin(labeled_img, labels[0][labels[1]<1000])
		img_keep = labeled_img.astype(np.uint8)
		img_keep[remove] = 0

		# To view the image from an interactive environment,
		# i.e. Jupyter notebook, you can use matplotlib.
		import matplotlib.pyplot as plt
		fig, ax = plt.subplots()
		ax.imshow(simplify_img_keep)
		fig.set_size_inches(30, 30)
		plt.show()

		# Or write to a file and view with standard image utilities.
		tifffile.imwrite('/home/jovyan/CAJAL/CAJAL/data/cleaned_file.tif',
		img_keep, photometric='minisblack')

After our cleaning, we get this:

.. image:: images/cleanedfile.png


		
		
