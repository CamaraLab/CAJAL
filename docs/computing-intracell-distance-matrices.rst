Computing Intracellular Distance Matrices
=========================================

*CAJAL* represents a cell as a finite set of points uniformly sampled from its outline, together with a notion of distance
between each pair of points. Internally this data is represented as a matrix
(the "intracellular distance matrix") where the rows and columns correspond to
points in the cell, and the entry at position (i, j) corresponds to the distance between
points x_i and x_j. In general, we find that 50 to 100 sampled points per cell is enough for most applications. In order to compute the Gromov-Wasserstein
distance between two cells, the user must first convert their cell morphology
data into intracellular distance matrices. This section discusses the functionality *CAJAL* provides
for this purpose. Currently, CAJAL is equipped to deal with three kinds of input data files:
neuronal tracing data (SWC files), 3D meshes (OBJ files), and 2D cell segmentation files (TIFF files).

Euclidean vs. geodesic distances
--------------------------------

*CAJAL* supports two types of intracellular distances:

1. the ordinary, straight-line Euclidean
   distance through space ("as the crow flies")
2. the geodesic distance, the length of the shortest path
   through the surface of the cell.

The choice between using Euclidean or geodesic distance will affect what kinds
of deformations *CAJAL* regards as relevant when comparing the shape of two cells. Using Euclidean distance to meassure intracellular distances 
leads to morphological distances that are insensitive to translations, rotations, or mirroring of a cell. However,
bending or flexing a cell will change the morphological distance between that cell and other cells. On the other hand, using geodesic
intracellular distances leads to morphological distances that are insensitive to translations, rotations, mirroring, bending, and flexing of the cells. 

To illustrate the distinction, suppose we have two pieces of string, A
and B. Both A and B are twelve inches long. A is laid out in a straight line,
whereas B is tightly coiled. If A and B are represented by Euclidean distance
matrices, then the Gromov-Wasserstein distance between them will be nontrivial,
because one must bend B to straighten it out into a line segment. However, if
they are represented by their geodesic distance matrices, then the
Gromov-Wasserstein distance will be zero.  One can deform A into B
without any stretching or elongating, as they are the same length. 

Neuronal Tracing Data
---------------------

CAJAL supports neuronal tracing data in the SWC spec as specified `here
<http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html>`_.

Sampling points from a SWC File
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The function :func:`sample_swc.get_sample_pts` can be used to read an SWC file
and sample its contents. 

For example, the following function call will return a point cloud
consisting of points equally spaced along the neuronal reconstruction in the file `a10_full_Chat-IRES-Cre-neo_Ai14-280699.05.02.01_570681325_m.swc <https://github.com/CamaraLab/CAJAL/blob/main/CAJAL/data/swc_files/a10_full_Chat-IRES-Cre-neo_Ai14-280699.05.02.01_570681325_m.swc>`_. 

.. code-block:: python
		
		from CAJAL import sample_swc
		pt_cloud = sample_swc.get_sample_pts(
		                          file_name="a10_full_Chat-IRES-Cre-neo_Ai14-280699.05.02.01_570681325_m.swc",
					  infolder="/CAJAL/data/swc_files",
					  types_keep=None,
					  goal_num_pts = 50)[0]

The ``types_keep`` flag is
optional and can be used to specify a list of node types (as specified in the SWC format) so that only points in the neuron with that
SWC structure identifier will be sampled. By default (types_keep = None) all nodes are eligible to be sampled.

*CAJAL* samples points in an evenly spaced way along the branches of
the neuron, or if there are multiple components in the SWC file, in an evenly
spaced way along the branches of each component. :func:`sample_swc.get_sample_pts` will return
"None" and raise a warning if there are more components in the SWC file than
points to sample.

One can then convert this to a Euclidean distance matrix with :func:`scipy.spatial.distance.pdist`:

.. code-block:: python
		
		from scipy.spatial.distance import pdist
		dist_mat = pdist(pt_cloud)

Alternatively, we can write it to a \*.csv file to be read later:

.. code-block:: python

		import numpy as np
		np.savetxt("mycloud.csv", pt_cloud, delimiter=",", fmt="%.16f")

The function :func:`run_gw.compute_intracell_distances_one` can be used to read a point cloud stored as a
\*.csv file into memory and compute the intracellular distance matrix.

Sampling points from multiple SWC Files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
*CAJAL* also provides a wrapper around the above functions to process multiple SWC files at a time. We walk through an example using the example SWC files provided with *CAJAL*. The function
:func:`sample_swc.compute_intracell_parallel` goes through each SWC file in the input directory, randomly samples a given number of points from each neuron, and computes an intracell distance matrix based on the sample.

For example, if we want to sample 50 points from each neuron in the folder :code:`/CAJAL/data/swc_files` using 8 cores (:code:`num_cores` is best set to the number
of cores on your machine), and compute the Euclidean intracell distance matrices we would use the command:

.. code-block:: python
		
		from CAJAL import sample_swc
		swc_infolder = "/CAJAL/data/swc_files"
		sample_swc.compute_intracell_parallel(
		    swc_infolder, "euclidean", types_keep=None,sample_pts=50, num_cores=8)

.. code-block:: python
		
		from CAJAL import sample_swc
		swc_infolder = "/CAJAL/data/swc_files"
		sampled_csv_folder = "/CAJAL/data/sampled_pts/swc_sampled_50"
		sample_swc.compute_and_save_intracell_parallel(
		    swc_infolder, "euclidean", sampled_csv_folder, sample_pts=50, num_cores=8)


		    
The second argument can be either "euclidean" or "geodesic" as preferred. The function returns a list of the file names in the directory for which an intracell distance matrix could not be produced.
		    
3D meshes
---------

CAJAL supports Wavefront \*.obj 3D mesh files. The lines of a mesh file are
expected to be either

- a comment, marked with a "#"
- a vertex, written as `v float1 float2 float3`
- a face, written as `f linenum1 linenum2 linenum3`

Examples of \*.obj files compatible with *CAJAL* can be found in the *CAJAL* Github
repository in ``CAJAL/data/obj_files``.

It is expected that a \*.obj file may contain several distinct connected
components. By default, these will be separated into individual cells.

However, the user may find themselves in a situation where each \*.obj file is
supposed to represent a single cell, but due to some measurement error, the
mesh given in the \*.obj file has multiple connected components - think of a
scan of a neuron where there are missing segments. In this case
*CAJAL* provides functionality to create a new mesh where all components will be
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



Overview of image segmentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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


.. warning::

   CAJAL is not a tool for image segmentation. The user is expected to segment
   and clean their own images.


However, we provide a
brief sample script here to show how a user might prepare data for use with
CAJAL.

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

This image is representative of the kind of image data CAJAL is meant to
process: a 2D array of integers, where each cell, and the background, are
represented by a connected block of integers with the same value. Two distinct
cells should have different values. Each cell should have a different labelling
value than the background. Be warned that this is only a toy example - for
example, in this image there are multiple overlapping cells that have been
grouped into a single continuous "cell" block. Such overlapping cells should be
discarded before analysis with CAJAL.

Sampling from segmented images (overview)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this section, a "segmented image" refers to a numpy integer array Arr of shape
(n, m) where Arr[i,j] represents the (i,j) pixel in an image. We say that a
pixel (i,j) is labeled with an integer k if Arr[(i,j)] = k.  We say that a cell is
labeled with the integer k if all pixels in that cell are labeled with the
integer k.

Each cell in a segmented image should be labeled with some integer. Two
distinct cells should be labeled with different integers. All background pixels
should be labelled with the same integer, which is different from the label of
any cell.

Given a numpy integer array :code:`imarray` of shape (n,m), we can use the
:func:`sample_seg.cell_boundaries` function to get a list of cell boundary
sample points for each cell.

.. code-block:: python

		bdaries = cell_boundaries(imarray, n_sample = 50, background= 0)

Cells which meet the image boundary are discarded, as we currently do not have
a reasonable theoretical approach for analyzing partial cell boundaries.

This sample script shows how to batch sample from all \*.tiff files in a given
directory, sample their points, and write the output to \*.csv files.

.. code-block:: python

		infolder ="/home/patn/CAJAL/CAJAL/data/tiff_images_cleaned/"
		outfolder="/home/patn/CAJAL/CAJAL/data/sampled_pts/tiff_sampled_50/"
		file_names = os.listdir("/home/patn/CAJAL/CAJAL/data/tiff_images_cleaned/")
		for image_file_name in file_names:
		    imarray = tifffile.imread(os.path.join(infolder,image_file_name))
		    cell_bdary_sample_list = sample_seg.cell_boundaries(imarray, 50)
		    i=0
		    for cell_bdary in cell_bdary_sample_list:
		        output_name = image_file_name.replace(".tiff","").replace(".tif","") + "_" + str(i)+ ".csv"
		        output_name=os.path.join(outfolder, output_name)
		        np.savetxt(output_name,cell_bdary, delimiter=",")		
		        i+=1
