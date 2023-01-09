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
of deformations *CAJAL* regards as relevant when comparing the shape of two
cells.  Using Euclidean distance to meassure intracellular distances leads to
morphological distances that are insensitive to translations, rotations, or
mirroring of a cell. However, bending or flexing a cell will change the
morphological distance between that cell and other cells.  On the other hand,
using geodesic intracellular distances leads to morphological distances that
are insensitive to translations, rotations, mirroring, bending, and flexing of
the cells.

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

The function `sample_swc.compute_and_save_intracell_all` operates on
directories of \*.swc files and populates a second directory with intracell
distance matrices, one for each cell in the source directory.

.. code-block:: python

		failed_cells = sample_swc.compute_and_save_intracell_all(
                    infolder = "/home/jovyan/CAJAL/CAJAL/data/swc_files",
		    metric = "geodesic",
		    outfolder = "/home/jovyan/CAJAL/CAJAL/data/intracell_dist_mats/swc_geodesic_50",
		    types_keep = None,
		    sample_pts = 50,
		    num_cores = 8,
		    keep_disconnect =False
		    )

Here, `infolder` is a directory full of \*.swc files and `outfolder` is an
empty directory where the intracell distance matrices will be written. The
argument `metric` can be either "euclidean" or "geodesic". `types_keep` is an
optional argument, one can supply a list of integers corresponding to node
types to sample from as in the SWC spec. (If `types_keep` is `None`, all types
will be sampled from.)  Thus, if `types_keep = [0,1,2,3,4]`, only the standard
parts identified in the SWC spec will be sampled (undefined, soma, axon, basal
dendrite, apical dendrite), and custom parts will be ignored. The soma will
always be included as a type to sample from, regardless of whether the user
includes it in this list.

`sample_pts` is the number of points from each cell that will be sampled. We
recommend between 50-100. Past 100 points, the increase in resolution is not
associated with a significant increase in statistical predictive power.

`num_cores` is the number of processes that will be launched in parallel, we recommend setting this to the number of cores on your machine.

If `keep_disconnect` is False, only branches of the neuron connected to the
soma will be sampled.  Otherwise if `keep_disconnected` is True, the function
will sample from all branches, including free-floating ones.

The function returns a list `failed_cells` of names of cells for which sampling
was unsuccessful. If the sampling is successful, the results are silently
written to a file.

		    
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

CAJAL provides one batch-processing function which
goes through all \*.obj files in a given directory, separates them into
connected components, computes intracell distance matrices for each
component, and writes all these square matrices as files to a standard
output. (Bundling file I/O and math together in one function is less modular
but it makes it easier to parallelize and not fill the memory)

.. code-block:: python

		failed_samples = sample_mesh.compute_and_save_intracell_all(
		            infolder="/home/jovyan/CAJAL/data/obj_files",
			    outfolder="/home/jovyan/CAJAL/data/sampled_pts/obj_geodesic_50",
			    n_sample=50,
			    metric = "segment",
			    segment = True
			    method="heat",
			    connect=False,
			    num_cores=8)

The arguments `infolder, outfolder, n_sample, metric` are as in :ref:`Neuronal Tracing Data`, except that `infolder` is a folder containing \*.obj files rather than \*.swc files.

If the Boolean flag `segment` is True, the function will break down each \*.obj
file into its connected components and treat them as individual, isolated
cells.  If `segment` is False, the function will treat each \*.obj file as a
single cell.  If the user chooses the "geodesic" metric and the contents of an
\*.obj file are not connected, CAJAL will automatically attempt to "repair" the
cell by modifying it to adjoin new paths between connected components, so that
a geodesic distance between points can be defined.

.. warning::

   Modifying the data by adjoining new triangles to the mesh is changing its
   topology. The user should keep this in mind while interpreting the data.
   The functionality of "repairing" the cell is premised on the
   assumption that the \*.obj file represents one single geometric object and
   that it fails to be connected for trivial reasons, e. g. a scan of a neuron
   that has missing segments along the dendrites due to measurement error.  If
   an \*.obj file genuinely contains multiple distinct components then the
   geodesic distances resulting from this process will not be meaningful.

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

Cells which meet the image boundary are discarded, as we currently do not have
a reasonable theoretical approach for analyzing partial cell boundaries.

CAJAL samples from \*.tiff / \*.tif files via the function
:func:`sample_seg.compute_and_save_intracell_all` which takes as an argument an
input directory full of (cleaned!) \*.tiff/\*.tif files and an output
directory. For each \*.tiff file in the input directory,
:func:`sample_seg.compute_and_save_intracell_all` breaks the image down into
its separate cells, samples a given number of points between each one, and
writes the resulting resulting intracell distance matrix for each cell to its
own text file in the output directory.

.. code-block:: python

		infolder ="/home/jovyan/CAJAL/CAJAL/data/tiff_images_cleaned/"
		outfolder="/home/jovyan/CAJAL/CAJAL/data/tiff_sampled_50/"
		sample_seg.compute_and_save_intracell_all(
		       infolder,
		       outfolder,
		       n_sample = 50,
		       background = 0,
		       discard_cells_with_holes = False,
		       only_longest = False,
		       num_cores = 8)

`infolder`, `outfolder`, and `n_sample` are as in the previous two
sections. `background` is the index for the background color; it is zero by
default.  If the flag `discard_cells_with_holes` is set to True, the function
will ignore any cells which have multiple boundaries, which helps to filter out
clusters of overlapping cells. The flag `only_longest` is only relevant if
`discard_cells_with_holes` is False. In this case if `only_longest` is True,
then the function only samples from the longest boundary of the cell, instead
of across all boundaries.






