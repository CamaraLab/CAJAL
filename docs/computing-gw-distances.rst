Computing GW Distances
======================

Currently, CAJAL is equipped to deal with three kinds of input data files: neuronal tracing data (SWC files), 3D meshes (two-dimensional simplicial complexes) and 2D segmentation files (tiff files.)

Neuronal Tracing Data
---------------------

CAJAL supports neuronal tracing data in the SWC spec as specified here:
http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html

To analyze the similarity of two neurons, CAJAL first replaces each neuron by
an intracell distance matrix, where the rows and columns correspond to points
in the cell, and the entry at position (i, j) is the distance between points
x_i and x_j. Experience has shown diminishing returns in predictive power past
n = 50-100 points.

We offer some functions to help the user load and process SWC files, see :doc:`sample_swc`.

We walk through an example. Suppose the user has a folder
:code:`/CAJAL/data/swc_files` containing a number of swc files. The function
:code:`compute_and_save_sample_pts_parallel` will go through each swc file in
the input directory and randomly sample a given number of points from each
neuron - in this case, 50 points from each. The 50 points are stored in the
given directory :code:`/CAJAL/data/sampled_pds/swc_sampled_50` as a
comma-separated value file with 50 lines, where each line contains one point as
a triple of (x, y, z) coordinates. :code:`num_cores` is best set to the number
of cores on your machine. 

.. code-block:: python
		
		from CAJAL import sample_swc
		swc_infolder = "/CAJAL/data/swc_files"
		sampled_csv_folder = "/CAJAL/data/sampled_pts/swc_sampled_50"
		sample_swc.compute_and_save_sample_pts_parallel(
		    swc_infolder, sampled_csv_folder, goal_num_pts=50, num_cores=8)

Next, the user should compute the pairwise Euclidean distances between the
sampled points of each SWC file. The function `get_intracell_distances_all` returns a list of distance
matrices, one for each \*.csv file in the given folder, linearized as arrays
(Python multiprocessing arrays by default)

.. code-block:: python

		from CAJAL import run_gw
		dist_mat_list = run_gw.get_intracell_distances_all(data_dir=sampled_csv_folder)

Once the user prepares the list of intracell distance matrices, they can use
the function :code:`compute_and_save_GW_dist_mat` to
compute the Gromov-Wasserstein distance between all matrices in the given list
and write the result to a single file in a given output directory. This output
file is the linearization of the
Gromov-Wasserstein distance matrix (or rather the entries above the diagonal).
It is a text file with one column and n \*
(n-1) / 2 rows, where n is the number of swc files to be processed.

The argument "file_prefix" tells the function what the output file should be named;
if file_prefix = "abc" then the output file will be titled
"abc_gw_dist_mat.txt".

If the flag save_mat is set to true, for each pair of cells A, B the function
will also return the "coupling matrix" for the cells, which expresses the best
possible deformation of A into B, that is, the deformation minimizing the
worst-case distortion between any pairs of points. The Gromov-Wasserstein
distance between A and B is the distortion induced by this optimal coupling
matrix. These coupling matrices will be grouped in a folder, compressed and
saved to the given directory as "abc_gw_matching.npz"


.. code-block:: python

		file_prefix = "a10_full_euclidean"
		gw_results_dir= "/CAJAL/data/gw_results"
		run_gw.compute_and_save_GW_dist_mat(dist_mat_list,file_prefix,gw_results_dir,
		    save_mat=True, num_cores=12)

CAJAL requires the user to encode a cell's morphology as an intracell distance
matrix. However, the Euclidean distance is not the only way to do this. The
user can also represent a neuron in terms of the geodesic distances between
points through the graph coded by the SWC file.

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

The functions :code:`compute_and_save_sample_pts_parallel` and
:code:`get_intracell_distances_all` are only appropriate when the user wants to
represent a cell by its Euclidean distance matrix. To convert a folder of
SWC files to a folder of intracell geodesic distance matrices, the user can run

.. code-block:: python

		infolder = "/CAJAL/data/swc_files"
		outfolder = "/CAJAL/data/sampled_pts/swc_geodesic_50"
		sample_swc.compute_and_save_geodesic_parallel(infolder, outfolder,
                                  goal_num_pts=50, num_cores=8)

The user can then read these files back into memory with the function
:code:`load_intracell_distances`:
		  
.. code-block:: python

		dist_mat_list = run_gw.load_intracell_distances(
		   distances_dir="/CAJAL/data/sampled_pts/swc_geodesic_50",
		   data_prefix="a10_full"

In this example, :code:`load_intracell_distances` takes a string parameter
:code:`data_prefix`. If :code:`data_prefix` is given, the function will only read
files whose name begins with that string.

The GW distance is calculated using the same function whether the distance
matrices represent the Euclidean or geodesic metric.

.. code-block:: python

		run_gw.compute_and_save_GW_dist_mat(
		    dist_mat_list,
		    file_prefix="a10_full_geodesic",
		    "/CAJAL/data/gw_results",
		    save_mat=True,
		    num_cores=8
		    )



		
3D meshes
---------




