Computing GW Distances
======================


Next, the user should compute the pairwise Euclidean distances between the
sampled points of each SWC file. The function `get_intracell_distances_all` returns a list of distance
matrices, one for each \*.csv file in the given folder, linearized as arrays
(Python multiprocessing arrays by default)

.. code-block:: python

		from CAJAL import run_gw
		dist_mat_list = run_gw.get_intracell_distances_all(data_dir=sampled_csv_folder)

The Euclidean distance is not the only way to do this. The
user can also represent a neuron in terms of the geodesic distances between
points through the graph coded by the SWC file.

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

CAJAL also contains a number of functions to read to or write from a file so
that one can save data between computing sessions.

We walk through an example.

The function :code:`obj_sample_parallel` will go through all \*.obj files in
the given directory and sample a point cloud with n_sample points from each
component of each \*.obj file, and save these point clouds as \*.csv files in
the given output directory. (It is not necessary to write the point clouds to a
file, they can be kept in memory as numpy arrays.)

.. code-block:: python

		from CAJAL.lib import sample_mesh
		infolder = "/CAJAL/data/obj_files"
		outfolder = "/CAJAL/data/sampled_pts/obj_sampled_50"
		sample_mesh.obj_sample_parallel(infolder, outfolder, n_sample=50, disconnect=True, num_cores=8)

The user can then compute a Euclidean intracell distance matrix for each
connected component, and compute the GW distances between all component
cells. This is identical to the process in :ref:`Neuronal Tracing Data`. Here,
we load the saved intracell distance data back into memory, compute the GW
distance matrix and write it to an output file. The flags "data_prefix" and
"data_suffix" are optional filters, only files beginning and ending with the given
string will be loaded into memory.

.. code-block:: python

		from CAJAL.lib import run_gw
		dist_mat_list = run_gw.get_intracell_distances_all(
		                     data_dir="/CAJAL/data/sampled_pts/obj_sampled_50",
				     data_prefix=None,
				     data_suffix="csv")
		run_gw.compute_and_save_GW_dist_mat(dist_mat_list,
		             file_prefix="obj_euclidean",
			     gw_results_dir="CAJAL/data/gw_results",
			     save_mat=False, num_cores=8)
		 
If the user wants to represent a cell by the matrix of geodesic distances
instead, then the "sample" functions (which ignore the topology) are
inappropriate. In this case CAJAL provides one batch-processing function which
goes through all \*.obj files in a given directory, separates them into
connected components, computes geodesic intracell distance matrices for each
component, and writes all these square matrices as files to a standard
output. (Bundling file I/O and math together in one function is less modular
but it makes it easier to parallelize.)

.. code-block:: python

		sample_mesh.compute_and_save_geodesic_from_obj_parallel(
		            infolder="/CAJAL/data/obj_files",
			    outfolder="CAJAL/data/sampled_pts/obj_geodesic_50",
			    n_sample=50,
			    method="heat",
			    connect=False,
			    num_cores=8)
