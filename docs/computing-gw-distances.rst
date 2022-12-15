Computing GW Distances
======================

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
		
This is identical to the process in :ref:`Neuronal Tracing Data`. Here,
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
