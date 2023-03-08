Use of the Graph Laplacian to identify features related to cell morphology
--------------------------------------------------------------------------

Suppose that we have a set of cells, :math:`G`, and a numerical feature
:math:`f : G\to \mathbb{R}`. Our question is whether the value of :math:`f`
is related to cell morphology. There are multiple ways to phrase the question
informally:

- If :math:`x` and :math:`y` are two cells in `G` with similar morphology, then
  are :math:`f(x)` and :math:`f(y)` closer on average then they would be if
  :math:`x` and :math:`y` were chosen randomly?
- If :math:`x` is a fixed cell in :math:`G`, and :math:`y` is a randomly chosen
  cell in :math:`G`, does :math:`f(y)` tend to be closer to :math:`f(x)` if
  :math:`y` is more morphologically similar to :math:`x`?
- Does cell morphology have explanatory power with regards to the value of :math:`f`?

We implement and offer a statistical test which represents one formalization of
these ideas, the *graph Laplacian*, building on work by He, Cai and Niyogi
(`Laplacian Score for Feature Selection
<https://proceedings.neurips.cc/paper/2005/hash/b5b03f06271f8917685d14cea7c6c50a-Abstract.html>`_);
see also `Multi-modal analysis and integration of single-cell morphological data <https://www.biorxiv.org/content/10.1101/2022.05.19.492525v3.full>`_.

We first use the Gromov-Wasserstein distance to equip the set of cells with an
(undirected) graph structure. Assume that we have already computed the
GW-distance between any two cells in :math:`G`. Choose also a positive real
number :math:`varepsilon` which is much less than the maximum distance between
two cells; for example, one could take the median observed difference. Now from
this point forward we will view :math:`G` as an undirected graph, where two
distinct nodes :math:`x,y` are connected if and only if
:math:`d_{GW}(x,y)\lt\varepsilon`.

 The graph Laplacian of :math:`f` with respect to the graph :math:`G` is defined by

.. math::

   C_G(f) = \frac{\sum_{(i,j)\in E(G)} (f(i) - f(j))^2}{\operatorname{Var}_G(f)}


where :math:`E(G)` is the set of edges in the graph `G`, :math:`i,j` range over
nodes of :math:`G`, `n(i)` is the number of neighbors of :math:`i` in
:math:`G`, and :math:`\operatorname{Var}_G(f)` is a weighted
variance of `f` where the weight of node :math:`i` is proportional to :math:`n(i)`.

The graph Laplacian takes values between 0 and 2. When the Laplacian is near
zero we interpret it as showing that the values of :math:`f(i)` and
:math:`f(j)` are linearly correlated when :math:`i,j` are connected by an edge
in the graph. If the Laplacian is close to 1, they are uncorrelated.

In cases where it is unreasonable to expect a strong linear correlation, (for
example, :math:`f` is not continuous or not modelled by any reasonable
distribution) the graph Laplacian cannot be interpreted directly. We provide a
permutation test which compares :math:`C_G(f)` to :math:`C_G(f\circ\pi)` for
many randomly chosen permutations :math:`\pi : G\to G` of the set of nodes of
the graph. If :math:`C_G(f) < C_G(f\circ\pi)` for all but a small fraction
:math:`\alpha` of the permutations, we conclude that it is unlikely that
:math:`C_G(f)` could have arisen by chance by choosing a random function with
the same values.

We also provide functionality to let the user regress out on covariates. If
:math:`g_1,\dots,g_n` are features on :math:`G`, and the user wants to know
whether :math:`C_G(f)` is below what would be expected from
:math:`C_G(g_1),\dots, C_G(g_n)`, they can fit a multilinear regression to
predict :math:`C_G(f\circ\pi)` as a sum :math:`\sum_i \beta_i
C_G(g_i\circ\pi)+\beta_0` plus a residual error term :math:`\varepsilon_i`. If
the true residual :math:`\varepsilon =C_G(f) - \widehat{C_G(f)}` is at the
lower tail end of the residuals, we conclude that :math:`f` respects the
morphology graph structure in excess of what would be predicted given with
:math:`C_G(g_1),\dots, C_G(g_n)`.

.. autofunction:: cajal.graph_laplacian.graph_laplacians

Example - C. Elegans Dataset
============================

We will illustrate how to use graph Laplacian to identify features in a C. elegans
neuron SWC dataset which are correlated with cell morphology.

First, download and unzip `this folder
<https://www.dropbox.com/s/uwcgluux2r0dwvb/c_elegans_gw_dists.csv?dl=0>`_
containing 799 \*.swc files, which are neurons from C. elegans sampled at
different days throughout their development.

We then compute the Gromov-Wasserstein distances between each pair of cells in
this folder. It is hopefully clear at this point from the other examples how to
sample points from each cell and compute the pairwise distances between
neurons. At a resolution of 100 points per cell this takes about 1 hour 45
minutes on a machine with 20 cores. Let us name the file
`c_elegans_gw_dists.csv`.

Download the precomputed Gromov-Wasserstein distances `here
<https://www.dropbox.com/s/uwcgluux2r0dwvb/c_elegans_gw_dists.csv?dl=0>`__.
Lastly, download the neuron features we want to analyze `here
<https://www.dropbox.com/s/jli4hqbc9vuyd4f/c_elegans_features.csv?dl=0>`__. We
have eleven features we want to measure. Each feature is binary and corresponds
to the expression of a certain gene.

We will use Pandas for this analysis.

.. code-block:: python

		import os
		from cajal.utilities import read_gw, list_sort_files,dist_mat_of_dict
		import pandas as pd

		project_dir=os.getcwd()
		gw_csv_loc=project_dir+"/c_elegans_gw_dists.csv"
		features_file = project_dir+"/c_elegans_features.csv"
		cell_names, gw_dist_dict = read_gw(gw_csv_loc,header=True)
		feature_matrix = pd.read_csv(features_file)
		# Clean the features table up a bit for analysis.
		feature_matrix.index = feature_matrix['cell_name']
		feature_matrix=feature_matrix.drop('cell_name',axis=1)

The neuron samples are organized by the age of the worm on the date of the sample. (No samples were collected on day 4.)

.. code-block:: python

		cell_names_day1 = [cell_name for cell_name in cell_names if "day1" in cell_name]
		cell_names_day2 = [cell_name for cell_name in cell_names if "day2" in cell_name]
		cell_names_day3 = [cell_name for cell_name in cell_names if "day3" in cell_name]
		cell_names_day5 = [cell_name for cell_name in cell_names if "day5" in cell_name]
		# print(len(cell_names_day1)+len(cell_names_day2)+len(cell_names_day3)+len(cell_names_day5)) # = 799
		# print(len(cell_names)) # = 799 
		df_day1 = feature_matrix.loc[cell_names_day1]
		df_day2 = feature_matrix.loc[cell_names_day2]
		df_day3 = feature_matrix.loc[cell_names_day3]
		df_day5 = feature_matrix.loc[cell_names_day5]

Before we can apply our analysis tool we have to remove any constant features, otherwise there is
a divide-by-zero error in the computation of the graph Laplacian.

.. code-block:: python

		df_day1.apply(sum, axis=0)

		>> nrx-1     15
		mir-1      5
		unc-49     0
		nlg-1      5
		unc-25    18
		unc-97    14
		lim-6      0
		lat-2      0
		ptp-3      0
		sup-17     0
		pkd-2      0
		dtype: int64

As you can see, many genes were not observed at all on certain days. Let us
restrict to the columns for which there is nonzero data.

.. code-block:: python

		day1_cols=['nrx-1','mir-1','nlg-1','unc-25','unc-97']
		df_day1= df_day1[day1_cols]
		day2_cols=['nrx-1','unc-97']
		df_day2= df_day2[day2_cols]
		# Day 3 doesn't need to be cleaned, as every feature is nonconstant on day 3.
		day5_cols=['nrx-1','nlg-1','unc-97']
		df_day5= df_day5[day5_cols]

		feature_arr_day1=df_day1.to_numpy(dtype=np.float_)
		feature_arr_day2=df_day2.to_numpy(dtype=np.float_)
		feature_arr_day3=df_day3.to_numpy(dtype=np.float_)
		feature_arr_day5=df_day5.to_numpy(dtype=np.float_)

		import statistics
		gw_dists_day1 = dist_mat_of_dict(cell_names_day1,gw_dist_dict)
		median1=statistics.median(gw_dists_day1)
		gw_dists_day2 = dist_mat_of_dict(cell_names_day2,gw_dist_dict)
		median2=statistics.median(gw_dists_day2)
		gw_dists_day3 = dist_mat_of_dict(cell_names_day3,gw_dist_dict)
		median3=statistics.median(gw_dists_day3)
		gw_dists_day5 = dist_mat_of_dict(cell_names_day5,gw_dist_dict)
		median5=statistics.median(gw_dists_day5)

		results_df_day1 = pd.DataFrame(graph_laplacians(feature_arr_day1,gw_dists_day1,median1, 2000, None, False))
		results_df_day2 = pd.DataFrame(graph_laplacians(feature_arr_day2,gw_dists_day2,median2, 2000, None, False))
		results_df_day3 = pd.DataFrame(graph_laplacians(feature_arr_day3,gw_dists_day3,median3, 2000, None, False))
		results_df_day5 = pd.DataFrame(graph_laplacians(feature_arr_day5,gw_dists_day5,median5, 2000, None, False))

		print("Day 1:")
		print(results_df_day1)
		print("Day 2:")
		print(results_df_day2)
		print("Day 3:")
		print(results_df_day3)
		print("Day 5:")
		print(results_df_day5)		

Output:

.. code-block::

   Day 1:
   feature_laplacians  laplacian_p_values  laplacian_q_values
   0            0.993843            0.534233            0.534233
   1            0.990893            0.421289            0.526612
   2            0.983587            0.207896            0.346493
   3            0.967470            0.028986            0.144928
   4            0.981699            0.161419            0.403548
   Day 2:
   feature_laplacians  laplacian_p_values  laplacian_q_values
   0            0.903342            0.111444            0.111444
   1            0.843193            0.023988            0.047976
   Day 3:
   feature_laplacians  laplacian_p_values  laplacian_q_values
   0             0.980892            0.000500            0.003665
   1             1.000079            0.792104            0.792104
   2             0.997310            0.223888            0.410461
   3             0.998686            0.482259            0.589428
   4             0.998223            0.381309            0.524300
   5             0.980563            0.000500            0.003665
   6             0.999509            0.686157            0.754773
   7             0.989684            0.001499            0.005497
   8             0.993579            0.023988            0.052774
   9             0.989100            0.002499            0.006872
   10            0.997994            0.349825            0.549725
   Day 5:
   feature_laplacians  laplacian_p_values  laplacian_q_values
   0            0.978943            0.113443            0.113443
   1            0.934330            0.001499            0.002249
   2            0.829818            0.000500            0.001499
