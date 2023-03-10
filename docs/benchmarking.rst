Examples
========

Classifying the CRE-driver line of neurons
------------------------------------------

Here we perform some basic data analysis on a set of neuron morphological reconstructions which are provided by the `Allen Brain Atlas <https://celltypes.brain-map.org/>`_.

The `brain cell database <https://celltypes.brain-map.org/data>`_ contains 509 mouse neuron specimens that carry either a full or dendrite-only morphology reconstruction.


Downloading the SWC Files
^^^^^^^^^^^^^^^^^^^^^^^^^
The \*.SWC files we use in our experiments are available in a compressed \*.tar.gz file here : https://www.dropbox.com/s/aq0ovetjtqihf4f/allen_brain_atlas_509_SWCs_mouse_full_or_dendrite_only.tar.gz?dl=0.

The user should download and extract the SWC files from this link, ignore this
subsection, and skip ahead to the next subsection unless they are interested in
downloading a different set of neurons from the Allen Brain Atlas.

If the user wants to download a different set of SWC files, for example the
human samples, we provide the script we used to download these SWC files so the
user can modify it at their convenience.

Save the following metadata table to a convenient location: 
https://celltypes.brain-map.org/cell_types_specimen_details.csv

Now we will download the data (about 141 MB) to some convenient working directory. (The
Allen Brain Atlas API is available `here <http://help.brain-map.org/display/celltypes/API#API-download_swc>`_. They also
provide a Python package `allensdk` (`project
homepage <https://allensdk.readthedocs.io/en/latest/index.html>`_, `PyPI
<https://pypi.org/project/allensdk/>`_), however it is not yet updated to
support Python 3.10.)

.. code-block:: python

  		import os
		import pandas as pd
		import urllib.request
                import shutil
		import json

		# https://celltypes.brain-map.org/cell_types_specimen_details.csv
		cell_types_specimen_details_loc = "/home/jovyan/cell_types_specimen_details.csv"
		metadata = pd.read_csv(cell_types_specimen_details_loc)
		# We consider only the mouse neurons, either full reconstructions or dendrite_only reconstructions. 
		metadata_filtered = metadata[(metadata["donor__species"]=="Mus musculus")
                             & ((metadata["nr__reconstruction_type"]=="full") |
                               (metadata["nr__reconstruction_type"]=="dendrite-only"))]
			     
		swc_dir = "/home/jovyan/swc_dir/"
		os.mkdir(swc_dir)
		counter = 0
		num_rows = len(metadata_filtered)
		for specimen__id in metadata_filtered["specimen__id"]:
		    counter +=1
		    if (counter % 10 == 0):
		        print("Downloading cell " + str(counter) + " of " + str(num_rows))
		    # First, we have to look up where the morphological reconstruction is stored.
		    query_url = "http://api.brain-map.org/api/v2/data/query.json?criteria=model::Specimen[id$eq%d],neuron_reconstructions(well_known_files),rma::include,neuron_reconstructions(well_known_files(well_known_file_type[name$eq'3DNeuronReconstruction']))" % specimen__id 
		    with urllib.request.urlopen(query_url) as response:
		        data=json.load(response)
		    download_link=data['msg'][0]['neuron_reconstructions'][0]['well_known_files'][0]['download_link']
		    url = "http://api.brain-map.org" + download_link
		    file_name = swc_dir + str(specimen__id) + ".swc"
		    with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
		        shutil.copyfileobj(response, out_file)

Sampling from the neurons
^^^^^^^^^^^^^^^^^^^^^^^^^

For our analysis, we will discard the axons of the neurons so that and focus
only on the morphology of the dendrites, so we set `types_keep = [1,3,4]`,
telling the function to keep only the soma and the basal and apical
dendrites. (Not all of the neurons we study have documented axons, so we would get
inconsistent results from including the axons when they exist.) For each
neuron, we sample 50 points from that neuron and compute the geodesic distance
between each pair of points in that neuron.

.. code-block:: python

		from cajal import sample_swc
		sample_swc.compute_and_save_intracell_all(
		    infolder="/home/jovyan/CAJAL/CAJAL/data/swc/",
		    out_csv="/home/patn/CAJAL/CAJAL/data/swc_icdm.csv",
		    metric = "geodesic",
		    types_keep = [1,3,4],
		    n_sample=100,
		    num_cores=8)

Once the sampling is finished, we compute the Gromov-Wasserstein distance
between each pair of neurons. In our experiment, it takes about 0.4s to compute
the Gromov-Wasserstein distance between two cells on a single-core machine, and
processing speed improves about linearly with the number of cores, so on a
machine with 8 cores it might take about 1 hour 48 minutes to compute the
pairwise GW distances between 509 neurons.

.. code-block:: python

		run_gw.compute_gw_distance_matrix(
		    "/home/jovyan/CAJAL/CAJAL/data/swc_icdm.csv",
		    "/home/jovyan/CAJAL/CAJAL/data/swc_gwm.csv",
		    save_mat=False)

Predicting Cre-driver Lines from Cell Morphology
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the Allen Brain Atlas cell types database, each neuron sampled is from a
specific Cre-driver mouse line, and its morphology and electrophysiology have
characteristic distinguishing features which derive from the genes being
studied in that driver line. We might conjecture that cells of the same
Cre-driver line have similar morphologies and that it is possible to guess the
Cre-driver line of a given neuron from its morphological features, by taking
the Cre-driver line labels on some of the neurons as given and classifying the
others based on how similar they are to the neurons for which the correct label
is known. In this experiment, we will divide our neurons into 7 equal parts. 6
parts will be training data for a nearest-neighbors classifier, and the 7th
will be test data. For each neuron in the test data, we will look at the 10
nearest neighbors in the training data (under the GW distance) and guess the
Cre-driver line of the neuron.

We will use pandas, numpy and sk-learn for this analysis.  First, we get all
the mouse neurons which have a full or dendrite-only reconstruction of their
neurons, and get their specimen ids and their Cre-driver lines.

.. code-block:: python

		import pandas as pd
		import numpy as np
		cell_types_specimen_details_loc = "/home/jovyan/CAJAL/CAJAL/data/cell_types_specimen_details.csv"
		metadata = pd.read_csv(cell_types_specimen_details_loc)
		# We consider only the mouse neurons, either full reconstructions or dendrite_only reconstructions. 
		metadata = metadata[(metadata["donor__species"]=="Mus musculus")
                             & ((metadata["nr__reconstruction_type"]=="full") |
                               (metadata["nr__reconstruction_type"]=="dendrite-only"))]
		metadata.index = (metadata["specimen__id"])
		clusters = np.array(metadata["line_name"])
		cell_ids = np.array(metadata["specimen__id"])
		

We read the Gromov-Wasserstein distances into a square matrix, `gw_dist_mat`, which sklearn can
use as a precomputed distance metric. We give two ways to access the data, one can either 
look up the distances in a dictionary as `gw_dist_dictionary[(cell_name1, cell_name2)]`, or
use indices, where we have `gw_dist_mat[i,j]` equal to the distance between cell_names[i] and
cell_names[j].

.. code-block:: python

		from cajal.utilities import read_gw
		from scipy.spatial.distance import squareform

		cell_names, gw_dist_dictionary, gw_dist_arr = read_gw("/home/jovyan/swc_gwm.csv")
		gw_dist_mat = squareform(gw_dist_arr)

Now we use the sklearn library to divide the data into 7 equally sized sets and
classify each element of a given set based on the nearest 10 neighbors in the 6
other sets.

.. code-block:: python

		from sklearn.neighbors import KNeighborsClassifier
		from sklearn.model_selection import StratifiedKFold,cross_val_score,cross_val_predict
		
		clf = KNeighborsClassifier(metric="precomputed", n_neighbors=10, weights="distance")
		cv=StratifiedKFold(n_splits=7, shuffle=True)
		cvs = cross_val_score(clf, X=gw_dist_mat, y=clusters,cv=cv))
		print(cvs)
		# array([0.2739726 , 0.32876712, 0.2739726 , 0.21917808, 0.28767123, 0.31944444, 0.30555556])
		
We see that the average accuracy is between 27% and 30%. However, this number is a bit inflated, as merely evaluating the percentage of correct classifications will underweigh the smallest groups of the dataset. For a more realistic appraisal we will compute the `Matthews correlation coefficient <https://bmcgenomics.biomedcentral.com/counter/pdf/10.1186/s12864-019-6413-7.pdf>_` of the classification, which appropriately weights the error arising from misclassifying elements of smaller classes.

.. code-block:: python

		from sklearn.metrics import matthews_corrcoef
		cvp = cross_val_predict(clf, X=gw_dist_mat, y=clusters, cv=cv)
		print(matthews_corrcoef(cvp,clusters))
		# 0.25205529424157797

So the class-weighted accuracy of the classifier is about 25%.

Use of the graph Laplacian to identify features related to cell morphology
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
nodes of :math:`G`, :math:`n(i)` is the number of neighbors of :math:`i` in
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

Example - C. Elegans Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

This gives us the information we need to compute the graph Laplacians: the features we want to assess,
the GW distance matrix, the distance between points to form the associated graph, and the number of permutations we want to carry out.

.. code-block:: python
		
		results_df_day1 = pd.DataFrame(graph_laplacians(feature_arr_day1,gw_dists_day1,median1, 5000, None, False),index=day1_cols)
		results_df_day2 = pd.DataFrame(graph_laplacians(feature_arr_day2,gw_dists_day2,median2, 5000, None, False),index=day2_cols)
		results_df_day3 = pd.DataFrame(graph_laplacians(feature_arr_day3,gw_dists_day3,median3, 5000, None, False),index=df_day3.columns)
		results_df_day5 = pd.DataFrame(graph_laplacians(feature_arr_day5,gw_dists_day5,median5, 5000, None, False),index=day5_cols)
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
   nrx-1             0.993843            0.535093            0.535093
   mir-1             0.990893            0.441112            0.551390
   nlg-1             0.983587            0.199560            0.332600
   unc-25            0.967470            0.031394            0.156969
   unc-97            0.981699            0.164567            0.411418

   Day 2:
           feature_laplacians  laplacian_p_values  laplacian_q_values
   nrx-1             0.903342            0.102979            0.102979
   unc-97            0.843193            0.024395            0.048790

   Day 3:
           feature_laplacians  laplacian_p_values  laplacian_q_values
   nrx-1             0.980892            0.000200            0.001466
   mir-1             1.000079            0.815637            0.815637
   unc-49            0.997310            0.222356            0.407652
   nlg-1             0.998686            0.493501            0.603168
   unc-25            0.998223            0.391922            0.538892
   unc-97            0.980563            0.000200            0.001466
   lim-6             0.999509            0.689462            0.758408
   lat-2             0.989684            0.001800            0.005656
   ptp-3             0.993579            0.020596            0.045311
   sup-17            0.989100            0.001800            0.005656
   pkd-2             0.997994            0.332733            0.522867

   Day 5:
           feature_laplacians  laplacian_p_values  laplacian_q_values
   nrx-1             0.978943            0.122775            0.122775
   nlg-1             0.934330            0.000800            0.001200
   unc-97            0.829818            0.000200            0.000600

As you can see, from an absolute perspective the Laplacians are not much
smaller than 1; but this is to be expected as the data is 0-1 valued and so we
will not get a nice linear correlation between values. However, for the
nonparametric permutation test, some of the Laplacians are low relative to the
Laplacians of randomly selected functions on the graph with the same range.

The q-values represent the adjustment of the reported p-values by the
Benjamini-Hochberg procedure. After this transformation we can see that some of
the values are still reported as significant. For example, on day 5, after 5000
permutations, none of the observed random permutations generated a Laplacian
score for unc-97 that was as low as the true score.

Through the C. elegans lifecycle the morphology of the neurons changes, so if
we know that the level of expression of a certain gene is correlated with age,
we might expect that the expression of this gene is correlated with cell
morphology indirectly through age. A natural question then is whether the low
Laplacian score for that gene is entirely explained by its correlation with
age, or whether the gene is still correlated with cell morphology after
controlling for the relationship with age.

Let us write :math:`g` for the age of the worm and :math:`f` for the gene
expression vector. For many choices of permutation :math:`\pi` we will sample
points :math:`C_G(f\circ\pi), C_G(g\circ\pi)` and plot a line of best fit to
identify whether there is a linear relationship between :math:`C_G(f\circ\pi)`
and :math:`C_G(g\circ\pi)`. If there is, we will compare the residual
:math:`C_G(f) - \widehat{C_G(f)}` to the other residuals, and reject the null
hypothesis if we observe that this is on the lower tail end of the residuals.

.. code-block:: python

		import os
		import pandas as pd
		import numpy as np
		from cajal.utilities import read_gw, list_sort_files,dist_mat_of_dict
		from cajal.graph_laplacian import graph_laplacians

		project_dir=os.getcwd()
		gw_csv_loc=project_dir+"/c_elegans_gw_dists.csv"
		# Get the binary features we're trying to classify from the features file.
		# There are 11 binary features on the 799 neurons, and we want to identify the ones which are correlated with cell morphology.
		features_file = project_dir+"/c_elegans_features.csv"
		# Get the cell names and the GW distance dictionary from file.
		cell_names, gw_dist_dict = read_gw(gw_csv_loc,header=True)
		feature_matrix = pd.read_csv(features_file)
		feature_matrix.index = feature_matrix['cell_name']
		feature_matrix=feature_matrix.drop('cell_name',axis=1)
		feature_arr = feature_matrix.to_numpy()
		gw_dist_arr = dist_mat_of_dict(feature_matrix.index,gw_dist_dict)

		covariates : list[float] = []
		for a in feature_matrix.index:
		    if "day1" in a:
		        covariates.append(1.0)
               	    elif "day2" in a:
                        covariates.append(2.0)
		    elif "day3" in a:
           		covariates.append(3.0)
		    elif "day5" in a:
	        	covariates.append(5.0)
                    else:
                        raise exception("No day found.")


		covariates = np.array(covariates, dtype=np.float_)
		epsilon= statistics.median(gw_dist_arr) # 71.26842320321848
		N = 799
		T, other = graph_laplacians(
		    feature_arr,
		    gw_dist_arr,
		    epsilon,
		    5000,
		    covariates,
		    False)
		
		df = pd.DataFrame(T)
		df.index = feature_matrix.columns
		print(df)

.. raw:: html

	 <embed> <div style="overflow-x:auto;">
	 <table border="1" class="dataframe"> <thead> <tr style="text-align:
	 right;"> <th></th> <th>feature_laplacians</th> <th>laplacian_p_values</th>
	 <th>laplacian_q_values</th> <th>beta_0</th> <th>beta_1</th>
	 <th>beta_1_p_value</th> <th>regression_coefficients_fstat_p_values</th>
	 <th>laplacian_p_values_post_regression</th>
	 <th>laplacian_q_values_post_regression</th> </tr> </thead> <tbody> <tr>
	 <th>nrx-1</th> <td>0.995131</td> <td>0.010398</td> <td>0.022875</td>
	 <td>0.989490</td> <td>0.009513</td> <td>0.247961</td> <td>0.495922</td>
	 <td>0.014597</td> <td>0.032114</td> </tr> <tr> <th>mir-1</th>
	 <td>0.998708</td> <td>0.374125</td> <td>0.457264</td> <td>0.982360</td>
	 <td>0.016585</td> <td>0.134405</td> <td>0.268809</td> <td>0.656669</td>
	 <td>0.656669</td> </tr> <tr> <th>unc-49</th> <td>0.995577</td>
	 <td>0.021396</td> <td>0.033622</td> <td>0.998180</td> <td>0.000788</td>
	 <td>0.478283</td> <td>0.956566</td> <td>0.022595</td> <td>0.041425</td> </tr>
	 <tr> <th>nlg-1</th> <td>0.992440</td> <td>0.001400</td> <td>0.005132</td>
	 <td>0.961300</td> <td>0.037716</td> <td>0.004166</td> <td>0.008332</td>
	 <td>0.005199</td> <td>0.019063</td> </tr> <tr> <th>unc-25</th>
	 <td>0.993152</td> <td>0.003599</td> <td>0.009898</td> <td>0.933363</td>
	 <td>0.065637</td> <td>0.000004</td> <td>0.000007</td> <td>0.048390</td>
	 <td>0.076042</td> </tr> <tr> <th>unc-97</th> <td>0.958901</td>
	 <td>0.000200</td> <td>0.002200</td> <td>0.984779</td> <td>0.014189</td>
	 <td>0.154183</td> <td>0.308365</td> <td>0.000200</td> <td>0.002200</td> </tr>
	 <tr> <th>lim-6</th> <td>0.999139</td> <td>0.519896</td> <td>0.571886</td>
	 <td>1.009379</td> <td>-0.010522</td> <td>0.750707</td> <td>0.498587</td>
	 <td>0.361528</td> <td>0.441867</td> </tr> <tr> <th>lat-2</th>
	 <td>0.990366</td> <td>0.000800</td> <td>0.004399</td> <td>1.004542</td>
	 <td>-0.005596</td> <td>0.648077</td> <td>0.703847</td> <td>0.000800</td>
	 <td>0.004399</td> </tr> <tr> <th>ptp-3</th> <td>0.997769</td>
	 <td>0.149570</td> <td>0.205659</td> <td>0.995700</td> <td>0.003274</td>
	 <td>0.410331</td> <td>0.820663</td> <td>0.175365</td> <td>0.241127</td> </tr>
	 <tr> <th>sup-17</th> <td>0.994819</td> <td>0.014397</td> <td>0.026395</td>
	 <td>1.026308</td> <td>-0.027426</td> <td>0.966689</td> <td>0.066623</td>
	 <td>0.005999</td> <td>0.016497</td> </tr> <tr> <th>pkd-2</th>
	 <td>0.999256</td> <td>0.556689</td> <td>0.556689</td> <td>1.000614</td>
	 <td>-0.001721</td> <td>0.543784</td> <td>0.912432</td> <td>0.525695</td>
	 <td>0.578264</td> </tr> </tbody> </table> </embed>

We ignore the last two columns for any feature which does not have a small
value for `regression_coefficients_fstat_p_values`, which here represents the
probability that we would observe this data given that the feature and the
covariate are independent and normally distributed.

p-value for `regression_coefficients_fstat_p_values` as in this case the line of best fit and the residuals are meaningless.
