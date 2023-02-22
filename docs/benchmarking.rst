Worked Examples
===============

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

Visualizing and Clustering Data
-------------------------------
