<p align="center"><img src="https://github.com/CamaraLab/CAJAL/blob/main/logo.png" width="320" align="center"></p><hr />

_CAJAL_ is a python library for multi-modal cell morphology analyses using Gromov-Wasserstein (GW) distance. Detailed information about the methods implemented in CAJAL can be found in:

K. W. Govek, J. Crawford, A. B. Saturnino, K. Zoga, M. P. Hart, P.  G. Camara, _Multimodal analysis and integration of single-cell morphological data_. bioRxiv (2022). [DOI:10.1101/2022.05.19.492525](https://www.biorxiv.org/content/10.1101/2022.05.19.492525v3.full)

## Installation
Until we upload the package to PyPI, the pip installation works from GitHub:
```commandline
pip install git+https://github.com/CamaraLab/CAJAL.git
```
Installation on a standard desktop computer should take a few minutes.

----

On Windows, the Python Optimal Transport (pot) library requires Microsoft Visual C++ 14.0 or greater, which can be installed via the [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/). On Ubuntu, it requires g++ and may require the package python3.x-dev, which registers the Python header files with g++.

----

The easiest way to run _CAJAL_ is via [Jupyter](https://jupyter.org/). The _CAJAL_ tutorials are also provided as Jupyter Notebooks. Install Jupyter with
```commandline
pip install notebook
```
Then start up Jupyter from terminal / Powershell using
```commandline
jupyter notebook
```

## Docker image
We provide two Docker images which contain _CAJAL_ and its dependencies, ```cajal:minimal``` and ```cajal:maximal```. ```cajal:minimal``` is built on top of the Jupyter notebook Docker image ```base-notebook``` and contains only _CAJAL_ and its dependencies, ```cajal:maximal``` is built on top of the Docker image ```tensorflow-notebook``` and contains numerous data science tools for further analysis of the output of _CAJAL_. Running the following command will launch a Jupyter notebook server on localhost with _CAJAL_ and its dependencies installed:
```commandline
docker run -it -p 8888:8888 -v C:\Users\myusername\Documents\myfolder:/home/jovyan/work camaralab/cajal:maximal
```
The ```-p``` flag controls the port number on local host. For example, writing ```-p 4264:8888``` will let you access the Jupyter server from 127.0.0.1:4264. The ```-v``` "bind mount" flag allows one to mount a local directory on the host machine to a folder inside the container so that you can read and write files on the host machine from within the Docker image. Here one must mount the folder on the host machine as /home/jovyan/work or /home/jovyan/some_other_folder as the primary user "jovyan" in the Docker image only has access to that directory and to the /opt/conda folder. See the [Jupyter docker image documentation](https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html) for more information.

## Documentation
Extensive documentation, including several tutorials, can be found in [_CAJAL_'s readthedocs.io website](https://cajal.readthedocs.io/en/readthedocs_dev/). This website is under development and will continue to be substantially updated during the coming months.

In a nutshell:

[CAJAL.lib.run_gw](https://github.com/CamaraLab/CAJAL/blob/main/CAJAL/lib/run_gw.py) contains the bulk of the functions which use the Python Optimal Transport (POT) library to compute the GW distance between pairs of point clouds or distance matrices.

[CAJAL.lib.sample_mesh](https://github.com/CamaraLab/CAJAL/blob/main/CAJAL/lib/sample_mesh.py), [CAJAL.lib.sample_swc](https://github.com/CamaraLab/CAJAL/blob/main/CAJAL/lib/sample_swc.py), and [CAJAL.lib.sample_seg](https://github.com/CamaraLab/CAJAL/blob/main/CAJAL/lib/sample_seg.py) provide helper functions for sampling points and computing geodesic distance from (respectively) triangular meshes in OBJ format, neuron reconstructions in SWC format, and 2D segmentation TIFF images.

These are some examples of usage:

- [Saving pairwise GW distances between triangular meshes](https://github.com/CamaraLab/CAJAL/blob/main/notebooks/save_gw_obj_mesh.ipynb). Sample points from a triangular mesh from OBJ format and compute geodesic distance using heat or graph methods. Compute Euclidean distance of point coordinates, or load geodesic distances in vector form (output by scipy.spatial.distance.squareform on symmetric matrices). Save pairwise GW between cells on those distances.

- [Saving pairwise GW distances between neurons](https://github.com/CamaraLab/CAJAL/blob/main/notebooks/save_gw_neurons.ipynb). Sample points radially from neuron tracing SWC format. Compute Euclidean distance of point coordinates, or load graph geodesic distances in vector form (output by scipy.spatial.distance.squareform on symmetric matrices). Save pairwise GW between neurons on those distances.

- [Visualizing pairwise GW distances between neurons](https://github.com/CamaraLab/CAJAL/blob/main/notebooks/analyze_gw_neurons.ipynb). Visualize GW morphology space using a UMAP embedding, plot other morphology features, and compute the average neuron graph shape for each cluster.
