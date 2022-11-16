# CAJAL
```CAJAL``` uses Gromov-Wasserstein (GW) distance to compare cell shapes

## Installation
Until we upload this package to PyPI, the pip installation works from GitHub:
```commandline
pip install git+https://github.com/CamaraLab/CAJAL.git
```
Installation on a standard desktop computer should take a few minutes.

----

On Windows, the Python Optimal Transport (pot) library requires Microsoft Visual C++ 14.0 or greater, which can be installed via the [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

----

The easiest way to run CAJAL's plotting functions is via [Jupyter](https://jupyter.org/). The CAJAL tutorials are also provided as Jupyter Notebooks. Install Jupyter with
```commandline
pip install notebook
```
Then start up Jupyter from terminal / Powershell using
```commandline
jupyter notebook
```

## Docker
We provide a Docker image, built on top of the Jupyter/tensorflow-notebook image, which contains CAJAL and its dependencies. Running the following command will launch a Jupyter notebook server on localhost with CAJAL and its dependencies installed. The -p flag controls the port number on local host, writing "-p 4264:8888" will let you access the Jupyter server from 127.0.0.1:4264. The -v "bind mount" flag allows one to mount a local directory on the host machine to a folder inside the container so that you can read and write files on the host machine from within the Docker image. Here one must mount the folder on the host machine as /home/jovyan/work or /home/jovyan/someotherfolder as the primary user "jovyan" in the Docker image only has access to that directory and to the /opt/conda folder. See https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html
```commandline
docker run -it -p 8888:8888 -v C:\Users\myusername\Documents\myfolder:/home/jovyan/work camaralab/cajal:maximal
```

## Overview
[CAJAL.lib.run_gw](https://github.com/CamaraLab/CAJAL/blob/main/CAJAL/lib/run_gw.py) contains the bulk of the functions which use the Python Optimal Transport (POT) library to compute the GW distance between pairs of point clouds or distance matrices.

[CAJAL.lib.sample_mesh](https://github.com/CamaraLab/CAJAL/blob/main/CAJAL/lib/sample_mesh.py), [CAJAL.lib.sample_swc](https://github.com/CamaraLab/CAJAL/blob/main/CAJAL/lib/sample_swc.py), and [CAJAL.lib.sample_seg](https://github.com/CamaraLab/CAJAL/blob/main/CAJAL/lib/sample_seg.py) provide helper functions for sampling points and computing geodesic distance from (respectively) triangular meshes in OBJ format, neuron reconstructions in SWC format, and 2D segmentation TIFF images.

## Examples

[Saving pairwise GW distances between triangular meshes](https://github.com/CamaraLab/CAJAL/blob/main/notebooks/save_gw_obj_mesh.ipynb)

Sample points from a triangular mesh from OBJ format and compute geodesic distance using heat or graph methods. Compute Euclidean distance of point coordinates, or load geodesic distances in vector form (output by scipy.spatial.distance.squareform on symmetric matrices). Save pairwise GW between cells on those distances.

----

[Saving pairwise GW distances between neurons](https://github.com/CamaraLab/CAJAL/blob/main/notebooks/save_gw_neurons.ipynb)

Sample points radially from neuron tracing SWC format. Compute Euclidean distance of point coordinates, or load graph geodesic distances in vector form (output by scipy.spatial.distance.squareform on symmetric matrices). Save pairwise GW between neurons on those distances.

----

[Visualizing pairwise GW distances between neurons](https://github.com/CamaraLab/CAJAL/blob/main/notebooks/analyze_gw_neurons.ipynb)

Visualize GW morphology space using a UMAP embedding, plot other morphology features, and compute the average neuron graph shape for each cluster.
