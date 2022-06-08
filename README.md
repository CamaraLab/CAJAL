# CAJAL
```CAJAL``` uses Gromov-Wasserstein (GW) distance to compare cell shapes

## Installation
Until we upload this package to PyPI, the pip installation works from GitHub:
```commandline
pip install git+https://github.com/CamaraLab/CAJAL.git
```

## Docker
We will eventually release a Docker image running Jupyter with a stable version of this package installed, but for now the below command runs a container with all dependencies installed:
```commandline
docker run -it -p 8888:8888 -e GRANT_SUDO=yes --user root camaralab/python3:ot
```

## Overview
[CAJAL.lib.run_gw](https://github.com/CamaraLab/CAJAL/blob/main/CAJAL/lib/run_gw.py) contains the bulk of the functions which use the Python Optimal Transport (POT) library to compute the GW distance between pairs of point clouds or distance matrices.

[CAJAL.lib.sample_mesh](https://github.com/CamaraLab/CAJAL/blob/main/CAJAL/lib/sample_mesh.py), [CAJAL.lib.sample_swc](https://github.com/CamaraLab/CAJAL/blob/main/CAJAL/lib/sample_swc.py), and [CAJAL.lib.sample_tiff](https://github.com/CamaraLab/CAJAL/blob/main/CAJAL/lib/sample_tiff.py) provide helper functions for sampling points and computing geodesic distance from (respectively) triangular meshes in OBJ format, neuron reconstructions in SWC format, and 2D segmentation TIFF images.

## Examples

[Saving pairwise GW distances between triangular meshes](https://github.com/CamaraLab/CAJAL/blob/main/notebooks/save_gw_obj_mesh.ipynb)

Sample points from a triangular mesh from OBJ format and compute geodesic distance using heat or graph methods. Compute Euclidean distance of point coordinates, or load geodesic distances in vector form (output by scipy.spatial.distance.squareform on symmetric matrices). Save pairwise GW between cells on those distances.

----

[Saving pairwise GW distances between neurons](https://github.com/CamaraLab/CAJAL/blob/main/notebooks/save_gw_neurons.ipynb)

Sample points radially from neuron tracing SWC format. Compute Euclidean distance of point coordinates, or load graph geodesic distances in vector form (output by scipy.spatial.distance.squareform on symmetric matrices). Save pairwise GW between neurons on those distances.

----

[Visualizing pairwise GW distances between neurons](https://github.com/CamaraLab/CAJAL/blob/main/notebooks/analyze_gw_neurons.ipynb)

Visualize GW morphology space using a UMAP embedding, plot other morphology features, and compute the average neuron graph shape for each cluster.