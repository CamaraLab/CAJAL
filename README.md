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

[CAJAL.lib.sample_mesh](https://github.com/CamaraLab/CAJAL/blob/main/CAJAL/lib/sample_mesh.py), [CAJAL.lib.sample_swc](https://github.com/CamaraLab/CAJAL/blob/main/CAJAL/lib/sample_swc.py), and [CAJAL.lib.sample_seg](https://github.com/CamaraLab/CAJAL/blob/main/CAJAL/lib/sample_seg.py) provide helper functions for sampling points and computing geodesic distance from (respectively) triangular meshes in OBJ format, neuron reconstructions in SWC format, and 2D segmentation TIFF images.

## Examples

__TODO: Sampling even points from SWC neuron reconstruction and saving geodesic distance.__

----

[Saving pairwise GW distances between cells](https://github.com/CamaraLab/CAJAL/blob/main/notebooks/save_gw_pairwise.ipynb)

Compute Euclidean distance of point coordinates from cell boundary or skeleton, or load geodesic distances in vector form (output by scipy.spatial.distance.squareform on symmetric matrices). Save pairwise GW between cells on those distances.

----

__TODO: Reading in GW distance matrix and visualizing morphology summary space.__
