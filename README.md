# CAJAL
```CAJAL``` uses Gromov-Wasserstein (GW) distance to compare cell shapes

## Installation
Until we upload this package to PyPI, the pip installation works from GitHub:
```commandline
pip install git+https://github.com/CamaraLab/CAJAL.git@package_dev
```

## Docker
We will eventually release a Docker image running Jupyter with a stable version of this package installed, but for now the below command runs a container with all dependencies installed:
```commandline
docker run -it -p 8888:8888 -e GRANT_SUDO=yes --user root camaralab/python3:ot
```

## Overview
[run_gw.py](https://github.com/CamaraLab/CAJAL/blob/package_dev/CAJAL/lib/run_gw.py) contains the bulk of the functions which use the Python Optimal Transport (POT) library to compute the GW distance between pairs of point clouds or distance matrices.


## Examples

__TODO: Sampling even points from SWC neuron reconstruction and saving geodesic distance.__

----

[Saving pairwise GW distances between cells](https://github.com/CamaraLab/CAJAL/blob/package_dev/notebooks/save_gw_pairwise.ipynb)

Compute Euclidean distance of point coordinates from cell boundary or skeleton, or load geodesic distances in vector form (output by scipy.spatial.distance.squareform on symmetric matrices). Save pairwise GW between cells on those distances.

----

__TODO: Reading in GW distance matrix and visualizing morphology summary space.__