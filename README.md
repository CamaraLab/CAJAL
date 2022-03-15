# CAJAL
```CAJAL``` uses Gromov-Wasserstein (GW) distance to compare cell shapes


## Overview
[run_gw.py](https://github.com/CamaraLab/MorphGW/blob/main/run_gw.py) contains the bulk of the functions which use the Python Optimal Transport (POT) library to compute the GW distance between pairs of point clouds or distance matrices.

[gw_pairwise_roi.ipynb](https://github.com/CamaraLab/MorphGW/blob/main/gw_pairwise_roi.ipynb) is a Jupyter Notebook demonstrating calls to the functions from ```run_gw.py```. It has already been run once on the example folders in [sampled_pts/](https://github.com/CamaraLab/MorphGW/tree/main/sampled_pts) and the resulting GW distance matrices are saved in [gw_results/](https://github.com/CamaraLab/MorphGW/tree/main/gw_results) for comparison.