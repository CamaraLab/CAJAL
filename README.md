# CAJAL <a href='https://github.com/CamaraLab/CAJAL'><img src="docs/images/logo.png" align="right" width="24%"/></a>
[![Build and Test](https://github.com/CamaraLab/CAJAL/actions/workflows/python-package.yml/badge.svg?branch=main)](https://github.com/CamaraLab/CAJAL/actions/workflows/python-package.yml)
[![codecov](https://codecov.io/github/CamaraLab/CAJAL/branch/main/graph/badge.svg?token=RU5ZR1SE8Z)](https://codecov.io/github/CamaraLab/CAJAL)
![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/CamaraLab/CAJAL?include_prereleases&color=green)

CAJAL is a Python library for multi-modal cell morphology analyses using Gromov-Wasserstein (GW) distance. Detailed information about the methods implemented in CAJAL can be found in:

K. W. Govek, P. Nicodemus, Y. Lin, J. Crawford, A. B. Saturnino, H. Cui, K. Zoga, M. P. Hart, P.  G. Camara, _Multimodal analysis and integration of single-cell morphological data usig metric geometry_. bioRxiv (2022). [DOI:10.1101/2022.05.19.492525](https://www.biorxiv.org/content/10.1101/2022.05.19.492525v3.full)

## Installation
Until we upload the package to PyPI, the pip installation works from GitHub:
```commandline
pip install git+https://github.com/CamaraLab/CAJAL.git
```
Installation on a standard desktop computer should take a few minutes.

----

A C++ compiler is required for the Gromov-Wasserstein computation and may be required for the potpourri3d library if the precompiled binaries are not compatible with your system.
On Windows, we recommend Microsoft Visual C++ 14.0 or greater, which can be installed via the [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/). On Ubuntu, it requires g++ and may require the package python3.x-dev, which registers the Python header files with g++.

----

The easiest way to run CAJAL is via [Jupyter](https://jupyter.org/). Install Jupyter with
```commandline
pip install notebook
```
Then start up Jupyter from terminal / Powershell using
```commandline
jupyter notebook
```

## Docker image
We provide two Docker images which contain CAJAL and its dependencies, ```cajal:minimal``` and ```cajal:maximal```. ```cajal:minimal``` is built on top of the Jupyter notebook Docker image ```base-notebook``` and contains only CAJAL and its dependencies, ```cajal:maximal``` is built on top of the Docker image ```tensorflow-notebook``` and contains numerous data science tools for further analysis of the output of CAJAL. Running the following command will launch a Jupyter notebook server on localhost with CAJAL and its dependencies installed:
```commandline
docker run -it -p 8888:8888 -v C:\Users\myusername\Documents\myfolder:/home/jovyan/work camaralab/cajal:maximal
```
The ```-p``` flag controls the port number on local host. For example, writing ```-p 4264:8888``` will let you access the Jupyter server from 127.0.0.1:4264. The ```-v``` "bind mount" flag allows one to mount a local directory on the host machine to a folder inside the container so that you can read and write files on the host machine from within the Docker image. Here one must mount the folder on the host machine as /home/jovyan/work or /home/jovyan/some_other_folder as the primary user "jovyan" in the Docker image only has access to that directory and to the /opt/conda folder. See the [Jupyter docker image documentation](https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html) for more information.

## Documentation
Extensive documentation, including several tutorials, can be found in [CAJAL's readthedocs.io website](https://cajal.readthedocs.io/en/latest/index.html). This website is under development and will continue to be substantially updated during the coming months.
