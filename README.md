# CAJAL <a href='https://github.com/CamaraLab/CAJAL'><img src="docs/images/logo.png" align="right" width="30%"/></a>

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
