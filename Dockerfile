# escape=`
# The above line is a parser directive, not a comment.

# This Dockerfile builds a Docker image containing CAJAL and all its dependencies, from a local git repo for CAJAL.
# To run this Dockerfile (build the docker image), put the CAJAL Git repository and this Dockerfile together in the same
# folder. Rename this Dockerfile to "Dockerfile" if it is not called that already.
# Navigate to that folder and run "docker build ." i.e.
# $ ls myfolder
# CAJAL Dockerfile
# $ cd myfolder; docker build -t myimagename .
# Do not use the syntax "docker build - < Dockerfile" for this file.

# We build on the jupyter/base-notebook.
# The hash value following the image name marks a specific iteration of the jupyter/tensorflow-notebook,
# occurring earlier in the build chain. The version below was pushed to Dockerhub on November 8, 2022.
# The incorporated version of Python is 3.10. The incorporated version of Jupyter is 3.5.0.

# FROM jupyter/base-notebook@sha256:66830d423f076d186560fb52fe32e6af637888f85b6c9b942fb0b0c36e869b7b
# FROM quay.io/jupyter/base-notebook/sha256:0e6d24b2cc644f3170f058fbf752d7765928ee43d2bd9311a1408b566a8a124f
FROM quay.io/jupyter/base-notebook
# Jupyter releases two versions of each Docker image, the one above targets the amd64 architecture, the one below
# targets the arm64 architecture.
# FROM jupyter/base-notebook@sha256:e471b4bf9680c3fb24c799a23fb7240def04b51e88913f877b9a6b411eaa8be2
# Theoretically these two packages are exactly the same except that they run on different CPU's,
# so you should be able to swap one out for the other.

USER root
RUN apt-get update
RUN apt-get upgrade -y
# git is necessary to clone the repository off Github, but we can just download the zip and it works fine.
# RUN apt-get install -y git g++
RUN apt-get install -y g++

# A guide to the different Docker images provided by Jupyter can be found here:
# https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html
# The base-notebook image contains only a "minimally-functional Jupyter Notebook", conda, a "cross-platform, language-agnostic binary package manager", and Jupuyter itself.

# To see a complete list of all the Python packages and their versions that jupyter installs in the tensorflow-notebook,
# run "conda list" or "pip list".

# When editing this command, be careful to preserve the space between the version number and the backtick.
# There should not be any space or comments after the backtick; only a newline/carriage return.
# The job of the backtick is to tell Docker to ignore the newline/carriage return,
# letting us write the statement in a more readable way.
USER jovyan
RUN python3 -m pip install `
PyWavelets==1.4.1 `
contourpy==1.0.6 `
cycler==0.11.0 `
cython==0.29.32 `
fonttools==4.38.0 `
igraph==0.10.2 `
imageio==2.22.4 `
ipywidgets==8.0.2 `
joblib==1.2.0 `
jupyterlab-widgets==3.0.3 `
kiwisolver==1.4.4 `
leidenalg==0.9.0 `
llvmlite `
matplotlib==3.6.2 `
networkx==2.8.8 `
numba `
numpy==1.23.4 `
pandas==1.5.1 `
pillow==9.3.0 `
pot==0.7.0.post1 `
potpourri3d `
pynndescent==0.5.8 `
python-louvain==0.16 `
scikit-image==0.19.3 `
scikit-learn==1.1.3 `
scipy==1.9.3 `
texttable==1.6.4 `
threadpoolctl==3.1.0 `
tifffile==2022.10.10 `
trimesh==3.16.1 `
umap-learn==0.5.3 `
widgetsnbextension==4.0.3 `
 --upgrade setuptools

# This approach copies the entire git repo into the container and runs pip install on it.
# It seems more elegant to run pip install without copying the files,
# but the tool egg_info expects to be able to write to the directory it's installing from.
# I tried using the rw flag for --mound=type=bind to fix this, but it didn't work.

RUN --mount=type=bind,target=/home/cajal_repo,source=CAJAL`
    mkdir /home/jovyan/CAJAL ; `
    cp -r /home/cajal_repo/* /home/jovyan/CAJAL/ ;`
    python3 -m pip install /home/jovyan/CAJAL
