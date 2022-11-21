# escape=`
# The above line is a parser directive, not a comment.

# This Dockerfile builds a Docker image containing CAJAL and all its dependencies, from a local git repo for CAJAL.
# To run this Dockerfile (build the docker image), put the CAJAL Git repository and this Dockerfile together in the same
# folder. If this file is not already named "Dockerfile", rename it to "Dockerfile" - no suffix. Navigate to that folder and run "docker build ." i.e.
# $ ls myfolder
# CAJAL Dockerfile
# $ cd myfolder; docker build -t myimagename .
# Do not use the syntax "docker build - < Dockerfile" for this file.

# We build on the jupyter/tensorflow-notebook.
# The hash value following the image name marks a specific iteration of the jupyter/tensorflow-notebook,
# occurring earlier in the build chain. The version below was pushed to Dockerhub on November 8, 2022.
# The incorporated version of Python is 3.10. The incorporated version of Jupyter is 3.5.0.

FROM jupyter/tensorflow-notebook@sha256:c224e3a2c4f5180ab50913d129fe650f64033b3729e49dfb92968d9640cff544

# A guide to the different Docker images provided by Jupyter can be found here:
# https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html
# The tensorflow-notebook image is somewhat "maximalist". It is 4GB on its own. It provides most of the tools one needs
# to work within the Docker image and analyze the results presented by CAJAL, with the idea that
# one would not just use the Docker image to run CAJAL but as a full computing environment/workspace
# to do other kinds of analysis with the results provided by CAJAL.

# To see a complete list of all the Python packages and their versions that jupyter installs in the tensorflow-notebook,
# run "conda list" or "pip list".

# When editing this command, be careful to preserve the space between the version number and the backtick.
# There should not be any space or comments after the backtick; only a newline/carriage return.
# The job of the backtick is to tell Docker to ignore the newline/carriage return,
# letting us write the statement in a more readable way.
RUN python3 -m pip install `
igraph==0.10.2 `
leidenalg==0.9.0 `
pot==0.7.0 `
potpourri3d==0.0.8 `
pynndescent==0.5.8 `
python-louvain==0.16 `
texttable==1.6.4 `
trimesh==3.16.1 `
umap-learn==0.5.3 `
--upgrade setuptools

# The other dependencies to CAJAL (all listed in setup.py) are already present in the jupyter/tensorflow-notebook image.
# We list them here, together with their versions in the given version of jupyter/tensorflow-notebook, for convenience.
# matplotlib==3.6.2
# networkx==2.8.8
# numpy==1.23.4
# pandas==1.5.1
# scipy==1.9.3
# scikit-image==0.19.3
# tifffile==2022.10.10

# This approach copies the entire git repo into the container and runs pip install on it.
# It seems more elegant to run pip install without copying the files,
# but the tool egg_info expects to be able to write to the directory it's installing from.
# I tried using the rw flag for --mound=type=bind to fix this, but it didn't work.

RUN --mount=type=bind,target=/home/cajal_repo,source=CAJAL`
    mkdir /home/jovyan/CAJAL ; `
    cp -r /home/cajal_repo/* /home/jovyan/CAJAL/ ;`
    python3 -m pip install /home/jovyan/CAJAL
