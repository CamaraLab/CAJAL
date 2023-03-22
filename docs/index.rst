.. CAJAL documentation master file, created by
   sphinx-quickstart on Mon Nov 21 14:31:18 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

CAJAL: a python package for multimodal analyses of single-cell morphological data
=================================================================================

CAJAL is a python package designed to explore and analyze the morphology of cells
and its relationship with other single-cell data.

CAJAL leverages the Python Optimal Transport library to compute the
Gromov-Wasserstein (GW) distance between every pair of cells in a given sample. This
distance quantifies the degree to which the shape of one cell can
be transformed into that of another with minimal stretching or bending. One of the
key benefits of using the GW distance is that it does not require any prior
knowledge or model for the morphology of the cells. This feature makes CAJAL suitable
for studying arbitrarily heterogeneous mixtures of cells with highly complex and diverse
morphologies that may defy straightforward classification.

The morphological distance produced by CAJAL is a bona-fide mathematical distance
in a latent space of cell morphologies. In this latent space, each cell is represented
by a point, and distances between cells indicate the amount of physical deformation
needed to change the morphology of one cell into that of another. By formulating the
problem in this way, CAJAL can make use of standard statistical and machine learning approaches to
define cell populations based on their morphology; dimensionally reduce and visualize
cell morphology spaces; and integrate cell morphology spaces across tissues, technologies,
and with other single-cell data modalities, among other analyses.

.. toctree::
   :maxdepth: 2
   :caption: Overview and Walkthrough

   what-is-cajal
   computing-intracell-distance-matrices
   computing-gw-distances
   benchmarking
   average_swc_shape
   notebooks/Example_1

.. toctree::
   :maxdepth: 2
   :caption: API

   swc
   sample_swc
   sample_mesh
   sample_seg
   run_gw
   graph_laplacian
   average_cell_shapes

.. This is a comment.
   \\:hidden:
   \\To add a caption in the TOC use :caption: in the toctree, i.e. :caption: First steps

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
