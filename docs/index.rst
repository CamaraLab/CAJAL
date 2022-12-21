.. cajal documentation master file, created by
   sphinx-quickstart on Mon Nov 21 14:31:18 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Cajal : a cell morphology analysis package
==========================================

Cajal is a suite of tools to study cell morphology and its relationship with other data.

Cajal uses the Python Optimal Transport library (POT) to compute the Gromov-Wasserstein distance between two cells. The Gromov-Wasserstein distance is a similarity score which is small
if the cells are geometrically similar in the sense that one cell can be deformed into another without too much stretching or bending. Computing the Gromov-Wasserstein distance between cells does not require the user to fit the cells into a given model of what a cell looks like - for example, to compare two neurons, one does not have to supply the length of the axon, the branching degree of the dendrites, or the number of branches. Thus, Cajal can be used to study cells whose morphology is highly complex and defies straightforward categorization.

.. toctree::
   :maxdepth: 2
   :caption: Overview and Walkthrough

   what-is-cajal
   computing-intracell-distance-matrices
   computing-gw-distances

.. toctree::
   :maxdepth: 2
   :caption: API

   sample_swc
   sample_mesh
   sample_seg

.. This is a comment.
   \\:hidden:
   \\To add a caption in the TOC use :caption: in the toctree, i.e. :caption: First steps

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
