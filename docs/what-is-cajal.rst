.. -*- coding: utf-8 -*-

What is CAJAL?
==============

CAJAL is a general computational framework for the multi-modal analysis and integration of single-cell morphological data.

The structure, shape and size of a cell are important to understand its function, its regulation of cell activities, and its interaction with other cells. Other tools to compare and analyze the
shape of cells require the analyst to fit the cell into an existing framework or paradigm of possible cell shapes. For example, a record in a table may contain the cell's length along a central axis, its diameter at the midsection, and so on. For more complex shapes, like that of a neuron, a record may contain the length of its axon, the number of primary dendrites, whether it is unipolar, bipolar, multipolar, and so on.

There are some flaws with this approach. First, the choice of these features necessarily involves a bias as to which aspects of the cells are geometrically important and significant. Second, the categorization schema constructed by the analyst may be hard to apply unambiguously, requiring judgement calls in how to measure some aspect of a cell (for example, the branching degree of the dendrites of a neuron.) It may even be the case that the cell simply does not fit the model at all, forcing the analyst to revise and expand the model or omit some data points.

CAJAL does not use a predefined feature list to represent a cell. CAJAL's internal representation of a cell is a list of points randomly sampled from the surface of the cell (usually between 50 and 200) together with a matrix of the pairwise distances between these points.

We will first give an oversimplified explanation of how CAJAL works. In an ideal world where computation was infinitely fast, CAJAL would compare two intracell distance matrices as follows.
Say we have cells *A* and *B*, each with 50 chosen sample points. Let us write *A(i,j)* for the distance between sample points *i* and *j* in *A*. If *f* is any one-to-one pairing between the sample points of *A* and *B*, then we can regard *f* as an attempt to overlay *A* on *B*. The distortion of *A* associated to this pairing is measured by

.. math::  \Gamma_f = \max_{i,j \in A} \lvert A(i,j) - B(f(i),f(j)) \rvert

This quantifies how much *A* has to be deformed or stretched in order to overlay it on *B* along the given pairing.

The Gromov-Hausdorff distance between *A* and *B* is then defined as the distortion arising from the best possible pairing, when all possible one-to-one pairings are considered.

.. math::  d_{GH}(A,B) = \min_{f : A\cong B} \max_{i,j \in A} \lvert A(i,j) - B(f(i),f(j)) \rvert

Unfortunately, this quantity cannot be computed in practice, as there are 50! or about 3x10^64 ways to give a one-to-one pairing between the points of A and B, and we cannot search through all of these. Therefore, CAJAL relies on a more computationally efficient approximation, the Gromov-Wasserstein distance. Both the Gromov-Hausdorff distance and the Gromov-Wasserstein distance satisfy the axioms for a metric, meaning that they give a sensible and reasonably well-behaved notion of distance.

CAJAL provides tools to compute the pairwise Gromov-Wasserstein distance between all cells in a directory of cell image data. Building on this foundation, the analyst can then use standard clustering
tools to identify groups of cells with similar morphology, and use this to predict features of a new cell by comparing its shape with other cells, or investigate whether a cell feature is highly
correlated with its morphology.

CAJAL also provides visualization tools for the user to interpret the results of the Gromov-Wasserstein distance computation.

CAJAL is written and developed by the `Cámara Lab <https://camara-lab.org/>`_ at the University of Pennsylvania. 
