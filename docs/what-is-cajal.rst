.. -*- coding: utf-8 -*-

What is CAJAL?
==============

CAJAL is a general computational framework for the multi-modal analysis and integration of
single-cell morphological data. It builds upon recent advances in applied metric
geometry and shape registration to enable the characterization of morphological
cellular processes from a biophysical perspective and produce a mathematical
distance function upon which algebraic and statistical analytic approaches can be built.

In its simplest form, the study of cell morphology involves comparing cell shapes
irrespective of distance-preserving transformations such as rotations and translations.
To facilitate this, CAJAL internally represents a cell as a list
of points randomly sampled from the surface of the cell (usually between 50 and 200),
together with a matrix of the (Euclidean or geodesic) pairwise distances between these points
in the cell, known as the intracellular distance matrix.

In an ideal world where computation was infinitely fast, we would compare two intracellular distance matrices as follows.
Say we have cells *A* and *B*, each with 50 chosen sample points. Let us write *A(i,j)* for
the distance between sample points *i* and *j* in cell *A*. If *f* is any pairing
between the sample points of *A* and *B*, then we can regard *f* as an attempt to overlay *A* on *B*. The distortion of *A* associated to this pairing is measured by

.. math::  \Gamma_f = \max_{i,j \in A} \lvert A(i,j) - B(f(i),f(j)) \rvert

This quantifies how much *A* has to be deformed or stretched in order to overlay it on *B*
along the given pairing.

The Gromov-Hausdorff distance between *A* and *B* is then defined as the distortion arising from the best possible pairing, when all possible pairings are considered.

.. math::  d_{GH}(A,B) = \min_{f : A\cong B} \max_{i,j \in A} \lvert A(i,j) - B(f(i),f(j)) \rvert

Unfortunately, this quantity cannot be computed in practice, as there are 50! or about 3x10^64 ways
to give a one-to-one pairing between the points of A and B, and we cannot search through all of
these. Therefore, CAJAL relies on a more computationally efficient approximation, the
Gromov-Wasserstein distance. Both the Gromov-Hausdorff distance and the Gromov-Wasserstein distance
satisfy the axioms for a metric, giving a sensible and reasonably well-behaved
notion of distance.

CAJAL provides tools to compute the pairwise Gromov-Wasserstein distance between all cells in a
directory of cell image data and exploring, interpreting, and analyzing the resulting cell
morphology latent space. For example, the user can use clustering approaches to identify groups of
cells with similar morphology and predict features of new cells by comparing their shape
with other cells. They can also investigate whether a cell feature is highly correlated with its morphology. CAJAL provides tools for exploring, interpreting,
and analyzing the cell morphology latent space produced by CAJAL.

CAJAL is written and developed by the `Cámara Lab <https://camara-lab.org/>`_ at the
University of Pennsylvania. More information about the theoretical foundations of CAJAL can be found
at:
