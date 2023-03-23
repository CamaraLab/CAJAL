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

In a hypothetical scenario where computational speed is infinite, comparing two
intracellular distance matrices would be conducted as follows. Consider cells *A* and *B*,
each containing 50 selected sample points. The distance between sample points *i* and *j*
in cell *A* can be denoted as *A(i,j)*. If we have a pairing *f* between the sample
points of *A* and *B*, we can consider *f* as an attempt to superimpose *A* on *B*. The
distortion of *A* that arises from this pairing can be quantified by:

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

- Govek, K. W., et al. `Analysis and integration of single-cell morphological data using metric geometry. <https://www.biorxiv.org/content/10.1101/2022.05.19.492525v3>`_ (2022). DOI: 10.1101/2022.05.19.492525 (bioRxiv).

- Mémoli, F. `On the use of Gromov-Hausdorff distances for shape comparison. <https://facundo-memoli.org/papers/dghlp-PBG-fin.pdf>`_ Eurographics
Symposium on Point-Based Graphics (2007)

- Mémoli, F. `Gromov–Wasserstein distances and the metric approach to object matching. <https://media.adelaide.edu.au/acvt/Publications/2011/2011-Gromov–Wasserstein%20Distances%20and%20the%20Metric%20Approach%20to%20Object%20Matching.pdf>`_ Foundations
of computational mathematics 11, 417-487 (2011).

- Mémoli, F. & Sapiro, G. `A theoretical and computational framework for isometry invariant recognition of point cloud data. <http://graphics.stanford.edu/courses/cs468-08-fall/pdf/isodgh.pdf>`_ Foundations of Computational Mathematics 5, 313-347 (2005).