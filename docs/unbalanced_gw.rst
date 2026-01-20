Unbalanced Gromov-Wasserstein module
====================================

We have implemented a Python module that allows the user to compute the unbalanced Gromov-Wasserstein distance between cells. By default, CAJAL ships with a single core version of the algorithm and a multicore version, and the user can uncomment the appropriate line in the package's `setup.py` build script to get a version of the algorithm for a GPU using either CUDA or OpenCL. These are disabled by default as the end user must configure their machine so that the CUDA (respectively, OpenCL) header files can be found and all necessary libraries are available. A few other backends can be made available upon request. Our experience shows that the GPU backends are only likely to be useful when the individual UGW problems are very large (i.e., the metric spaces are large)

The user should only import one of the backend modules at a time due to technical limitations of C (C has no namespacing, so there will be symbol conflicts from identically named functions in the two backend modules).
Restart the Python interpreter if you want to load a different backend module.

The basic usage is that we import the backend module we want to use (in this case the multicore implementation) and the UGW class.
The constructor for the UGW class takes the backend module as its argument, establishes a connection with the library, and returns an object that maintains the internal state of the computation.

.. code-block::  python

    from cajal.ugw import _multicore, UGW # Substitute _single_core for single-threaded usage, useful if you want to parallelize at the level of Python processes
    UGW_multicore = UGW(_multicore) # For GPU backends, the constructor has to negotiate a connection to the GPU, so it may take a long time to initialize.

The wrapper functions for the C backend are then accessible as *methods* of this object.
For example, the ".from_futhark()" method converts the library's internal representation of the output to a Numpy array.
If the user intends to parallelize at the level of Python processes, each process should instantiate the class.
As usual one can call `help(UGW_multicore)`, `help(UGW_multicore.ugw_armijo)`, and so on for documentation of the functions.

Some utilities
--------------
.. autofunction:: cajal.ugw.ugw_bound
.. autofunction:: cajal.ugw.mass_lower_bound
.. autofunction:: cajal.ugw.estimate_distr
.. autofunction:: cajal.ugw.rho_of

UGW class and methods
---------------------

.. module:: cajal.ugw
.. class:: cajal.ugw.UGW
.. autofunction:: cajal.ugw.UGW.ugw_armijo
.. autofunction:: cajal.ugw.UGW.ugw_armijo_pairwise_increasing
.. autofunction:: cajal.ugw.UGW.ugw_armijo_pairwise
.. autofunction:: cajal.ugw.UGW.ugw_armijo_euclidean