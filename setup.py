from setuptools import setup
from Cython.Build import cythonize
import numpy
import os


ROOT = os.path.abspath(os.path.dirname(__file__))
include_path = [numpy.get_include(), os.path.join(ROOT, "CAJAL/src/cajal")]

setup(
    ext_modules=cythonize(
        ["src/cajal/*.pyx", "src/cajal/EMD_wrapper.cpp"], language="c++"
    ),
    include_dirs=include_path,
)
