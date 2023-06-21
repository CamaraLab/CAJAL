from setuptools import setup
from Cython.Build import cythonize
import numpy
import os
import sys

ROOT = os.path.abspath(os.path.dirname(__file__))
include_path = [numpy.get_include(), os.path.join(ROOT, "CAJAL/src/cajal")]

compile_args = ["/O2" if sys.platform == "win32" else "-O3"]
link_args = []

setup(
    ext_modules=cythonize(
        ["src/cajal/*.pyx", "src/cajal/EMD_wrapper.cpp"], language="c++", annotate=True
    ),
    include_dirs=[numpy.get_include(), os.path.join(ROOT, "CAJAL/src/cajal")],
)
