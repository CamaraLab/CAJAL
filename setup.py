from setuptools import setup
from Cython.Build import cythonize
import numpy
import os


ROOT = os.path.abspath(os.path.dirname(__file__))
include_path = [numpy.get_include()]

setup(
    ext_modules=cythonize(["src/cajal/*.pyx"]),
    compiler_directives={"language_level": "3"},
    include_dirs=include_path,
)
