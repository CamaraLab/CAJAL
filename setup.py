from setuptools import setup
from Cython.Build import cythonize
import numpy

include_path = [numpy.get_include()]

setup(
    ext_modules=cythonize(
        "src/cajal/*.pyx", gdb_debug=True, compiler_directives={"language_level": "3"}
    ),
    include_dirs=[numpy.get_include()],
)
