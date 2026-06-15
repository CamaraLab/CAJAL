from sys import platform
from setuptools import setup
from Cython.Build import cythonize
import numpy
import os

from setuptools.command.build import build as _build

class build(_build):
    def finalize_options(self):
        super().finalize_options()
        if (platform == "win32"):
            self.compiler = "mingw32"


def skip_ugw_build() -> bool:
    value = os.environ.get("CAJAL_SKIP_UGW", "")
    return value.lower() in {"1", "true", "yes", "on"}

ROOT = os.path.abspath(os.path.dirname(__file__))
include_path = [numpy.get_include()]

ext_modules = cythonize(["src/cajal/*.pyx"])

if skip_ugw_build():
    cffi_modules = []
else:
    cffi_modules = [
        "src/cajal/ugw/build_backends.py:build_single_core",
        "src/cajal/ugw/build_backends.py:build_multicore",
        # "src/cajal/ugw/build_backends.py:build_opencl",
        # "src/cajal/ugw/build_backends.py:build_cuda",
    ]

setup(
    ext_modules=ext_modules,
    compiler_directives={"language_level": "3"},
    include_dirs=include_path,
    cffi_modules=cffi_modules,
    cmdclass={"build": build},
)
