from setuptools import setup, Extension
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
        [
            Extension(
                "libEMD_wrapper",
                ["src/cajal/EMD_wrapper.cpp"],
                language="c++",
            ),
            Extension(
                "patstest",
                ["src/cajal/patstest.pyx"],
                libraries=["EMD_wrapper.cpython-310-x86_64-linux-gnu"],
                library_dirs=[
                    "/home/patn/.local/lib/python3.10/site-packages",
                    "/home/patn/CAJAL/build/lib.linux-x86_64-cpython-310",
                ],
                runtime_library_dirs=[
                    "/home/patn/.local/lib/python3.10/site-packages",
                    "/home/patn/CAJAL/build/lib.linux-x86_64-cpython-310",
                ],
                language="c++",
            ),
            # Extension("gw_cython",
            # [ "src/cajal/gw_cython.pyx" ],
            # libraries=["EMD_wrapper.cpython-310-x86_64-linux-gnu"],
            # # library_dirs=['/home/patn/.local/lib/python3.10/site-packages'],
            # # runtime_library_dirs=['/home/patn/.local/lib/python3.10/site-packages'],
            # language='c++',
            #           )
        ]
    ),
    include_dirs=[numpy.get_include(), os.path.join(ROOT, "CAJAL/src/cajal")],
)

# setup(
#     ext_modules=cythonize(
#         [
#         # Extension(
#         #     "EMD_wrapper",
#         #     ["src/cajal/EMD_wrapper.cpp"]
#         # ),
#          Extension(
#             "gw_cython",
#             ["src/cajal/EMD_wrapper.cpp","src/cajal/gw_cython.pyx"],
#              libraries=["EMD_wrapper.cpython-310-x86_64-linux-gnu.so"],
#              library_dirs=[
#                            'build/lib.linux-x86_64-cpython-310',
#                            ],
#             # runtime_library_dirs=['/home/patn/.local/lib/python3.10/site-packages/']
#         ),
#           # Extension(
#           #   "test",
#           #   ["src/cajal/EMD_wrapper.cpp","src/cajal/test.pyx"],
#           #     libraries=["EMD_wrapper.cpython-310-x86_64-linux-gnu.so"],
#           #     library_dirs=['/home/patn/.local/lib/python3.10/site-packages/',
#           #                   '/home/patn/CAJAL/build/lib.linux-x86_64-cpython-310',
#           #                   ],
#           #     runtime_library_dirs=['/home/patn/.local/lib/python3.10/site-packages/cajal',
#           #                           '/home/patn/CAJAL/build/lib.linux-x86_64-cpython-310'
#           #                           ],
#           #     language="c++"
#           # )
#          ],
#         gdb_debug=True,
#         compiler_directives={"language_level": "3"},
#         language="c++",

#         # extra_link_args=["-Wl,-rpath=$ORIGIN/."]
#     ),
#     include_dirs=[numpy.get_include(),os.path.join(ROOT, 'CAJAL/src/cajal')]
# )


# setup(
#     ext_modules=
#     [   cythonize(
#         [ Extension("EMD_wrapper",["src/cajal/EMD_wrapper.cpp"]),
#           Extension("gw_cython",sources=["src/cajal/*.pyx"],libraries=["/home/patn/.local/lib/python3.10/site-packages/EMD_wrapper.cpython-310-x86_64-linux-gnu.so"],runtime_library_dirs=["/home/patn/.local/lib/python3.10/site-packages/"])],

#         )
#     ],
#     include_dirs=[numpy.get_include(),os.path.join(ROOT, 'CAJAL/src/cajal')],
# )
