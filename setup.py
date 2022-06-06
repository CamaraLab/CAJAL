from setuptools import setup

setup(
    name='cajal',    # This is the name of your PyPI-package.
    version='0.1',                          # python versioneer
    url="https://github.com/CamaraLab/CAJAL",
    author="Pablo Camara",
    author_email='pcamara@pennmedicine.upenn.edu',
    license='GPLv3',
    packages=["CAJAL", "CAJAL.lib"],
    python_requires=">=3.6",

    install_requires=[
        'networkx',
        'numpy',
        'pandas',
        'pot==0.7.0',
        'potpourri3d',
        'scipy',
        'scikit-image',
        'tifffile',
        'trimesh',
        'umap-learn'
     ],
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Framework :: Jupyter',
        'Topic :: Scientific/Engineering :: Bio-Informatics'
    ],
)
