from setuptools import setup

setup(
    name='cajal',    # This is the name of your PyPI-package.
    version='1.0',                          # python versioneer
    url="https://github.com/CamaraLab/CAJAL",
    author="Pablo Camara",
    author_email='pcamara@pennmedicine.upenn.edu',
    license='GPLv3',
    packages=["CAJAL","CAJAL.scripts","CAJAL.lib"],
    python_requires=">=3.6",

    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'pot==0.7.0'
     ],
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Framework :: Jupyter',
        'Topic :: Scientific/Engineering :: Bio-Informatics'
    ],
)
