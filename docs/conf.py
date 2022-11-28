# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sys


project = 'cajal'
copyright = '2022, Pablo Camara'
author = 'Camara Lab'

# This release should be updated manually.
release = '0.10'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# The specific version of the sphinx_rtd_theme is specified in requirements.txt, as is the version of sphinx itself.
extensions = ['myst_parser',
              'sphinx_rtd_theme',
              'sphinx.ext.autodoc'
              ]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'

sys.path.append('../CAJAL/lib')
