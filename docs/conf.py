# -*- coding: utf-8 -*-
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from __future__ import annotations

import sys

project = 'cajal'
# This looks weird to me but it renders correctly in the HTML.
copyright = '2022, Pablo Cámara'
author = 'Cámara Lab'

# This release should be updated manually.
release = '0.10'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# The specific version of the sphinx_rtd_theme is specified in requirements.txt, as is the version of sphinx itself.
extensions = ['myst_parser',
              'sphinx_rtd_theme',
              'sphinx.ext.autodoc',
              'sphinx.ext.autosectionlabel',
              'sphinx.ext.intersphinx'
              ]
autodoc_typehints = 'both'
autodoc_type_aliases = { 'VertexArray' : 'sample_mesh.VertexArray',
                         'FaceArray' : 'sample_mesh.FaceArray',
                         'WeightedTree' : 'sample_swc.WeightedTree'
                        }

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'

sys.path.insert(0,"../src/")

html_static_path = ['_static']
html_css_files = [
    'css/custom.css',
]

# -- Intersphinx --
intersphinx_mapping = { 'scipy' : ('https://docs.scipy.org/doc/scipy/reference/', None),
                        'numpy' : ('https://numpy.org/doc/stable/', None),
                        'python' : ('https://docs.python.org/3.10/', None),
                        'pandas' : ('https://pandas.pydata.org/docs/', None)
                       }

# Type Aliases

