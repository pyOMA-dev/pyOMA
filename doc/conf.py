# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../'))


# -- Project information -----------------------------------------------------

project = 'pyOMA'
copyright = '2021, Simon Marwitz, Volkmar Zabel, Andrei Udrea'
author = 'Simon Marwitz, Volkmar Zabel, Andrei Udrea'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['autoclasstoc', 'sphinx.ext.autodoc','sphinx.ext.autosummary', 
              'sphinx.ext.napoleon', 'sphinx.ext.viewcode', 'sphinx.ext.mathjax',
              'sphinx.ext.todo','sphinxcontrib.blockdiag', 'myst_nb',"sphinxcontrib.collections",]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

# Build the documentation including the Todos
todo_include_todos=True

# 
autodoc_default_flags = ['members']
autosummary_generate = True

# Do not execute ipynb Notebooks, but rather display saved outputs from the last manual run
nb_execution_mode = "off"

collections = {
   'single_setup_analysis.ipynb': {
      'driver': 'copy_file',
      'source': '../scripts/single_setup_analysis.ipynb',
      'final_clean': False
   }
}
