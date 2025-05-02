# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

project = "multisim"
copyright = "2025, Alex Lepauvre"
author = "Alex Lepauvre"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "numpydoc",
    "nbsphinx",
]

highlight_language = "python"
pygments_style = "sphinx"

templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_favicon = "favicon.ico"

html_theme_options = {
    "navigation_depth": 4,
    "show_nav_level": 2,
    "show_toc_level": 2,
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links"],
    "logo": {"image_light": "logo.svg"}
}
nbsphinx_prolog = """
.. raw:: html

    <style>
    /* Hide Notebook Titles from Sidebar */
    .toctree-wrapper > ul { display: none !important; }
    </style>
"""

# Fix potential warnings for notebooks
nbsphinx_allow_errors = True
