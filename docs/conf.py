"""Sphinx configuration for xtrax documentation."""

import sys
from pathlib import Path

# Add source directory to path for autodoc
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Read version from importlib.metadata
try:
    from importlib.metadata import version

    project_version = version("xtrax")
except Exception:
    project_version = "0.2.0"

# Project information
project = "xtrax"
copyright = "2024, Marielle Russo"
author = "Marielle Russo"
release = project_version
version = project_version

# Sphinx extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

# MyST parser configuration
myst_enable_extensions = ["tasklist"]

# Napoleon configuration (Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_method = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc configuration
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_preserve_defaults = True

# Autosummary configuration
autosummary_generate = True
autosummary_generate_overwrite = True

# Mock heavy optional dependencies that may not be available during doc build
autodoc_mock_imports = []

# Intersphinx mapping
intersphinx_mapping = {
    "jax": ("https://jax.readthedocs.io/en/latest", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "python": ("https://docs.python.org/3", None),
}

# HTML theme
html_theme = "furo"
html_theme_options = {
    "sidebar_hide_name": False,
    "source_repository": "https://github.com/maraxen/xtrax",
    "source_branch": "main",
    "source_directory": "docs",
}

# Source file format
source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

# Master document
master_doc = "index"

# Directories to exclude
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Highlighting
pygments_style = "sphinx"

# HTML output options
html_static_path = []
html_logo = None
html_favicon = None

# Suppress warnings for missing intersphinx targets
nitpicky = False
