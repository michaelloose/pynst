"""Sphinx configuration for the PyNST documentation."""

from __future__ import annotations

import os
from pathlib import Path
import sys


# Make a source checkout importable for local documentation builds. Read the
# Docs additionally installs the package and all runtime dependencies normally.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import pynst  # noqa: E402


project = "PyNST"
author = "Michael Loose"
copyright = "2025-2026, Michael Loose"
version = pynst.__version__
release = pynst.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_nb",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", ".jupyter_cache", "Thumbs.db", ".DS_Store"]
language = "en"

nb_execution_mode = "force"
nb_execution_raise_on_error = True
nb_execution_timeout = 90

autodoc_member_order = "bysource"
autodoc_typehints = "signature"
autodoc_typehints_format = "short"
autodoc_preserve_defaults = True
autoclass_content = "both"

html_theme = "sphinx_rtd_theme"
html_title = f"PyNST {release} documentation"
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "/")
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_show_sourcelink = True
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
    "sticky_navigation": True,
    "titles_only": False,
}
html_context = {
    "display_github": True,
    "github_user": "michaelloose",
    "github_repo": "pynst",
    "github_version": "main",
    "conf_py_path": "/docs/",
}
