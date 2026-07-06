# Copyright 2025 ISAE-SUPAERO, https://www.isae-supaero.fr/en/
# Copyright 2025 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
# International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Sphinx configuration for the NOADS documentation."""

import os
import sys
from importlib.metadata import version as pkg_version
from pathlib import Path

from sphinx_gallery.sorting import ExplicitOrder

DOCS_DIR = Path(__file__).parent
sys.path.insert(0, str(DOCS_DIR))

# -- Project information -----------------------------------------------------

project = "NOADS"
author = "Ian Costa-Alves"
copyright = "2025, ISAE-SUPAERO and IRT Saint Exupéry"  # noqa: A001
try:
    release = pkg_version("noads")
except Exception:  # noqa: BLE001
    release = "dev"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_gallery.gen_gallery",
    "sphinxcontrib.bibtex",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
]

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "paper/latex_src",
    # Gallery source scripts and their README headers are consumed by
    # sphinx-gallery, not by the regular Sphinx reader.
    "examples",
]

numfig = True
math_numfig = True
math_eqref_format = "Eq. {number}"

templates_path = ["_templates"]

# -- MyST --------------------------------------------------------------------

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
    "deflist",
    "fieldlist",
    "attrs_inline",
    "substitution",
]
myst_dmath_double_inline = True
myst_heading_anchors = 3

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_book_theme"
html_title = "NOADS"
html_logo = "_static/partner_logos.png"
html_favicon = "_static/favicon.png"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_theme_options = {
    "repository_url": "https://github.com/iancostalves/noads",
    "repository_provider": "github",
    "use_repository_button": True,
    "use_download_button": False,
    "show_toc_level": 2,
    "show_navbar_depth": 1,
    "home_page_in_toc": True,
}

# -- Autodoc / autosummary ---------------------------------------------------

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "show-inheritance": True,
    "undoc-members": True,
}
autodoc_member_order = "groupwise"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "gemseo": ("https://gemseo.readthedocs.io/en/stable", None),
    "jax": ("https://docs.jax.dev/en/latest", None),
}

# -- Bibliography ------------------------------------------------------------

bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "unsrt"

# -- Sphinx-gallery ----------------------------------------------------------

# The optimization examples load pre-computed optima instead of re-running the
# (expensive) optimizations: point the results loader at the shared data
# directory shipped with the repository.
os.environ["NOADS_RESULTS_DIR"] = str(
    DOCS_DIR / "examples" / "optimization" / "results"
)
os.environ.setdefault("MPLBACKEND", "Agg")


sphinx_gallery_conf = {
    "examples_dirs": ["examples/models", "examples/optimization"],
    "gallery_dirs": ["gallery/models", "gallery/optimization"],
    # Only plot_*.py scripts are executed at build time; run_*.py scripts
    # (full optimizations, hours of compute) are rendered without execution.
    "filename_pattern": r"[\\/]plot_",
    "example_extensions": {".py"},
    "subsection_order": ExplicitOrder([
        "examples/models/demand",
        "examples/models/aircraft",
        "examples/models/energy",
        "examples/optimization/baseline",
        "examples/optimization/dropin",
        "examples/optimization/breakthrough",
        "examples/optimization/robust_policy",
        "examples/optimization/ar6_comparison",
    ]),
    "within_subsection_order": "sphinx_gallery_sort.RunScriptsFirstKey",
    "download_all_examples": False,
    "remove_config_comments": True,
    "reference_url": {"noads": None},
    # Required for the minigallery directives of the paper pages.
    "backreferences_dir": "gallery/backreferences",
    "doc_module": ("noads",),
}

suppress_warnings = ["mystnb.unknown_mime_type"]
