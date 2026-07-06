# Copyright 2025 ISAE-SUPAERO, https://www.isae-supaero.fr/en/
# Copyright 2025 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""Fast checks that the documentation is self-contained and deployable.

These run before the (expensive) Sphinx build in CI and catch the classes of
problems that silently break the published docs: dangling citations, missing figure
assets, a broken bibliography, or absent pre-computed optimization results.
"""

import re
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"
BIB = DOCS / "references.bib"
RESULTS = DOCS / "examples" / "optimization" / "results"

pytestmark = pytest.mark.skipif(
    not DOCS.is_dir(), reason="documentation sources not available"
)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".svg", ".gif", ".pdf"}


def _bib_keys():
    """Return the list of BibTeX entry keys, in file order."""
    return re.findall(r"(?m)^@\w+\{\s*([^,\s]+)\s*,", BIB.read_text(encoding="utf-8"))


def _markdown_files():
    return sorted(p for p in DOCS.rglob("*.md") if "_build" not in p.parts)


def test_references_bib_has_no_duplicate_keys():
    """BibTeX keys must be unique, case-insensitively.

    Two entries whose keys differ only in case (e.g. ``stern_2007`` and
    ``Stern_2007``) make bibtex silently drop the *entire* bibliography, so every
    citation in the site then fails to resolve.
    """
    keys = _bib_keys()
    dupes = [k for k, n in Counter(k.lower() for k in keys).items() if n > 1]
    assert not dupes, f"duplicate BibTeX keys (case-insensitive): {sorted(dupes)}"


def test_cited_keys_are_defined():
    """Every ``{cite}`` key used in the docs must exist in references.bib."""
    defined = {k.lower() for k in _bib_keys()}
    missing = {}
    pattern = re.compile(r"\{cite(?::[a-z]+)?\}`([^`]+)`")
    for md in _markdown_files():
        for group in pattern.findall(md.read_text(encoding="utf-8")):
            for key in (k.strip() for k in group.split(",")):
                if key and key.lower() not in defined:
                    missing.setdefault(key, md.relative_to(DOCS).as_posix())
    assert not missing, f"undefined citation keys: {missing}"


def test_referenced_figures_exist():
    """Every committed image referenced from a doc page must exist on disk."""
    figure_re = re.compile(r"\{(?:figure|image)\}\s+(\S+)")
    markdown_re = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")
    missing = []
    for md in _markdown_files():
        text = md.read_text(encoding="utf-8")
        refs = figure_re.findall(text) + markdown_re.findall(text)
        for ref in refs:
            if ref.startswith(("http://", "https://", "#")):
                continue
            # Gallery images are generated at build time, not committed.
            if "/gallery/" in ref or ref.startswith("gallery/"):
                continue
            if Path(ref).suffix.lower() not in _IMAGE_EXTS:
                continue
            target = (md.parent / ref).resolve()
            if not target.is_file():
                missing.append(f"{md.relative_to(DOCS).as_posix()} -> {ref}")
    assert not missing, "referenced figures not found:\n" + "\n".join(missing)


def test_precomputed_optima_present():
    """The gallery loads pre-computed optima instead of re-running optimizations.

    Without them the comparison examples would attempt hours-long optimizations at
    build time (or fail), so the shared results directory must be populated.
    """
    assert RESULTS.is_dir(), f"missing results directory: {RESULTS}"
    optima = list(RESULTS.glob("*/opt_result.json"))
    assert len(optima) >= 27, f"expected >= 27 pre-computed optima, found {len(optima)}"

    # The scenario families exercised by the comparison scripts must be present.
    required = [
        "SSP1-19-Fossil-midTech",
        "SSP2-26-Fossil-midTech",
        "SSP5-45-Fossil-midTech",
        "SSP2-26-DropIn-midTech",
        "SSP2-26-midTech",  # breakthrough trend
        "robust-SSP2-midTech",
        "robust-SSP2-lowdemand-LowDemand-midTech",
    ]
    for name in required:
        assert (RESULTS / name / "opt_result.json").is_file(), (
            f"missing pre-computed optimum: {name}"
        )
