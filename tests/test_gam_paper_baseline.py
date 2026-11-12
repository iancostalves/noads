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
"""Guard the paper aircraft model: its designs must not change."""

import json

import pytest

from noads.application.base_objects import categories_mission

from .gam_paper_baseline import BASELINE_FILE
from .gam_paper_baseline import compute_baseline


@pytest.fixture(scope="module")
def current():
    return compute_baseline()


@pytest.mark.parametrize("category", list(categories_mission))
def test_paper_designs_unchanged(current, category):
    reference = json.loads(BASELINE_FILE.read_text())
    keys = [key for key in reference if key.startswith(f"{category}|")]
    assert keys
    for key in keys:
        assert current[key] == pytest.approx(reference[key], rel=1e-9), key
