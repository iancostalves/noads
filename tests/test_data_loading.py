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

"""The bundled data files must load regardless of the current working directory.

This is a regression test for the packaging bug that made ``pip install`` unusable:
the data reads used to be resolved against the CWD (``"../../../../src/noads/..."``),
so an installed package raised ``FileNotFoundError`` from anywhere but the repo root.
The reads now resolve package-relative (via :func:`noads._data.data_file`), so running
from an unrelated directory must still succeed.
"""

from noads.application.background_scenario_data import get_ar6_input_data
from noads.demand_calibration.calibration_utils import get_departures_data


def test_ar6_input_data_loads_from_arbitrary_cwd(tmp_path, monkeypatch):
    """AR6 background-scenario CSVs load when the CWD is unrelated to the package."""
    monkeypatch.chdir(tmp_path)
    data = get_ar6_input_data(plot_data=False)
    assert data is not None


def test_departures_data_loads_from_arbitrary_cwd(tmp_path, monkeypatch):
    """The demand-calibration Excel workbooks load from an unrelated CWD."""
    monkeypatch.chdir(tmp_path)
    years, gdp_pc, dep_pc = get_departures_data("WLD")
    assert len(years) > 0
    assert len(gdp_pc) == len(years)
    assert len(dep_pc) == len(years)
