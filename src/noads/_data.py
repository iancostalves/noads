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

"""Locate data files bundled inside the installed package.

Data (AR6 scenario CSVs, demand-calibration workbooks) is shipped as package data, so
it must be resolved relative to the package rather than the current working directory,
otherwise an installed package cannot find it.
"""

from importlib.resources import files


def data_file(subpackage: str, *parts: str) -> str:
    """Return the path to a data file shipped inside ``subpackage``.

    Args:
        subpackage: The package holding the data directory, e.g.
            ``"noads.application"``.
        parts: The path components below that package, e.g.
            ``"ar6_scenarios_data", "population.csv"``.

    Returns:
        The filesystem path to the data file (as a string usable by pandas).
    """
    return str(files(subpackage).joinpath(*parts))
