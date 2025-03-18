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
"""Elementary Streams."""

from __future__ import annotations


class Stream:  # noqa: B903
    """Basic flow of a generic stream."""

    name: str
    unit: str

    def __init__(self, name: str, unit: str):
        """Initialize Stream from name and unit."""
        self.name = name
        self.unit = unit


class Impact(Stream):  # noqa: B903
    """Basic flow of impacts (CO2, land, euro, ...)."""

    budget: float

    def __init__(self, name, unit, budget: float):
        """Initialize Impact from name, unit and budget."""
        self.budget = budget
        super().__init__(name=name, unit=unit)


class MaterialStream(Stream):
    """Basic flow of material."""

    density: float
    unit = "kg"

    def __init__(self, name, density: float):
        """Initialize MaterialStream from name and density."""
        self.density = density
        super().__init__(name=name, unit="kg")

    def volume_to_mass(self, volume):
        """Convert volume to mass."""
        return volume * self.density

    def mass_to_volume(self, mass):
        """Convert mass to volume."""
        return mass / self.density
