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

from __future__ import annotations


class Stream:
    name: str
    unit: str

    def __init__(self, name: str, unit: str):
        self.name = name
        self.unit = unit


class Impact(Stream):
    budget: float

    def __init__(self, name, unit, budget: float):
        self.budget = budget
        super().__init__(name=name, unit=unit)


class MaterialStream(Stream):
    density: float
    unit = "kg"

    def __init__(self, name, density: float):
        self.density = density
        super().__init__(name=name, unit="kg")

    def volume_to_mass(self, volume):
        return volume * self.density

    def mass_to_volume(self, mass):
        return mass / self.density
