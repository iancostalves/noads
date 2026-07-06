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
"""Maturing aircraft technology parameter."""

from __future__ import annotations

from jax.numpy import array

from noads.core.models.interpolation import InterpolatedUnivariateSpline


class AircraftTechParameter:
    """A maturing component technology parameter (e.g. battery specific energy).

    The parameter value is interpolated with a second-order spline through its
    (2020, 2040, 2060) values, chosen per technology scenario (Lower/Mid/Upper), and
    evaluated at an aircraft's entry-into-service year during design. See the
    aircraft technology table of the extended paper for the values and sources.
    """

    name: str
    """Technology parameter name."""

    value_2020: float
    """Parameter value at 2020."""

    value_2040: float
    """Parameter value at 2040."""

    value_2060: float
    """Parameter value at 2060."""

    def __init__(self, name: str, values: tuple[float, float, float]):
        """Initialize AircraftTechParameter."""
        self.name = name
        self.value_2020 = values[0]
        self.value_2040 = values[1]
        self.value_2060 = values[2]

    def value_at_entry_into_service(self, entry_into_service):
        """Interpolate parameter value at Entry-Into-Service."""
        x_data = array([2020, 2040, 2060])
        y_data = array([self.value_2020, self.value_2040, self.value_2060])
        spline = InterpolatedUnivariateSpline(x_data, y_data, k=2)
        return spline(entry_into_service)
