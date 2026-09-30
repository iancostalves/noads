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
from jax.numpy import exp
from jax.numpy import log10

from noads.core.models.interpolation import InterpolatedUnivariateSpline


class AircraftTechParameter:
    """A maturing component technology parameter (e.g. battery specific energy).

    The parameter value is interpolated with a second-order spline through its
    (2020, 2040, 2060) values, chosen per technology scenario (Lower/Mid/Upper), and
    evaluated at an aircraft's entry-into-service year during design. See the
    aircraft technology table of the extended paper for the values and sources.

    With ``log_scale``, the spline is fitted to the base-10 logarithm of the values,
    which suits quantities growing geometrically (e.g. maximum power per
    propulsor). The interpolant is then monotonic on 2020-2060 if and only if
    ``log10(v2060) >= (4 log10(v2040) - log10(v2020)) / 3`` for increasing values.
    """

    name: str
    """Technology parameter name."""

    value_2020: float
    """Parameter value at 2020."""

    value_2040: float
    """Parameter value at 2040."""

    value_2060: float
    """Parameter value at 2060."""

    log_scale: bool
    """Whether to interpolate the logarithm of the values."""

    def __init__(
        self, name: str, values: tuple[float, float, float], log_scale: bool = False
    ):
        """Initialize AircraftTechParameter."""
        self.name = name
        self.log_scale = log_scale
        self.value_2020 = values[0]
        self.value_2040 = values[1]
        self.value_2060 = values[2]

    def value_at_entry_into_service(self, entry_into_service):
        """Interpolate parameter value at Entry-Into-Service."""
        x_data = array([2020, 2040, 2060])
        y_data = array([self.value_2020, self.value_2040, self.value_2060])
        if self.log_scale:
            spline = InterpolatedUnivariateSpline(x_data, log10(y_data), k=2)
            return 10.0 ** spline(entry_into_service)
        spline = InterpolatedUnivariateSpline(x_data, y_data, k=2)
        return spline(entry_into_service)


class LogisticTechParameter:
    """A maturing technology parameter following a logistic curve in time.

    ``value(t) = v_inf + (v_0 - v_inf) / (1 + exp((t - t_50) / tau))``, going from
    ``v_0`` long before ``t_50`` to ``v_inf`` long after, with half of the change
    reached at ``t_50`` and a transition time scale ``tau`` (years). It has the same
    interface as :class:`AircraftTechParameter`, and is monotonic in time.
    """

    name: str
    """Technology parameter name."""

    initial_value: float
    """Value long before the transition, v_0."""

    final_value: float
    """Value long after the transition, v_inf."""

    mid_year: float
    """Year of half transition, t_50."""

    time_scale: float
    """Transition time scale, tau (years)."""

    def __init__(self, name: str, values: tuple[float, float, float, float]):
        """Initialize from ``(v_0, v_inf, t_50, tau)``."""
        self.name = name
        (
            self.initial_value,
            self.final_value,
            self.mid_year,
            self.time_scale,
        ) = values

    def value_at_entry_into_service(self, entry_into_service):
        """Parameter value at Entry-Into-Service."""
        return self.final_value + (self.initial_value - self.final_value) / (
            1.0 + exp((entry_into_service - self.mid_year) / self.time_scale)
        )
