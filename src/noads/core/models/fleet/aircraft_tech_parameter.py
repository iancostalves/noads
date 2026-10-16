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

import numpy as np
from jax.numpy import array
from jax.numpy import exp
from jax.numpy import log10
from jax.numpy import maximum

from noads.core.models.interpolation import InterpolatedUnivariateSpline
from noads.core.models.interpolation import pchip_interpolate


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

    ``value(t) = v_inf + (v_0 - v_inf) * sigma(t) / sigma(t_0)``, with
    ``sigma(t) = 1 / (1 + exp((t - t_50) / tau))``, goes from ``v_0`` at the start
    year ``t_0`` (2020) towards ``v_inf``, with a transition centred on ``t_50``
    over a time scale ``tau`` (years). It is constant before ``t_0``, monotonic, and
    has the same interface as :class:`AircraftTechParameter`. Scenarios sharing
    ``v_0``, ``t_50`` and ``tau`` start from the same value in 2020 and then only
    diverge.
    """

    name: str
    """Technology parameter name."""

    initial_value: float
    """Value at the start year, v_0."""

    final_value: float
    """Value long after the transition, v_inf."""

    mid_year: float
    """Year of the transition centre, t_50."""

    time_scale: float
    """Transition time scale, tau (years)."""

    start_year: float
    """Year of the initial value, t_0."""

    def __init__(
        self,
        name: str,
        values: tuple[float, float, float, float],
        start_year: float = 2020.0,
    ):
        """Initialize from ``(v_0, v_inf, t_50, tau)``."""
        self.name = name
        (
            self.initial_value,
            self.final_value,
            self.mid_year,
            self.time_scale,
        ) = values
        self.start_year = start_year

    def _sigmoid(self, t):
        return 1.0 / (1.0 + exp((t - self.mid_year) / self.time_scale))

    def value_at_entry_into_service(self, entry_into_service):
        """Parameter value at Entry-Into-Service."""
        t = maximum(entry_into_service, self.start_year)
        return self.final_value + (self.initial_value - self.final_value) * (
            self._sigmoid(t) / self._sigmoid(self.start_year)
        )


SCENARIO_YEARS = (2020.0, 2040.0, 2060.0, 2080.0)
"""Years at which the scenario technology parameters are given."""


class ScenarioTechParameter:
    """A maturing technology parameter of one scenario, consistent with the others.

    The parameter is given at :data:`SCENARIO_YEARS` for the Lower, Mid and Upper
    technology scenarios. The Mid curve is a monotone cubic (PCHIP) interpolation of
    its values, and the Lower and Upper curves are the Mid curve plus a PCHIP
    interpolation of their gap to Mid. The scenarios share their 2020 value and
    their gaps to Mid never shrink, so that the Upper-to-Lower band starts at zero
    in 2020 and only widens, and the Mid curve always lies within it. Beyond 2080
    the parameters keep their 2080 value.

    With ``log_scale``, the construction applies to the base-10 logarithm of the
    values, which suits quantities growing geometrically (e.g. maximum power per
    propulsor).
    """

    name: str
    """Technology parameter name."""

    lower_mid_upper: tuple
    """Values of the Lower, Mid and Upper scenarios at the scenario years."""

    index: int
    """Scenario of this parameter (0: Lower, 1: Mid, 2: Upper)."""

    log_scale: bool
    """Whether to interpolate the logarithm of the values."""

    def __init__(self, name, lower_mid_upper, index, log_scale=False):
        """Initialize ScenarioTechParameter.

        Raises:
            ValueError: If the scenarios do not share their 2020 value, or if a gap
                to the Mid scenario shrinks or changes sign.
        """
        self.name = name
        self.lower_mid_upper = tuple(
            np.asarray(values, dtype=float) for values in lower_mid_upper
        )
        self.index = index
        self.log_scale = log_scale
        mid = self._transform(self.lower_mid_upper[1])
        for values in (self.lower_mid_upper[0], self.lower_mid_upper[2]):
            gap = self._transform(values) - mid
            increasing = np.diff(np.abs(gap)) >= -1e-12
            same_sign = np.all(gap >= -1e-12) or np.all(gap <= 1e-12)
            if abs(gap[0]) > 1e-12 or not (np.all(increasing) and same_sign):
                msg = (
                    f"{name}: the scenarios must share their 2020 value, and their "
                    f"gaps to the Mid scenario must not shrink, got {gap.tolist()}"
                )
                raise ValueError(msg)

    def _transform(self, values):
        return np.log10(values) if self.log_scale else values

    def value_at_entry_into_service(self, entry_into_service):
        """Parameter value at Entry-Into-Service."""
        mid = self._transform(self.lower_mid_upper[1])
        value = pchip_interpolate(entry_into_service, SCENARIO_YEARS, mid)
        if self.index != 1:
            gap = self._transform(self.lower_mid_upper[self.index]) - mid
            value = value + pchip_interpolate(entry_into_service, SCENARIO_YEARS, gap)
        return 10.0**value if self.log_scale else value
