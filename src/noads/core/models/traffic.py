# Copyright 2025 ISAE-SUPAERO, https://www.isae-supaero.fr/en/
# Copyright 2025 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""Air Traffic modelling."""

from __future__ import annotations

from jax.numpy import divide
from jax.numpy import exp

from noads.core.model import AutoModel


def generalised_logistic(
    x,
    left_asymptote=0.0,
    capacity=0.01118996,
    growth_rate=0.00010489,
    logistic_nu=0.18720497,
    asymptote_coeff=1.27603214,
    x_lag=0.0,
):
    """Generalized logistic (Richards) function.

    Used to model the saturation of per-capita air traffic demand with rising
    per-capita income; see the demand model of the extended paper for the
    calibrated parameters and their interpretation.
    """
    y = left_asymptote + divide(
        capacity - left_asymptote,
        (asymptote_coeff + exp(-growth_rate * (x - x_lag))) ** (1.0 / logistic_nu),
    )
    return y  # noqa: RET504


class AirTraffic(AutoModel):
    """The air traffic demand model.

    Computes trend RPK from population and per-capita income through the calibrated
    generalized logistic function, applies the SSP storyline multiplier and the
    COVID-recovery adjustment, converts demand into supply (ASK) with the
    time-quadratic load factor, and applies the market supply-shift ratios of the
    low-demand formulation, together with the associated relative ticket price
    change and its discounted burden.
    """

    def __init__(self):
        """Initialize AirTraffic."""
        super().__init__(name="Air Traffic from Socioeconomic Drivers")

    def _jax_func(
        self,
        year=2025.0,
        gdp_per_capita=2.0e4,
        population=7742682218.0,
        gdp_per_capita_2019=1.5e4,
        gdp_per_capita_covid_end=1.5e4,
        load_factor_end_year=85.0,
        end_year=2050.0,
        capacity_factor=1.0,
    ):
        # RPKpc logistic and total RPK
        gdp_per_capita_2019_mean = 0.5 * (
            gdp_per_capita_2019 + 8.77281e13 / 7742682218.0
        )
        covid_shift = gdp_per_capita_covid_end - gdp_per_capita_2019_mean
        rpk_per_capita = 1.0e6 * (
            generalised_logistic(  # Logistic calibration
                gdp_per_capita,
                left_asymptote=0.0,
                capacity=0.01118996,
                growth_rate=0.00010489,
                logistic_nu=0.18720497,
                asymptote_coeff=1.27603214,
                x_lag=0.0 + covid_shift,
                # generalised_logistic(  # Logistic calibration + price effect
                #     gdp_per_capita,
                #     left_asymptote=0.0,
                #     capacity=0.00889545,
                #     growth_rate=0.00011641,
                #     logistic_nu=0.18192608,
                #     asymptote_coeff=1.16370027,
                #     x_lag=0.0 + covid_shift,
            )
            * generalised_logistic(
                gdp_per_capita,
                left_asymptote=3.0 / (2 * capacity_factor + 1.0),
                capacity=capacity_factor,
                growth_rate=0.5 / gdp_per_capita_covid_end,
                logistic_nu=1.0,
                asymptote_coeff=1.0,
                x_lag=2.0 * gdp_per_capita_covid_end,
            )
        )
        rpk_trend = population * rpk_per_capita

        # load factor parameters
        load_factor_2019 = 82.39931216799707
        derivative = 2 * (-5.62003082e-05) * 31 + 3.59670410e-03
        a = (
            -(load_factor_end_year - load_factor_2019 - derivative * (end_year - 2019))
            / (end_year - 2019) ** 2
        )
        b = derivative - 2 * a * (end_year - 2019)
        load_factor = a * (year - 2019) ** 2 + b * (year - 2019) + load_factor_2019

        # ASK
        ask_trend = rpk_trend / (load_factor * 1e-2)

        return rpk_per_capita, rpk_trend, load_factor, ask_trend
