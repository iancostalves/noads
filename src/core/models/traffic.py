
from __future__ import annotations
from jax.numpy import divide
from jax.numpy import exp

from core.model import AutoModel


def generalised_logistic(
    x,
    left_asymptote=0.,
    capacity=0.00556283,
    growth_rate=0.00011641,
    logistic_nu=0.21666014,
    asymptote_coeff=1.17810123,
    x_lag=492.0691946,
):
    y = left_asymptote + divide(
        capacity - left_asymptote,
        (asymptote_coeff + exp(-growth_rate * (x - x_lag))) ** (1 / logistic_nu),
    )
    return y


class AirTraffic(AutoModel):

    def __init__(self):
        super().__init__(name="Air Traffic from Socioeconomic Drivers")

    def _jax_func(
        self,
        year=2025.0,
        gdp_per_capita=2.0e4,
        population=7742682218.0,
        gdp_per_capita_covid_end=1.5e4,
        load_factor_end_year=85.0,
        end_year=2050.0,
    ):
        # RPKpc logistic and total RPK
        gdp_per_capita_2019 = 8.77281e+13 / 7742682218.0
        covid_shift = gdp_per_capita_covid_end - gdp_per_capita_2019
        rpk_per_capita = generalised_logistic(
            gdp_per_capita,
            left_asymptote=0.,
            capacity=0.00556283,
            growth_rate=0.00011641,
            logistic_nu=0.21666014,
            asymptote_coeff=1.17810123,
            x_lag=492.0691946 + covid_shift,
        ) * 1e6
        rpk_trend = population * rpk_per_capita

        # load factor parameters
        load_factor_2019 = 82.39931216799707
        derivative = 2 * (-5.62003082e-05) * 31 + 3.59670410e-03
        a = (
                -(load_factor_end_year - load_factor_2019 - derivative * (
                        end_year - 2019
                )) / (end_year - 2019) ** 2
        )
        b = derivative - 2 * a * (end_year - 2019)
        load_factor = a * (year - 2019) ** 2 + b * (year - 2019) + load_factor_2019

        # ASK
        ask_trend = rpk_trend / (load_factor * 1e-2)

        return rpk_per_capita, rpk_trend, load_factor, ask_trend
