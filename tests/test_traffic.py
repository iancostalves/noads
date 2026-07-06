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

import numpy as np
import pytest

from noads.core.models.traffic import AirTraffic
from noads.core.models.traffic import generalised_logistic


def test_generalised_logistic():
    x = np.linspace(0.0, 2.0e5, 50)
    y = np.asarray(generalised_logistic(x))
    assert np.all(np.diff(y) > 0.0)
    assert np.all(y > 0.0)
    assert np.all(y < 0.01118996)

    # Value at x = 0 from the closed-form expression.
    expected = 0.01118996 / (1.27603214 + 1.0) ** (1.0 / 0.18720497)
    assert float(generalised_logistic(0.0)) == pytest.approx(expected, rel=1e-12)

    # Asymptotes.
    assert float(generalised_logistic(1.0e7)) == pytest.approx(
        0.01118996 / 1.27603214 ** (1.0 / 0.18720497), rel=1e-6
    )
    assert float(generalised_logistic(-1.0e7, left_asymptote=0.5)) == pytest.approx(
        0.5, rel=1e-6
    )


def test_air_traffic_model(execute_model):
    traffic = AirTraffic()
    assert set(traffic.discipline.input_grammar.names) == {
        "year",
        "gdp_per_capita",
        "population",
        "gdp_per_capita_2019",
        "gdp_per_capita_covid_end",
        "load_factor_end_year",
        "end_year",
        "capacity_factor",
    }
    assert set(traffic.discipline.output_grammar.names) == {
        "rpk_per_capita",
        "rpk_trend",
        "load_factor",
        "ask_trend",
    }

    output_data = execute_model(traffic)
    for name in traffic.discipline.output_grammar.names:
        assert np.all(np.isfinite(output_data[name]))
        assert np.all(output_data[name] > 0.0)

    population = traffic.discipline.default_input_data["population"]
    assert output_data["rpk_trend"] == pytest.approx(
        population * output_data["rpk_per_capita"], rel=1e-12
    )
    assert output_data["ask_trend"] == pytest.approx(
        output_data["rpk_trend"] / (output_data["load_factor"] * 1e-2), rel=1e-12
    )

    # The load factor is a quadratic anchored at 2019 and at the end year.
    at_2019 = execute_model(traffic, {"year": 2019.0})
    assert at_2019["load_factor"] == pytest.approx(82.39931216799707, rel=1e-12)
    at_end = execute_model(
        traffic, {"year": 2050.0, "end_year": 2050.0, "load_factor_end_year": 90.0}
    )
    assert at_end["load_factor"] == pytest.approx(90.0, rel=1e-12)

    # A realistic population makes the finite-difference step vanish against
    # the output magnitude, so the jacobian is checked at a scaled population.
    input_data = dict(traffic.discipline.default_input_data)
    input_data["population"] = 100.0
    assert traffic.discipline.check_jacobian(input_data, threshold=1e-3)
