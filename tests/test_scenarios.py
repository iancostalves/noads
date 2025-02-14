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

import pytest
from gemseo_jax.jax_discipline import DataType
from numpy import array

from noads.core.model import AutoModel
from noads.core.model import JAXModel
from noads.core.scenarios.multiscenario import MultiScenario
from noads.core.scenarios.temporalscenario import TemporalScenario


class TestAutoModel(AutoModel):
    def __init__(self):
        super().__init__("auto")

    def _jax_func(self, a=0.0, b=1.0, c=2.0, d=3.0) -> float:
        return 2 * a + b - c * d


def compute_yz(input_data: DataType) -> DataType:
    a = input_data["a"]
    b = input_data["b"]
    c = input_data["c"]
    d = input_data["d"]

    y = a * b
    z = c * d
    return {"y": y, "z": z}


class TestJAXModel(JAXModel):
    def __init__(self):
        super().__init__(
            function=compute_yz,
            input_names=["a", "b", "c", "d"],
            output_names=["y", "z"],
            default_inputs={"a": 0.0, "b": 1.0, "c": 2.0, "d": 3.0},
            name="jax",
        )


@pytest.fixture(scope="module")
def models():
    return [TestAutoModel(), TestJAXModel()]


@pytest.fixture(scope="module")
def interpolated_inputs():
    return ["b"]


@pytest.fixture(scope="module")
def constant_inputs():
    return ["c"]


@pytest.fixture(scope="module")
def delayed_inputs():
    return ["d"]


@pytest.fixture(scope="module")
def time_integrated_outputs():
    return ["x", "z"]


@pytest.fixture(scope="module")
def start_year():
    return 0.0


@pytest.mark.parametrize(
    ("delay_time", "interpolated_time", "end_year", "time_step", "n_inputs"),
    [
        (1.0, array([0.0, 5.0, 10.0]), 10.0, 1.0, 5),
        (1.0, array([0.0, 5.0]), 10.0, 1.0, 5),
        (1.0, array([0.0, 5.0, 10.0]), 15.0, 1.0, 5),
        (1.0, array([0.0, 5.0, 10.0]), 10.0, 2.0, 5),
    ],
)
def test_temporal_scenario(
    models,
    constant_inputs,
    interpolated_inputs,
    delayed_inputs,
    time_integrated_outputs,
    start_year,
    end_year,
    time_step,
    delay_time,
    interpolated_time,
    n_inputs,
):
    scenario = TemporalScenario(
        "test",
        models,
        constant_inputs,
        delayed_inputs,
        delay_time,
        interpolated_inputs,
        interpolated_time,
        time_integrated_outputs,
        start_year,
        end_year,
        time_step,
    )

    assert scenario.discipline is not None
    for n in range(n_inputs):
        input_data = {
            name: n + n * value
            for name, value in scenario.discipline.default_input_data.items()
        }
        assert scenario.discipline.check_jacobian(input_data, threshold=1e-4)

    scenario.discipline.add_differentiated_inputs(["constant.c", "control.d"])
    scenario.discipline.add_differentiated_outputs(["cumulative.x", "cumulative.z"])
    scenario.discipline.compile_jit(pre_run=True)

    for n in range(n_inputs):
        input_data = {
            name: n + n * value
            for name, value in scenario.discipline.default_input_data.items()
        }
        assert scenario.discipline.check_jacobian(
            input_data,
            threshold=1e-4,
            # inputs=["constant.c", "control.d"],
            # outputs=["cumulative.x", "cumulative.z"],
        )


@pytest.fixture(scope="module")
def mean_outputs():
    return ["cumulative.x", "cumulative.z"]


@pytest.fixture(scope="module")
def scenario_names():
    return ["A1", "A2"]


@pytest.fixture(scope="module")
def fixed_inputs():
    return ["control.d"]


@pytest.mark.parametrize(
    ("delay_time", "interpolated_time", "end_year", "time_step", "n_inputs"),
    [
        (1.0, array([0.0, 5.0, 10.0]), 10.0, 1.0, 5),
        (1.0, array([0.0, 5.0]), 10.0, 1.0, 5),
        (1.0, array([0.0, 5.0, 10.0]), 15.0, 1.0, 5),
        (1.0, array([0.0, 5.0, 10.0]), 10.0, 2.0, 5),
    ],
)
def test_multi_scenario(
    mean_outputs,
    scenario_names,
    fixed_inputs,
    models,
    constant_inputs,
    interpolated_inputs,
    delayed_inputs,
    time_integrated_outputs,
    start_year,
    end_year,
    time_step,
    delay_time,
    interpolated_time,
    n_inputs,
):
    temporal_scenario = TemporalScenario(
        "test",
        models,
        constant_inputs,
        delayed_inputs,
        delay_time,
        interpolated_inputs,
        interpolated_time,
        time_integrated_outputs,
        start_year,
        end_year,
        time_step,
    )

    scenario = MultiScenario(
        temporal_scenario, mean_outputs, scenario_names, fixed_inputs
    )
    assert scenario.discipline is not None
    for n in range(n_inputs):
        input_data = {
            name: n + n * value
            for name, value in scenario.discipline.default_input_data.items()
        }
        assert scenario.discipline.check_jacobian(input_data, threshold=1e-4)

    scenario.discipline.add_differentiated_inputs([
        "A1.constant.c",
        "A2.constant.c",
        "fixed.control.d",
    ])
    scenario.discipline.add_differentiated_outputs([
        "mean.cumulative.x",
        "mean.cumulative.z",
    ])
    scenario.discipline.compile_jit(pre_run=True)

    for n in range(n_inputs):
        input_data = {
            name: n + n * value
            for name, value in scenario.discipline.default_input_data.items()
        }
        assert scenario.discipline.check_jacobian(
            input_data,
            threshold=1e-4,
            # inputs=["A1.constant.c", "A2.constant.c", "fixed.control.d"],
            # outputs=["mean.cumulative.x", "mean.cumulative.z"],
        )
