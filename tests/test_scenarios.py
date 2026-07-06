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
from gemseo_jax.jax_discipline import DataType
from gemseo_jax.jax_discipline import JAXDiscipline
from jax.numpy import array as jnp_array

from noads.core.model import AutoModel
from noads.core.model import JAXModel
from noads.core.models.interpolation import interpolate_data
from noads.core.scenarios.multiscenario import MultiScenario
from noads.core.scenarios.temporalscenario import TemporalScenario


class TestAutoModel(AutoModel):
    def __init__(self):
        super().__init__("auto")

    def _jax_func(self, a=0.0, b=1.0, c=2.0, d=3.0) -> float:
        x = 2 * a + b - c * d
        return x  # noqa: RET504


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


# @pytest.fixture(scope="module")
# def delayed_inputs():
#     return ["d"]


@pytest.fixture(scope="module")
def time_integrated_outputs():
    return ["x", "z"]


@pytest.fixture(scope="module")
def start_year():
    return 0.0


@pytest.mark.parametrize(
    ("delay_times", "interp_step", "end_year", "time_step", "n_inputs"),
    [
        ({"d": 1.0}, 1.0, 10.0, 1.0, 5),
        ({"d": 1.0}, 2.0, 10.0, 1.0, 5),
        ({"d": 1.0}, 5.0, 15.0, 1.0, 5),
        ({"d": 1.0}, 5.0, 10.0, 2.0, 5),
    ],
)
def test_temporal_scenario(
    models,
    constant_inputs,
    interpolated_inputs,
    time_integrated_outputs,
    start_year,
    end_year,
    time_step,
    delay_times,
    interp_step,
    n_inputs,
):
    scenario = TemporalScenario(
        name="test",
        models=models,
        constant_inputs=constant_inputs,
        control_delay_times=delay_times,
        constrained_control_groups={},
        custom_controls=[],
        final_rates=[],
        interpolated_inputs=interpolated_inputs,
        interp_step=interp_step,
        time_integrated_outputs=time_integrated_outputs,
        start_year=start_year,
        end_year=end_year,
        time_step=time_step,
    )

    assert scenario.discipline is not None
    for n in range(n_inputs):
        input_data = {
            name: n + n * value
            for name, value in scenario.discipline.default_input_data.items()
        }
        assert scenario.discipline.check_jacobian(input_data, threshold=1e-3)

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
            threshold=1e-3,
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
    ("delay_times", "interp_step", "end_year", "time_step", "n_inputs"),
    [
        ({"d": 1.0}, 1.0, 10.0, 1.0, 5),
        ({"d": 1.0}, 2.0, 10.0, 1.0, 5),
        ({"d": 1.0}, 5.0, 15.0, 1.0, 5),
        ({"d": 1.0}, 5.0, 10.0, 2.0, 5),
    ],
)
def test_multi_scenario(
    mean_outputs,
    scenario_names,
    fixed_inputs,
    models,
    constant_inputs,
    interpolated_inputs,
    time_integrated_outputs,
    start_year,
    end_year,
    time_step,
    delay_times,
    interp_step,
    n_inputs,
):
    temporal_scenario = TemporalScenario(
        name="test",
        models=models,
        constant_inputs=constant_inputs,
        control_delay_times=delay_times,
        constrained_control_groups={},
        custom_controls=[],
        final_rates=[],
        interpolated_inputs=interpolated_inputs,
        interp_step=interp_step,
        time_integrated_outputs=time_integrated_outputs,
        start_year=start_year,
        end_year=end_year,
        time_step=time_step,
    )

    scenario = MultiScenario(
        "multi", temporal_scenario, mean_outputs, scenario_names, fixed_inputs
    )
    assert scenario.discipline is not None
    for n in range(n_inputs):
        input_data = {
            name: n + n * value
            for name, value in scenario.discipline.default_input_data.items()
        }
        assert scenario.discipline.check_jacobian(input_data, threshold=1e-3)

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
            threshold=1e-3,
            # inputs=["A1.constant.c", "A2.constant.c", "fixed.control.d"],
            # outputs=["mean.cumulative.x", "mean.cumulative.z"],
        )


def build_scenario(models, **overrides):
    """Build a TemporalScenario with a small canonical configuration."""
    options = {
        "name": "feature",
        "models": models,
        "constant_inputs": ["c"],
        "control_delay_times": {"d": 1.0},
        "constrained_control_groups": {},
        "custom_controls": [],
        "final_rates": [],
        "interpolated_inputs": ["b"],
        "time_integrated_outputs": ["x", "z"],
        "start_year": 0.0,
        "end_year": 10.0,
        "time_step": 1.0,
        "interp_step": 5.0,
    }
    options.update(overrides)
    return TemporalScenario(**options)


def test_invalid_differentiation_methods(models):
    with pytest.raises(ValueError, match="forward or reverse"):
        build_scenario(
            models,
            differentiation_method=JAXDiscipline.DifferentiationMethod.AUTO,
        )
    for adjoint_method in (
        TemporalScenario.AdjointMethod.RECURSIVE,
        TemporalScenario.AdjointMethod.BACKSOLVE,
    ):
        with pytest.raises(ValueError, match="not compatible"):
            build_scenario(
                models,
                differentiation_method=JAXDiscipline.DifferentiationMethod.FORWARD,
                adjoint_method=adjoint_method,
            )

    temporal_scenario = build_scenario(models)
    with pytest.raises(ValueError, match="forward or reverse"):
        MultiScenario(
            "multi",
            temporal_scenario,
            ["cumulative.x"],
            ["A1"],
            [],
            differentiation_method=JAXDiscipline.DifferentiationMethod.AUTO,
        )


@pytest.mark.parametrize(
    ("adjoint_method", "differentiation_method"),
    [
        (
            TemporalScenario.AdjointMethod.DIRECT,
            JAXDiscipline.DifferentiationMethod.FORWARD,
        ),
        (
            TemporalScenario.AdjointMethod.BACKSOLVE,
            JAXDiscipline.DifferentiationMethod.REVERSE,
        ),
    ],
)
def test_adjoint_methods(models, adjoint_method, differentiation_method):
    baseline = build_scenario(models).discipline.execute()
    scenario = build_scenario(
        models,
        adjoint_method=adjoint_method,
        differentiation_method=differentiation_method,
    )
    output_data = scenario.discipline.execute()
    for name in ("cumulative.x", "cumulative.z"):
        np.testing.assert_allclose(output_data[name], baseline[name], rtol=1e-6)


def test_cubic_interpolation(models):
    scenario = build_scenario(models, interp_step=2.5, cubic_interpolation=True)
    control_values = np.linspace(0.0, 1.0, scenario.time_interpolation.size)
    output_data = scenario.discipline.execute({"control.d": control_values})

    expected = interpolate_data(
        scenario.time_vector,
        scenario.time_interpolation,
        control_values,
        cubic=True,
    )
    np.testing.assert_allclose(
        output_data["interpolate.control.d"], expected, atol=1e-10
    )
    assert output_data["b"].shape == scenario.time_vector.shape


def test_constrained_control_groups(models):
    scenario = build_scenario(models, constrained_control_groups={"grp": ["d"]})

    # The default control.d is 3.0, so the sum-to-one constraint is exceeded.
    output_data = scenario.discipline.execute()
    np.testing.assert_allclose(output_data["grp.controls_constraint"], 2.0)
    np.testing.assert_allclose(output_data["grp.controls_constraint_violation"], 2.0)
    assert output_data["grp.controls_constraint"].shape == scenario.time_vector.shape

    below = scenario.discipline.execute({
        "control.d": 0.4 * np.ones(scenario.time_interpolation.shape)
    })
    np.testing.assert_allclose(below["grp.controls_constraint"], -0.6)
    np.testing.assert_allclose(below["grp.controls_constraint_violation"], 0.0)


def control_producer(input_data: DataType) -> DataType:
    e = input_data["e"]
    f = input_data["f"]
    return {
        "control.d.times": e * jnp_array([0.0, 3.0, 7.0, 10.0]),
        "control.d.values": f * jnp_array([0.0, 2.0, 2.0, 0.0]),
    }


def test_custom_controls(models):
    producer = JAXModel(
        function=control_producer,
        input_names=["e", "f"],
        output_names=["control.d.times", "control.d.values"],
        default_inputs={"e": 1.0, "f": 1.0},
        name="producer",
    )
    scenario = build_scenario(
        [*models, producer],
        constant_inputs=["c", "e", "f"],
        custom_controls=["d"],
        constrained_control_groups={"grp": ["d"]},
    )

    # The control trajectory comes from the producer, not a discipline input.
    input_names = scenario.discipline.input_grammar.names
    assert "control.d" not in input_names
    assert "constant.e" in input_names
    assert "constant.f" in input_names

    output_data = scenario.discipline.execute()
    times = np.array([0.0, 3.0, 7.0, 10.0])
    values = np.array([0.0, 2.0, 2.0, 0.0])
    expected = np.interp(scenario.time_vector, times, values)
    np.testing.assert_allclose(
        output_data["interpolate.control.d"], expected, atol=1e-10
    )
    np.testing.assert_allclose(
        output_data["grp.controls_constraint"], expected - 1.0, atol=1e-10
    )

    # The delayed control starts from the first control value and stays finite.
    assert output_data["d"][0] == pytest.approx(0.0, abs=1e-12)
    assert np.all(np.isfinite(output_data["d"]))

    # At e = 1 the control breakpoints coincide with time-grid points where the
    # interpolation is not differentiable, so the jacobian is checked nearby.
    input_data = dict(scenario.discipline.default_input_data)
    input_data["constant.e"] = 0.97
    input_data["constant.f"] = 1.1
    assert scenario.discipline.check_jacobian(input_data, threshold=1e-3)


def test_final_rates(models):
    scenario = build_scenario(models, final_rates=["x"])
    output_data = scenario.discipline.execute()
    expected = np.diff(output_data["x"])[-1] / np.diff(scenario.time_vector)[-1]
    assert float(output_data["final_rate.x"]) == pytest.approx(expected, rel=1e-9)


def test_interpolation_prefix(models):
    assert build_scenario(models).interpolation_prefix == "interpolate"
