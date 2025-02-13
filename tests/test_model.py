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

from collections.abc import Callable
from collections.abc import Mapping

import pytest
from gemseo_jax.jax_discipline import JAXDiscipline
from gemseo_jax.jax_discipline import NumberLike
from numpy import array
from numpy.testing import assert_equal

from core.model import AutoModel
from core.model import JAXModel
from core.model import Model


@pytest.fixture(scope="module")
def function() -> Callable[[Mapping[str, NumberLike]], Mapping[str, NumberLike]]:
    """A function computing linear combinations."""

    def my_function(input_data: Mapping[str, NumberLike]):
        """A function computing linear combinations.

        Args:
            input_data: The input data.

        Returns:
            The output data.
        """
        a = input_data["a"]
        b = input_data["b"]
        c = input_data["c"]
        d = input_data["d"]

        x = 2 * a + b - c * d
        y = a * b
        z = c * d
        return {"x": x, "y": y, "z": z}

    return my_function


@pytest.fixture(scope="module")
def input_names() -> list[str]:
    """The input names."""
    return ["a", "b", "c", "d"]


@pytest.fixture(scope="module")
def output_names() -> list[str]:
    """The output names."""
    return ["x", "y", "z"]


@pytest.fixture(scope="module")
def default_inputs() -> dict[str, NumberLike]:
    """The default input values."""
    # The mix of scalar, mono-dimensional and multidimensional input variables
    # generates scalar, mono-dimensional and multidimensional variables,
    # which allows to test the robustness of execute() and compute_jacobian().
    return {"a": 0.0, "b": array([1.0]), "c": 2.0, "d": array([3.0, 3.0])}


def test_model(function, input_names, output_names, default_inputs):
    discipline = JAXDiscipline(function, input_names, output_names, default_inputs)
    model = Model(discipline)

    assert model.discipline is not None
    assert set(model.discipline.input_grammar) == set(input_names)
    assert set(model.discipline.output_grammar) == set(output_names)
    for input_name in input_names:
        assert_equal(
            model.discipline.default_input_data[input_name], default_inputs[input_name]
        )

    output_data = model.discipline.execute()
    default_outputs = function(default_inputs)

    for output_name in output_names:
        assert_equal(output_data[output_name], default_outputs[output_name])

    assert model.discipline.check_jacobian()


def test_jax_model(function, input_names, output_names, default_inputs):
    model = JAXModel(function, input_names, output_names, default_inputs, "test")

    assert model.discipline is not None
    assert model.discipline.name == "test"
    assert set(model.discipline.input_grammar) == set(input_names)
    assert set(model.discipline.output_grammar) == set(output_names)
    for input_name in input_names:
        assert_equal(
            model.discipline.default_input_data[input_name], default_inputs[input_name]
        )

    output_data = model.discipline.execute()
    default_outputs = function(default_inputs)

    for output_name in output_names:
        assert_equal(output_data[output_name], default_outputs[output_name])

    assert model.discipline.check_jacobian()


class TestAutoModel(AutoModel):
    def __init__(self):
        super().__init__("test")

    def _jax_func(self, a=0.0, b=1.0, c=2.0, d=3.0) -> tuple[float, float, float]:
        x = 2 * a + b - c * d
        y = a * b
        z = c * d
        return x, y, z


@pytest.fixture(scope="module")
def auto_model() -> AutoModel:
    return TestAutoModel()


@pytest.fixture(scope="module")
def auto_defaults() -> dict[str, float]:
    return {"a": 0.0, "b": 1.0, "c": 2.0, "d": 3.0}


def test_auto_model(auto_model, function, input_names, output_names, auto_defaults):
    assert auto_model.discipline is not None
    assert auto_model.discipline.name == "test"
    assert set(auto_model.discipline.input_grammar) == set(input_names)
    assert set(auto_model.discipline.output_grammar) == set(output_names)
    for input_name in input_names:
        assert_equal(
            auto_model.discipline.default_input_data[input_name],
            auto_defaults[input_name],
        )

    output_data = auto_model.discipline.execute()
    default_outputs = function(auto_defaults)

    for output_name in output_names:
        assert_equal(output_data[output_name], default_outputs[output_name])

    assert auto_model.discipline.check_jacobian()
