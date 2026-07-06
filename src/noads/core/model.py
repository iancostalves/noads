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
"""Core model based on gemseo-jax classes."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from gemseo_jax.auto_jax_discipline import AutoJAXDiscipline
from gemseo_jax.jax_discipline import JAXDiscipline
from gemseo_jax.jax_discipline import NumberLike

if TYPE_CHECKING:
    from collections.abc import Sequence


class Model:  # noqa: B903
    """Base class of all NOADS models, wrapping a GEMSEO-JAX discipline.

    A model holds a single :class:`gemseo_jax.jax_discipline.JAXDiscipline`, i.e. a
    JAX-traceable function with named inputs and outputs that GEMSEO can execute,
    couple with other disciplines, differentiate with automatic differentiation, and
    compile just-in-time. Physical and behavioral models (aircraft, fleet, energy,
    traffic) expose one or several ``*_model()`` factory methods returning
    :class:`Model` instances, which are then assembled by
    :class:`~noads.core.scenarios.temporalscenario.TemporalScenario`.
    """

    discipline: JAXDiscipline

    def __init__(
        self,
        discipline: JAXDiscipline,
    ):
        """Wrap an existing discipline.

        Args:
            discipline: An instantiated GEMSEO-JAX discipline to wrap.
        """
        self.discipline = discipline


class JAXModel(Model):
    """Model built from a dict-based JAX function.

    The function maps a dictionary of named input arrays to a dictionary of named
    output arrays. Input and output names are given explicitly, which allows
    functions whose signature cannot be introspected (e.g. closures or partial
    applications). For plain functions with named arguments, prefer
    :class:`AutoModel`, which detects names automatically.
    """

    def __init__(
        self,
        function,
        input_names,
        output_names,
        default_inputs,
        name,
    ):
        """Build the discipline from the function and names.

        Args:
            function: The JAX function, mapping a dict of input arrays keyed by
                ``input_names`` to a dict of output arrays keyed by
                ``output_names``.
            input_names: The names of the inputs.
            output_names: The names of the outputs.
            default_inputs: The default value of each input, keyed by name; also
                determines the shapes used for tracing and compilation.
            name: The name of the discipline.

        """
        discipline = JAXDiscipline(
            function=function,
            input_names=input_names,
            output_names=output_names,
            default_inputs=default_inputs,
            name=name,
        )
        super().__init__(discipline)


class AutoModel(Model):
    """Model whose inputs and outputs are auto-detected from a JAX function.

    Subclasses implement :meth:`_jax_func` with named arguments; the argument names
    become the discipline input names and the returned variables the output names
    (via :class:`gemseo_jax.auto_jax_discipline.AutoJAXDiscipline`). This is the
    most common base class for NOADS models, as it removes the boilerplate of
    declaring grammars by hand.
    """

    discipline: AutoJAXDiscipline

    def __init__(self, name):
        """Create the discipline from the subclass JAX function.

        Args:
            name: The name of the discipline.
        """
        discipline = AutoJAXDiscipline(name=name, function=self._jax_func)
        super().__init__(discipline)

    @abstractmethod
    def _jax_func(self, *args: Sequence[NumberLike]) -> tuple[NumberLike]:
        """The JAX function wrapped by the discipline.

        Argument names define the discipline inputs and their default values; the
        names of the returned variables define the outputs. The function must be
        JAX-traceable (pure, no Python side effects on traced values).
        """
