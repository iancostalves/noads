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
    """Basic model, from an instantiated JAXDiscipline."""

    discipline: JAXDiscipline

    def __init__(
        self,
        discipline: JAXDiscipline,
    ):
        """Initialize Model from existing JAXDiscipline."""
        self.discipline = discipline


class JAXModel(Model):
    """Basic model, from a dict-based JAX function, names, and default inputs."""

    def __init__(
        self,
        function,
        input_names,
        output_names,
        default_inputs,
        name,
    ):
        """Initialize JAXModel and JAXDiscipline from scratch."""
        discipline = JAXDiscipline(
            function=function,
            input_names=input_names,
            output_names=output_names,
            default_inputs=default_inputs,
            name=name,
        )
        super().__init__(discipline)


class AutoModel(Model):
    """Basic model, from named args based JAX function."""

    discipline: AutoJAXDiscipline

    def __init__(self, name):
        """Initialize AutoModel from abstract JAX function."""
        discipline = AutoJAXDiscipline(name=name, function=self._jax_func)
        super().__init__(discipline)

    @abstractmethod
    def _jax_func(self, *args: Sequence[NumberLike]) -> tuple[NumberLike]:
        """The JAX function used by the JAXDiscipline."""
