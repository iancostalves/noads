
from __future__ import annotations

from abc import abstractmethod
from typing import Sequence

from gemseo_jax.auto_jax_discipline import AutoJAXDiscipline
from gemseo_jax.jax_discipline import JAXDiscipline
from gemseo_jax.jax_discipline import NumberLike


class Model:

    discipline: JAXDiscipline

    def __init__(
        self,
        discipline: JAXDiscipline,
    ):
        self.discipline = discipline


class JAXModel(Model):

    def __init__(
        self,
        function,
        input_names,
        output_names,
        default_inputs,
        name,
    ):
        discipline = JAXDiscipline(
            function=function,
            input_names=input_names,
            output_names=output_names,
            default_inputs=default_inputs,
            name=name,
        )
        super().__init__(discipline)


class AutoModel(Model):

    discipline: AutoJAXDiscipline

    def __init__(self, name):

        discipline = AutoJAXDiscipline(name=name, function=self._jax_func)
        super().__init__(discipline)

    @abstractmethod
    def _jax_func(self, *args: Sequence[NumberLike]) -> tuple[NumberLike]:
        """The JAX function used by the JAXDiscipline."""