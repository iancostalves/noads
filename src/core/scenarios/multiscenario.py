from collections.abc import Sequence

from gemseo_jax.jax_discipline import DataType
from jax import vmap
from jax.numpy import mean
from jax.numpy import stack

from gemseo_jax.jax_discipline import JAXDiscipline

from core.model import Model
from core.scenarios.temporalscenario import TemporalScenario


class MultiScenario(Model):
    _fixed_prefix: str = "fixed"

    _mean_prefix: str = "mean"

    temporal_scenario: TemporalScenario

    scenario_names: list[str]

    fixed_inputs: list[str]

    scenario_inputs: list[str]

    scenario_outputs: list[str]

    mean_outputs: list[str]

    def __init__(
        self,
        temporal_scenario: TemporalScenario,
        mean_outputs: Sequence[str],
        scenario_names: Sequence[str],
        fixed_inputs: Sequence[str],
        differentiation_method: JAXDiscipline.DifferentiationMethod =
        JAXDiscipline.DifferentiationMethod.REVERSE,
    ):
        if differentiation_method == JAXDiscipline.DifferentiationMethod.AUTO:
            raise ValueError("Chose either forward or reverse AutoDiff method.")

        self.temporal_scenario = temporal_scenario
        self.scenario_names = list(scenario_names)
        self.fixed_inputs = list(fixed_inputs)
        self.scenario_inputs = [
            name for name in self.temporal_scenario.discipline.input_grammar.names
            if name not in self.fixed_inputs
        ]
        self.scenario_outputs = list(self.temporal_scenario.discipline.output_grammar.names)
        self.mean_outputs = list(mean_outputs)

        default_inputs = {
            f"{self._fixed_prefix}.{name}":
                self.temporal_scenario.discipline.default_inputs[name]
            for name in self.fixed_inputs
        }
        default_inputs.update({
            f"{scenario}.{name}":
                self.temporal_scenario.discipline.default_inputs[name]
            for scenario in self.scenario_names for name in self.scenario_inputs
        })
        output_names = [
            f"{scenario}.{name}" for scenario in self.scenario_names
            for name in self.scenario_outputs
        ]
        output_names.extend(
            [f"{self._mean_prefix}.{name}" for name in self.mean_outputs]
        )
        discipline = JAXDiscipline(
            name=f"Multi-scenario {self.temporal_scenario.discipline.name}",
            function=self.__multi_scenario,
            input_names=list(default_inputs.keys()),
            output_names=output_names,
            default_inputs=default_inputs,
            differentiation_method=differentiation_method,
        )
        discipline.set_cache_policy()
        super().__init__(discipline)

    def __multi_scenario(self, input_data: DataType) -> DataType:
        stacked_scenario_inputs = {
            name: stack([
                input_data[f"{self._fixed_prefix}.{name}"] for s in self.scenario_names
            ])
            for name in self.fixed_inputs
        }
        stacked_scenario_inputs.update(
            {
                name: stack(
                    [
                        input_data[f"{scenario}.{name}"]
                        for scenario in self.scenario_names
                    ],
                    axis=0,
                ) for name in self.scenario_inputs
            }
        )

        vectorized_func = vmap(self.temporal_scenario.discipline.jax_out_func)
        stacked_scenario_outputs = vectorized_func(stacked_scenario_inputs)

        output_data = {
            f"{self._mean_prefix}.{name}": mean(stacked_scenario_outputs[name], axis=0)
            for name in self.mean_outputs
        }
        output_data.update({
            f"{scenario}.{name}": stacked_scenario_outputs[name][i]
            for i, scenario in enumerate(self.scenario_names)
            for name in self.scenario_outputs
        })
        return output_data
