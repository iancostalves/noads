from __future__ import annotations

from collections.abc import Sequence

from gemseo import generate_coupling_graph
from jax.numpy import array
from jax.numpy import sum
from jax.numpy import where

from noads.model import JAXModel
from noads.model import Model
from noads.models.energy.energy import ProducedEnergy
from noads.models.energy.energy import ProducedEnergyCarrier
from noads.models.energy.streams import Impact
from noads.models.energy.streams import Stream


class EnergyMix:
    produced_energies: set[ProducedEnergy]

    impacts: set[Impact]
    """Set of all impacts accounted."""

    input_streams: set[Stream]
    """Set of input streams that are not produced by any Pathway."""

    secondary_energies: set[ProducedEnergy]
    """Set of secondary energy (needed for producing other energies)."""

    final_energies: set[ProducedEnergyCarrier]
    """Set of final energy carriers (consumed by a direct.consumption)."""

    constrained_inputs: set[Stream]
    """Set of input streams with limited consumption."""

    def __init__(
        self,
        energies: Sequence[ProducedEnergy],
        extra_impacts: Sequence[Impact] | None = None,
        inputs_to_constrain: Sequence[Stream] | None = None,
        plot_coupling_graph=False,
    ):
        self.produced_energies = set(energies)
        self.impacts = set(
            [impact for energy in self.produced_energies for impact in energy.impacts]
        )
        self.constrained_inputs = set(
            inputs_to_constrain if inputs_to_constrain is not None else []
        )

        # add missing impacts for all pathways
        if extra_impacts is not None:
            for impact in extra_impacts:
                self.impacts.add(impact)

        # add missing impacts for all pathways
        for energy in self.produced_energies:
            for impact in self.impacts:
                if impact not in energy.impacts:
                    energy.impacts.add(impact)
                for pathway in energy.pathways:
                    if impact not in pathway.impacts:
                        pathway.impacts.add(impact)

        self.final_energies = set(
            [
                energy for energy in self.produced_energies
                if isinstance(energy, ProducedEnergyCarrier)
            ]
        )
        self.input_streams = set(
            [
                stream for energy in self.produced_energies
                for pathway in energy.pathways
                for stream in pathway.input_streams
                if stream not in self.produced_energies
            ]
        )
        self.secondary_energy = set(
            [
                stream for energy in self.produced_energies
                for pathway in energy.pathways
                for stream in pathway.input_streams
                if stream in self.produced_energies
            ]
        )
        for energy_to_be_consumed in self.produced_energies:
            for energy in self.produced_energies:
                if energy_to_be_consumed in energy.input_streams:
                    energy_to_be_consumed.add_output_stream(energy)

        if plot_coupling_graph:
            self.plot_pretty_couplings()

    @property
    def models(self) -> list[Model]:
        models = [self.input_streams_model(), self.total_impacts_model()]
        if any(self.constrained_inputs):
            models.append(self.constraint_model())
        for energy in self.produced_energies:
            models.extend(energy.models)
            for pathway in energy.pathways:
                models.extend(pathway.models)
        return models

    def plot_pretty_couplings(self):
        impact_models = []
        prod_conso_models = []
        for energy in self.produced_energies:
            impact_models.extend(
                [energy.impact_index_model()]
            )
            prod_conso_models.extend(
                [energy.production_model(), energy.consumption_model()]
            )
            for pathway in energy.pathways:
                impact_models.append(pathway.impact_index_model())
                prod_conso_models.append(pathway.consumption_model())

        impact_models.append(self.total_impacts_model())
        prod_conso_models.append(self.input_streams_model())

        generate_coupling_graph(
            [model.discipline for model in impact_models], "energy_mix_impact.png"
        )
        generate_coupling_graph(
            [model.discipline for model in prod_conso_models],
            "energy_mix_prod_conso.png",
        )
        generate_coupling_graph(
            [model.discipline for model in self.models], "energy_mix_complete.png"
        )

    def input_streams_model(self):
        default_values_units = {
            f"{energy.name}.{stream.name}.consumption": (0.0, stream.unit)
            for stream in self.input_streams for energy in self.produced_energies
            if stream in energy.input_streams
        }
        output_units = {
            f"{stream.name}.consumption": stream.unit for stream in self.input_streams
        }
        variables = set(default_values_units.keys()).union(output_units)
        fullnames = {
            name: name.replace(".", " ").replace("_", " ") for name in variables
        }
        units = {
            name: default_values_units[name][1] if name in default_values_units.keys()
            else output_units[name] for name in variables
        }
        return JAXModel(
            function=self._inputs_consumption,
            input_names=list(default_values_units.keys()),
            output_names=list(output_units.keys()),
            default_inputs={
                name: value_unit[0] for name, value_unit in default_values_units.items()
            },
            name="Aggregate consumption",
            # variable_full_names=fullnames,
            # variable_units=units,
        )

    def _inputs_consumption(self, input_data):
        consumption_per_stream_per_energy = {
            stream.name: {
                energy.name: input_data[f"{energy.name}.{stream.name}.consumption"]
                if stream in energy.input_streams else 0.0
                for energy in self.produced_energies
            }
            for stream in self.input_streams
        }
        consumption_per_stream = {
            stream.name: sum(array([
                consumption_per_stream_per_energy[stream.name][energy.name]
                for energy in self.produced_energies
            ]))
            for stream in self.input_streams
        }

        output_data = {
            f"{stream.name}.consumption": consumption_per_stream[stream.name]
            for stream in self.input_streams
        }
        return output_data

    def total_impacts_model(self):
        default_values_units = {
            f"{energy.name}.{impact.name}_index": (0.0, f"{impact.unit}/{energy.unit}")
            for energy in self.final_energies
            for impact in self.impacts
        }
        default_values_units.update({
            f"{energy.name}.consumption": (0.0, energy.unit)
            for energy in self.final_energies
        })
        output_units = {f"{impact.name}": impact.unit for impact in self.impacts}
        variables = set(default_values_units.keys()).union(output_units)
        fullnames = {
            name: name.replace(".", " ").replace("_", " ") for name in variables
        }
        units = {
            name: default_values_units[name][1] if name in default_values_units.keys()
            else output_units[name] for name in variables
        }
        return JAXModel(
            function=self._impact_production,
            input_names=list(default_values_units.keys()),
            output_names=list(output_units.keys()),
            default_inputs={
                name: value_unit[0] for name, value_unit in default_values_units.items()
            },
            name="Aggregate impacts",
            # variable_full_names=fullnames,
            # variable_units=units,
        )

    def _impact_production(self, input_data):
        impact_index = {
            energy.name: {
                impact.name: input_data[f"{energy.name}.{impact.name}_index"]
                if impact in energy.impacts else 0.0
                for impact in self.impacts
            }
            for energy in self.final_energies
        }
        consumption = {
            energy.name: input_data[f"{energy.name}.consumption"]
            for energy in self.final_energies
        }
        output_data = {
            impact.name: sum(array([
                consumption[energy.name] * impact_index[energy.name][impact.name]
                for energy in self.final_energies
            ])) for impact in self.impacts
        }
        return output_data

    def constraint_model(self):
        default_inputs = {
            f"{input_stream.name}.fair_share": 1.0
            for input_stream in self.constrained_inputs
        }
        default_inputs.update({
            f"{input_stream.name}.global_production": 1.0
            for input_stream in self.constrained_inputs
        })
        default_inputs.update({
            f"{input_stream.name}.consumption": 0.0
            for input_stream in self.constrained_inputs
        })
        default_inputs.update({
            f"{energy.pathways[-1].name}.share": 0.0
            for energy in self.produced_energies if len(energy.pathways) > 1
        })
        output_names = [
            f"{energy.name}.constraint"
            for energy in self.produced_energies if len(energy.pathways) > 1
        ]
        output_names.extend([
            f"{energy.name}.constraint_violation"
            for energy in self.produced_energies if len(energy.pathways) > 1
        ])
        output_names.extend([
            f"{input_stream.name}.consumed_share"
            for input_stream in self.constrained_inputs
        ])
        output_names.extend([
            f"{input_stream.name}.constraint"
            for input_stream in self.constrained_inputs
        ])
        output_names.extend(
            [
                f"{input_stream.name}.constraint_violation"
                for input_stream in self.constrained_inputs
            ]
        )
        return JAXModel(
            function=self._constraint,
            input_names=list(default_inputs.keys()),
            output_names=output_names,
            default_inputs=default_inputs,
            name="Energy constraints",
        )

    def _constraint(self, input_data):
        last_share = {
            energy.name: input_data[f"{energy.pathways[-1].name}.share"]
            for energy in self.produced_energies if len(energy.pathways) > 1
        }
        output_data = {
            f"{energy.name}.constraint": -last_share[energy.name]
            for energy in self.produced_energies if len(energy.pathways) > 1
        }
        output_data.update({
            f"{energy.name}.constraint_violation": where(
                last_share[energy.name] < 0, -last_share[energy.name], 0.0
            ) for energy in self.produced_energies if len(energy.pathways) > 1
        })

        global_production = {
            input_stream.name: input_data[f"{input_stream.name}.global_production"]
            for input_stream in self.constrained_inputs
        }
        consumption = {
            input_stream.name: input_data[f"{input_stream.name}.consumption"]
            for input_stream in self.constrained_inputs
        }

        consumed_share = {
            input_stream.name:
                consumption[input_stream.name] / global_production[input_stream.name]
            for input_stream in self.constrained_inputs
        }
        output_data.update({
            f"{input_stream.name}.consumed_share": consumed_share[input_stream.name]
            for input_stream in self.constrained_inputs
        })

        fair_share = {
            input_stream.name: input_data[f"{input_stream.name}.fair_share"]
            for input_stream in self.constrained_inputs
        }
        input_constraint = {
            f"{input_stream.name}.constraint":
                consumed_share[input_stream.name] - fair_share[input_stream.name]
            for input_stream in self.constrained_inputs
        }
        input_constraint.update({
            f"{input_stream.name}.constraint_violation": where(
                consumed_share[input_stream.name] > fair_share[input_stream.name],
                consumed_share[input_stream.name] - fair_share[input_stream.name],
                0.0,
            )
            for input_stream in self.constrained_inputs
        })
        output_data.update(input_constraint)
        return output_data
