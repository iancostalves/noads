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
"""Assembly of Energy types and Production pathways into an EnergyMix."""

from __future__ import annotations

from typing import TYPE_CHECKING

from jax.numpy import array
from jax.numpy import sum as jnp_sum
from jax.numpy import where

from noads.core.model import JAXModel
from noads.core.model import Model
from noads.core.models.energy.energy import ProducedEnergy
from noads.core.models.energy.energy import ProducedEnergyCarrier

if TYPE_CHECKING:
    from collections.abc import Sequence

    from noads.core.models.energy.streams import Impact
    from noads.core.models.energy.streams import Stream


class EnergyMix:
    """Mix of Energy types among Production Pathways."""

    produced_energies: list[ProducedEnergy]
    """List of all energy types with explicit production from Pathways."""

    impacts: list[Impact]
    """List of all impacts accounted."""

    input_streams: list[Stream]
    """List of input streams that are not produced by any Pathway."""

    secondary_energies: list[ProducedEnergy]
    """List of secondary energy (needed for producing other energies)."""

    final_energies: list[ProducedEnergyCarrier]
    """List of final energy carriers (consumed by a direct.consumption)."""

    constrained_inputs: list[Stream]
    """List of input streams with limited consumption."""

    def __init__(
        self,
        energies: Sequence[ProducedEnergy],
        extra_impacts: Sequence[Impact] | None = None,
        inputs_to_constrain: Sequence[Stream] | None = None,
        plot_coupling_graph=False,
    ):
        """Initialize EnergyMix."""
        self.produced_energies = set(energies)
        self.impacts = list({
            impact for energy in self.produced_energies for impact in energy.impacts
        })
        self.constrained_inputs = list(
            set(inputs_to_constrain if inputs_to_constrain is not None else [])
        )

        # add missing impacts for all pathways
        if extra_impacts is not None:
            for impact in extra_impacts:
                if impact not in self.impacts:
                    self.impacts.append(impact)

        # add missing impacts for all pathways
        for energy in self.produced_energies:
            for impact in self.impacts:
                if impact not in energy.impacts:
                    energy.impacts.append(impact)
                for pathway in energy.pathways:
                    if impact not in pathway.impacts:
                        pathway.impacts.append(impact)

        self.final_energies = list({
            energy
            for energy in self.produced_energies
            if isinstance(energy, ProducedEnergyCarrier)
        })
        self.input_streams = list({
            stream
            for energy in self.produced_energies
            for pathway in energy.pathways
            for stream in pathway.input_streams
            if stream not in self.produced_energies
        })
        self.secondary_energy = list({
            stream
            for energy in self.produced_energies
            for pathway in energy.pathways
            for stream in pathway.input_streams
            if stream in self.produced_energies
        })
        for energy_to_be_consumed in self.produced_energies:
            for energy in self.produced_energies:
                if energy_to_be_consumed in energy.input_streams:
                    energy_to_be_consumed.add_output_stream(energy)

        if plot_coupling_graph:
            self.plot_pretty_couplings()

    @property
    def models(self) -> list[Model]:
        """List of all EnergyMix models."""
        models = [self.input_streams_model(), self.total_impacts_model()]
        if any(self.constrained_inputs):
            models.append(self.constraint_model())
        for energy in self.produced_energies:
            models.extend(energy.models)
            for pathway in energy.pathways:
                models.extend(pathway.models)
        return models

    def plot_pretty_couplings(self):
        """Plot and save coupling graphs of the EnergyMix."""
        impact_models = []
        prod_conso_models = []
        for energy in self.produced_energies:
            impact_models.extend([energy.impact_index_model()])
            prod_conso_models.extend([
                energy.production_model(),
                energy.consumption_model(),
            ])
            for pathway in energy.pathways:
                impact_models.append(pathway.impact_index_model())
                prod_conso_models.append(pathway.consumption_model())

        impact_models.append(self.total_impacts_model())
        prod_conso_models.append(self.input_streams_model())

        # generate_coupling_graph(
        #     [model.discipline for model in impact_models], "energy_mix_impact.pdf"
        # )
        # generate_coupling_graph(
        #     [model.discipline for model in prod_conso_models],
        #     "energy_mix_prod_conso.pdf",
        # )
        # generate_coupling_graph(
        #     [model.discipline for model in self.models], "energy_mix_complete.pdf"
        # )

    def input_streams_model(self):
        """Aggregate consumption of input streams model."""
        default_values_units = {
            f"{energy.name}.{stream.name}.consumption": (0.0, stream.unit)
            for stream in self.input_streams
            for energy in self.produced_energies
            if stream in energy.input_streams
        }
        output_units = {
            f"{stream.name}.consumption": stream.unit for stream in self.input_streams
        }
        return JAXModel(
            function=self._inputs_consumption,
            input_names=list(default_values_units.keys()),
            output_names=list(output_units.keys()),
            default_inputs={
                name: value_unit[0] for name, value_unit in default_values_units.items()
            },
            name="Aggregate consumption",
        )

    def _inputs_consumption(self, input_data):
        consumption_per_stream_per_energy = {
            stream.name: {
                energy.name: input_data[f"{energy.name}.{stream.name}.consumption"]
                if stream in energy.input_streams
                else 0.0
                for energy in self.produced_energies
            }
            for stream in self.input_streams
        }
        consumption_per_stream = {
            stream.name: jnp_sum(
                array([
                    consumption_per_stream_per_energy[stream.name][energy.name]
                    for energy in self.produced_energies
                ])
            )
            for stream in self.input_streams
        }

        return {
            f"{stream.name}.consumption": consumption_per_stream[stream.name]
            for stream in self.input_streams
        }

    def total_impacts_model(self):
        """Aggregate production of impacts model."""
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
        return JAXModel(
            function=self._impact_production,
            input_names=list(default_values_units.keys()),
            output_names=list(output_units.keys()),
            default_inputs={
                name: value_unit[0] for name, value_unit in default_values_units.items()
            },
            name="Aggregate impacts",
        )

    def _impact_production(self, input_data):
        impact_index = {
            energy.name: {
                impact.name: input_data[f"{energy.name}.{impact.name}_index"]
                if impact in energy.impacts
                else 0.0
                for impact in self.impacts
            }
            for energy in self.final_energies
        }
        consumption = {
            energy.name: input_data[f"{energy.name}.consumption"]
            for energy in self.final_energies
        }
        return {
            impact.name: jnp_sum(
                array([
                    consumption[energy.name] * impact_index[energy.name][impact.name]
                    for energy in self.final_energies
                ])
            )
            for impact in self.impacts
        }

    def constraint_model(self):
        """Constraints on consumption of input streams and non-negative production."""
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
            for energy in self.produced_energies
            if len(energy.pathways) > 1
        })
        output_names = [
            f"{energy.name}.constraint"
            for energy in self.produced_energies
            if len(energy.pathways) > 1
        ]
        output_names.extend([
            f"{energy.name}.constraint_violation"
            for energy in self.produced_energies
            if len(energy.pathways) > 1
        ])
        output_names.extend([
            f"{input_stream.name}.consumed_share"
            for input_stream in self.constrained_inputs
        ])
        output_names.extend([
            f"{input_stream.name}.constraint"
            for input_stream in self.constrained_inputs
        ])
        output_names.extend([
            f"{input_stream.name}.constraint_violation"
            for input_stream in self.constrained_inputs
        ])
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
            for energy in self.produced_energies
            if len(energy.pathways) > 1
        }
        output_data = {
            f"{energy.name}.constraint": -last_share[energy.name]
            for energy in self.produced_energies
            if len(energy.pathways) > 1
        }
        output_data.update({
            f"{energy.name}.constraint_violation": where(
                last_share[energy.name] < 0, -last_share[energy.name], 0.0
            )
            for energy in self.produced_energies
            if len(energy.pathways) > 1
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
            input_stream.name: consumption[input_stream.name]
            / global_production[input_stream.name]
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
            f"{input_stream.name}.constraint": consumed_share[input_stream.name]
            - fair_share[input_stream.name]
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
