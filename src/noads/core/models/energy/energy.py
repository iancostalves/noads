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
"""Energy module."""

from __future__ import annotations

from typing import TYPE_CHECKING

from jax.numpy import append
from jax.numpy import array
from jax.numpy import sum as jnp_sum

from noads.core.model import JAXModel
from noads.core.model import Model
from noads.core.models.energy.streams import Impact
from noads.core.models.energy.streams import MaterialStream
from noads.core.models.energy.streams import Stream

if TYPE_CHECKING:
    from collections.abc import Sequence

    from noads.core.models.energy.production_pathway import ProductionPathway


class Energy(Stream):
    """Basic Energy flow."""

    unit = "MJ"

    def __init__(self, name):
        """Initialize Energy from name."""
        Stream.__init__(self, name=name, unit="MJ")


class ProducedEnergy(Energy):
    """Energy flow produced by a set of ProductionPathway's."""

    impacts: list[Impact]
    """Set of all impacts accounted."""

    input_streams: list[Stream]
    """Set of input streams that are directly consumed for production."""

    output_streams: list[Stream]
    """Set of output streams that consume this energy to be produced."""

    pathways: list[ProductionPathway]
    """Set of production pathways for this energy."""

    def __init__(self, name, pathways: Sequence[ProductionPathway]):
        """Initialize ProducedEnergy from name and ProductionPathway's."""
        Energy.__init__(self, name=name)
        self.pathways = list(pathways)
        self.impacts = list({
            impact for pathway in pathways for impact in pathway.impacts
        })
        self.input_streams = list({
            stream for pathway in pathways for stream in pathway.input_streams
        })
        self.output_streams = []

    @property
    def models(self) -> list[Model]:
        """List of models."""
        return [
            self.impact_index_model(),
            self.production_model(),
            self.consumption_model(),
        ]

    def set_output_streams(self, output_streams: Sequence[Stream]):
        """Set streams that consume this energy type."""
        self.output_streams = list(set(output_streams))

    def add_output_stream(self, output_stream: Stream):
        """Add a stream that consume this energy type."""
        if output_stream not in self.output_streams:
            self.output_streams.append(output_stream)

    def production_model(self):
        """Energy production model."""
        default_inputs = {f"{self.name}.production": 0.0}
        output_names = []
        for pathway in self.pathways:
            default_inputs[f"{pathway.name}.share"] = 0.0
            output_names.append(f"{pathway.name}.production")
        return JAXModel(
            function=self._dispatch_production,
            input_names=list(default_inputs.keys()),
            output_names=output_names,
            default_inputs=default_inputs,
            name=f"{self.name} production",
        )

    def _dispatch_production(self, input_data):
        production = input_data[f"{self.name}.production"]
        shares = array([
            input_data[f"{pathway.name}.share"] for pathway in self.pathways
        ])
        production_per_pathway = shares * production
        return {
            f"{pathway.name}.production": production_per_pathway[i]
            for i, pathway in enumerate(self.pathways)
        }

    def consumption_model(self):
        """Energy consumption model."""
        default_inputs = {
            f"{stream.name}.{self.name}.consumption": 0.0
            for stream in self.output_streams
        }
        default_inputs.update({
            f"{self.name}.{stream.name}.consumption_index": 0.0
            for stream in self.input_streams
        })
        output_names = [f"{self.name}.production"]
        output_names.extend([
            f"{self.name}.{stream.name}.consumption" for stream in self.input_streams
        ])
        return JAXModel(
            function=self._consumption_aggregation,
            input_names=list(default_inputs.keys()),
            output_names=output_names,
            default_inputs=default_inputs,
            name=f"{self.name} consumption",
        )

    def _consumption_aggregation(self, input_data):
        production = jnp_sum(
            array([
                input_data[f"{stream.name}.{self.name}.consumption"]
                for stream in self.output_streams
            ])
        )
        consumption_index = {
            stream.name: input_data[f"{self.name}.{stream.name}.consumption_index"]
            for stream in self.input_streams
        }

        output_data = {f"{self.name}.production": production}
        output_data.update({
            f"{self.name}.{stream.name}.consumption": production
            * consumption_index[stream.name]
            for stream in self.input_streams
        })
        return output_data

    def impact_index_model(self):
        """Impact index model."""
        default_inputs = {}
        output_names = []
        for pathway in self.pathways:
            if pathway != self.pathways[-1]:
                default_inputs[f"{pathway.name}.share"] = 0.0
            else:
                output_names.append(f"{pathway.name}.share")
            for stream in self.input_streams:
                if stream in pathway.input_streams:
                    default_inputs[
                        f"{pathway.name}.{stream.name}.consumption_index"
                    ] = 0.0
                output_names.append(f"{self.name}.{stream.name}.consumption_index")
            for impact in self.impacts:
                default_inputs[f"{pathway.name}.{impact.name}_index"] = 0.0
                output_names.append(f"{self.name}.{impact.name}_index")
        return JAXModel(
            function=self._impact_index,
            input_names=list(default_inputs.keys()),
            output_names=output_names,
            default_inputs=default_inputs,
            name=f"{self.name} impacts",
        )

    def _impact_index(self, input_data):
        all_but_last_shares = array([
            input_data[f"{self.pathways[i].name}.share"]
            for i in range(len(self.pathways) - 1)
        ])
        last_share = 1.0 - jnp_sum(all_but_last_shares)
        output_data = {f"{self.pathways[-1].name}.share": last_share}
        shares = append(all_but_last_shares, last_share)

        impact_index = {
            impact.name: array([
                input_data[f"{pathway.name}.{impact.name}_index"]
                for pathway in self.pathways
            ])
            for impact in self.impacts
        }
        consumption_index = {
            stream.name: array([
                input_data[f"{pathway.name}.{stream.name}.consumption_index"]
                if stream in pathway.input_streams
                else 0.0
                for pathway in self.pathways
            ])
            for stream in self.input_streams
        }

        output_data.update({
            f"{self.name}.{impact.name}_index": jnp_sum(
                shares * impact_index[impact.name]
            )
            for impact in self.impacts
        })
        output_data.update({
            f"{self.name}.{stream.name}.consumption_index": jnp_sum(
                shares * consumption_index[stream.name]
            )
            for stream in self.input_streams
        })
        return output_data


class EnergyCarrier(Energy, MaterialStream):
    """Basic flow of energy carrier (energy and material embarked in aircraft)."""

    specific_energy: float

    def __init__(
        self,
        name,
        density,
        specific_energy,
    ):
        """Initialize EnergyCarrier from name, density and specific energy."""
        MaterialStream.__init__(self, name=name, density=density)
        Energy.__init__(self, name=name)
        self.unit = Energy.unit
        self.specific_energy = specific_energy

    def mass_to_energy(self, mass):
        """Convert mass to energy."""
        return mass * self.specific_energy

    def energy_to_mass(self, energy):
        """Convert energy to mass."""
        return energy / self.specific_energy

    def volume_to_energy(self, volume):
        """Convert volume to energy."""
        return self.mass_to_energy(self.volume_to_mass(volume))

    def energy_to_volume(self, energy):
        """Convert energy to volume."""
        return self.mass_to_volume(self.energy_to_mass(energy))


class ProducedEnergyCarrier(ProducedEnergy, EnergyCarrier):
    """Energy Carrier flow produced by a set of ProductionPathway's."""

    def __init__(
        self,
        name,
        pathways: Sequence[ProductionPathway],
        density,
        specific_energy,
    ):
        """Initialize ProducedEnergyCarrier from EnergyCarrier and ProducedEnergy."""
        self.specific_energy = specific_energy
        MaterialStream.__init__(self, name=name, density=density)
        ProducedEnergy.__init__(self, name=name, pathways=pathways)
        self.unit = Energy.unit

    def consumption_model(self):
        """Energy consumption model."""
        default_inputs = {f"{self.name}.consumption": 0.0}
        default_inputs.update({
            f"{stream.name}.{self.name}.consumption": 0.0
            for stream in self.output_streams
        })
        default_inputs.update({
            f"{self.name}.{stream.name}.consumption_index": 0.0
            for stream in self.input_streams
        })
        output_names = [f"{self.name}.production"]
        output_names.extend([
            f"{self.name}.{stream.name}.consumption" for stream in self.input_streams
        ])
        return JAXModel(
            function=self._consumption_aggregation,
            input_names=list(default_inputs.keys()),
            output_names=output_names,
            default_inputs=default_inputs,
            name=f"{self.name} consumption",
        )

    def _consumption_aggregation(self, input_data):
        direct = input_data[f"{self.name}.consumption"]
        indirect = jnp_sum(
            array([
                input_data[f"{stream.name}.{self.name}.consumption"]
                for stream in self.output_streams
            ])
        )
        consumption_index = {
            stream.name: input_data[f"{self.name}.{stream.name}.consumption_index"]
            for stream in self.input_streams
        }

        production = direct + indirect

        output_data = {f"{self.name}.production": production}
        output_data.update({
            f"{self.name}.{stream.name}.consumption": production
            * consumption_index[stream.name]
            for stream in self.input_streams
        })
        return output_data
