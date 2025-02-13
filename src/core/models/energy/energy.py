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

from __future__ import annotations

from typing import TYPE_CHECKING

from jax.numpy import append
from jax.numpy import array
from jax.numpy import sum

from core.model import JAXModel
from core.model import Model
from core.models.energy.streams import Impact
from core.models.energy.streams import MaterialStream
from core.models.energy.streams import Stream

if TYPE_CHECKING:
    from collections.abc import Sequence

    from core.models.energy.production_pathway import ProductionPathway


class Energy(Stream):
    unit = "MJ"

    def __init__(self, name):
        Stream.__init__(self, name=name, unit="MJ")


class ProducedEnergy(Energy):
    impacts: list[Impact]
    """Set of all impacts accounted."""

    input_streams: list[Stream]
    """Set of input streams that are directly consumed for production."""

    output_streams: list[Stream]
    """Set of output streams that consume this energy to be produced."""

    pathways: list[ProductionPathway]
    """Set of production pathways for this energy."""

    def __init__(self, name, pathways: Sequence[ProductionPathway]):
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
        return [
            self.impact_index_model(),
            self.production_model(),
            self.consumption_model(),
        ]

    def set_output_streams(self, output_streams: Sequence[Stream]):
        self.output_streams = list(set(output_streams))

    def add_output_stream(self, output_stream: Stream):
        if output_stream not in self.output_streams:
            self.output_streams.append(output_stream)

    def production_model(self):
        default_values_units = {f"{self.name}.production": (0.0, self.unit)}
        output_units = {}
        for pathway in self.pathways:
            default_values_units[f"{pathway.name}.share"] = (0.0, "")
            output_units.update({f"{pathway.name}.production": Energy.unit})
        variables = set(default_values_units.keys()).union(output_units)
        {name: name.replace(".", " ").replace("_", " ") for name in variables}
        {
            name: default_values_units[name][1]
            if name in default_values_units
            else output_units[name]
            for name in variables
        }
        return JAXModel(
            function=self._dispatch_production,
            input_names=list(default_values_units.keys()),
            output_names=list(output_units.keys()),
            default_inputs={
                name: value_unit[0] for name, value_unit in default_values_units.items()
            },
            name=f"{self.name} production",
            # variable_full_names=fullnames,
            # variable_units=units,
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
        default_values_units = {
            f"{stream.name}.{self.name}.consumption": (0.0, self.unit)
            for stream in self.output_streams
        }
        default_values_units.update({
            f"{self.name}.{stream.name}.consumption_index": (
                0.0,
                f"{stream.unit}/{self.unit}",
            )
            for stream in self.input_streams
        })
        output_units = {f"{self.name}.production": self.unit}
        output_units.update({
            f"{self.name}.{stream.name}.consumption": self.unit
            for stream in self.input_streams
        })
        variables = set(default_values_units.keys()).union(output_units)
        {name: name.replace(".", " ").replace("_", " ") for name in variables}
        {
            name: default_values_units[name][1]
            if name in default_values_units
            else output_units[name]
            for name in variables
        }
        return JAXModel(
            function=self._consumption_aggregation,
            input_names=list(default_values_units.keys()),
            output_names=list(output_units.keys()),
            default_inputs={
                name: value_unit[0] for name, value_unit in default_values_units.items()
            },
            name=f"{self.name} consumption",
            # variable_full_names=fullnames,
            # variable_units=units,
        )

    def _consumption_aggregation(self, input_data):
        production = sum(
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
        default_values_units = {}
        output_units = {}
        for pathway in self.pathways:
            if pathway != self.pathways[-1]:
                default_values_units[f"{pathway.name}.share"] = (0.0, "")
            else:
                output_units.update({f"{pathway.name}.share": ""})
            for stream in self.input_streams:
                if stream in pathway.input_streams:
                    default_values_units[
                        f"{pathway.name}.{stream.name}.consumption_index"
                    ] = (0.0, f"{stream.unit}/{Energy.unit}")
                output_units.update({
                    f"{self.name}.{stream.name}.consumption_index": f"{stream.unit}/{self.unit}"
                })
            for impact in self.impacts:
                default_values_units[f"{pathway.name}.{impact.name}_index"] = (
                    0.0,
                    f"{impact.unit}/{Energy.unit}",
                )
                output_units.update({
                    f"{self.name}.{impact.name}_index": f"{impact.unit}/{self.unit}"
                })
        variables = set(default_values_units.keys()).union(output_units)
        {name: name.replace(".", " ").replace("_", " ") for name in variables}
        {
            name: default_values_units[name][1]
            if name in default_values_units
            else output_units[name]
            for name in variables
        }
        return JAXModel(
            function=self._impact_index,
            input_names=list(default_values_units.keys()),
            output_names=list(output_units.keys()),
            default_inputs={
                name: value_unit[0] for name, value_unit in default_values_units.items()
            },
            name=f"{self.name} impacts",
            # variable_full_names=fullnames,
            # variable_units=units,
        )

    def _impact_index(self, input_data):
        all_but_last_shares = array([
            input_data[f"{self.pathways[i].name}.share"]
            for i in range(len(self.pathways) - 1)
        ])
        last_share = 1 - sum(all_but_last_shares)
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
            f"{self.name}.{impact.name}_index": sum(shares * impact_index[impact.name])
            for impact in self.impacts
        })
        output_data.update({
            f"{self.name}.{stream.name}.consumption_index": sum(
                shares * consumption_index[stream.name]
            )
            for stream in self.input_streams
        })
        return output_data


class EnergyCarrier(Energy, MaterialStream):
    specific_energy: float

    def __init__(
        self,
        name,
        density,
        specific_energy,
    ):
        MaterialStream.__init__(self, name=name, density=density)
        Energy.__init__(self, name=name)
        self.unit = Energy.unit
        self.specific_energy = specific_energy

    def mass_to_energy(self, mass):
        return mass * self.specific_energy

    def energy_to_mass(self, energy):
        return energy / self.specific_energy

    def volume_to_energy(self, volume):
        return self.mass_to_energy(self.volume_to_mass(volume))

    def energy_to_volume(self, energy):
        return self.mass_to_volume(self.energy_to_mass(energy))


class ProducedEnergyCarrier(ProducedEnergy, EnergyCarrier):
    def __init__(
        self,
        name,
        pathways: Sequence[ProductionPathway],
        density,
        specific_energy,
    ):
        self.specific_energy = specific_energy
        MaterialStream.__init__(self, name=name, density=density)
        ProducedEnergy.__init__(self, name=name, pathways=pathways)
        self.unit = Energy.unit

    def consumption_model(self):
        default_values_units = {
            f"{self.name}.consumption": (0.0, self.unit),
        }
        default_values_units.update({
            f"{stream.name}.{self.name}.consumption": (0.0, self.unit)
            for stream in self.output_streams
        })
        default_values_units.update({
            f"{self.name}.{stream.name}.consumption_index": (
                0.0,
                f"{stream.unit}/{self.unit}",
            )
            for stream in self.input_streams
        })
        output_units = {f"{self.name}.production": self.unit}
        output_units.update({
            f"{self.name}.{stream.name}.consumption": stream.unit
            for stream in self.input_streams
        })
        variables = set(default_values_units.keys()).union(output_units)
        {name: name.replace(".", " ").replace("_", " ") for name in variables}
        {
            name: default_values_units[name][1]
            if name in default_values_units
            else output_units[name]
            for name in variables
        }

        return JAXModel(
            function=self._consumption_aggregation,
            input_names=list(default_values_units.keys()),
            output_names=list(output_units.keys()),
            default_inputs={
                name: value_unit[0] for name, value_unit in default_values_units.items()
            },
            name=f"{self.name} consumption",
            # variable_full_names=fullnames,
            # variable_units=units,
        )

    def _consumption_aggregation(self, input_data):
        direct = input_data[f"{self.name}.consumption"]
        indirect = sum(
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
