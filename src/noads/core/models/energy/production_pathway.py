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
"""Energy production pathway."""

from __future__ import annotations

from typing import TYPE_CHECKING

from jax.numpy import array
from jax.numpy import atleast_1d
from jax.numpy import sum as jnp_sum

from noads.core.model import JAXModel
from noads.core.model import Model

if TYPE_CHECKING:
    from collections.abc import Sequence

    from noads.core.models.energy.streams import Impact
    from noads.core.models.energy.streams import Stream


class ProductionPathway:
    """An energy conversion process, consuming inputs to produce an output energy.

    A pathway is an edge of the energy production graph: it consumes input streams
    (``<pathway>.<input>.efficiency``, in MJ of input per MJ produced) and generates
    direct impacts (``<pathway>.direct.<impact>_index``, e.g. g CO2 per MJ). Its
    total impact index adds the indirect impacts of its inputs, following Eq. (11)
    of the extended paper. These coefficients are scenario inputs, set in
    :mod:`noads.application.scenario_setup`.
    """

    name: str
    """Name of the pathway."""

    impacts: list[Impact]
    """Direct impacts of production."""

    input_streams: list[Stream]
    """Energy/material inputs for pathway production."""

    def __init__(
        self,
        name: str,
        impacts: Sequence[Impact],
        input_streams: Sequence[Stream],
    ):
        """Initialize ProductionPathway from name, impacts and inputs."""
        self.name = name
        self.impacts = list(set(impacts))
        self.input_streams = list(set(input_streams))

    @property
    def models(self) -> list[Model]:
        """List of models."""
        return [self.impact_index_model(), self.consumption_model()]

    def _impact_index(self, input_data):
        streams_impact_index = {
            impact.name: {
                stream.name: input_data[f"{stream.name}.{impact.name}_index"]
                for stream in self.input_streams
            }
            for impact in self.impacts
        }
        # input streams per production unit
        consumption_per_prod = {
            stream: 1.0 / input_data[f"{self.name}.{stream.name}.efficiency"]
            for stream in self.input_streams
        }

        # direct impacts from pathway
        direct_impact_per_prod = {
            impact: input_data[f"{self.name}.direct.{impact.name}_index"]
            for impact in self.impacts
        }

        # Impact index from direct production
        output_data = {
            f"{self.name}.direct.{impact.name}_index": direct_impact
            for impact, direct_impact in direct_impact_per_prod.items()
        }

        # Impact index imported from input streams
        output_data.update({
            f"{self.name}.indirect.{impact.name}_index": jnp_sum(
                array([
                    atleast_1d(
                        stream_consumption
                        * streams_impact_index[impact.name][stream.name]
                    )
                    for stream, stream_consumption in consumption_per_prod.items()
                ])
            )
            for impact in self.impacts
        })

        # Total impacts index
        output_data.update({
            f"{self.name}.{impact.name}_index": output_data[
                f"{self.name}.indirect.{impact.name}_index"
            ]
            + direct_impact
            for impact, direct_impact in direct_impact_per_prod.items()
        })

        # Final consumption
        output_data.update({
            f"{self.name}.{stream.name}.consumption_index": stream_consumption
            for stream, stream_consumption in consumption_per_prod.items()
        })

        return output_data

    def impact_index_model(self):
        """Pathway impact index model."""
        default_inputs = {
            f"{self.name}.direct.{impact.name}_index": 0.0 for impact in self.impacts
        }
        default_inputs.update({
            f"{self.name}.{input_stream.name}.efficiency": 1.0
            for input_stream in self.input_streams
        })
        output_names = []
        for impact in self.impacts:
            output_names.extend([
                f"{self.name}.{impact.name}_index",
                f"{self.name}.indirect.{impact.name}_index",
            ])
            for input_stream in self.input_streams:
                default_inputs.update({
                    f"{input_stream.name}.{impact.name}_index": 0.0,
                })
                output_names.append(
                    f"{self.name}.{input_stream.name}.consumption_index"
                )
        return JAXModel(
            function=self._impact_index,
            input_names=list(default_inputs.keys()),
            output_names=output_names,
            default_inputs=default_inputs,
            name=f"{self.name} impacts",
        )

    def consumption_model(self):
        """Pathway consumption model."""
        default_inputs = {f"{self.name}.production": 0.0}
        default_inputs.update({
            f"{self.name}.{stream.name}.consumption_index": 0.0
            for stream in self.input_streams
        })
        output_names = {
            f"{self.name}.{stream.name}.consumption": stream.unit
            for stream in self.input_streams
        }
        return JAXModel(
            function=self._consumption,
            input_names=list(default_inputs.keys()),
            output_names=list(output_names.keys()),
            default_inputs=default_inputs,
            name=f"{self.name} production",
        )

    def _consumption(self, input_data):
        production = input_data[f"{self.name}.production"]
        consumption_index = {
            stream.name: input_data[f"{self.name}.{stream.name}.consumption_index"]
            for stream in self.input_streams
        }
        return {
            f"{self.name}.{stream.name}.consumption": production
            * consumption_index[stream.name]
            for stream in self.input_streams
        }
