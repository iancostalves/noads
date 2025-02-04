from collections.abc import Sequence

from jax.numpy import array
from jax.numpy import atleast_1d
from jax.numpy import sum

from noads.model import JAXModel
from noads.model import Model
from noads.models.energy.streams import Impact
from noads.models.energy.streams import Stream


class ProductionPathway:
    name: str

    impacts: set[Impact]
    """Direct impacts per unit of production."""

    input_streams: set[Stream]
    """Energy/material inputs for pathway production."""

    def __init__(
        self,
        name: str,
        impacts: Sequence[Impact],
        input_streams: Sequence[Stream],
    ):
        self.name = name
        self.impacts = set(impacts)
        self.input_streams = set(input_streams)

    @property
    def models(self) -> list[Model]:
        return [self.impact_index_model(), self.consumption_model()]

    def _impact_index(self, input_data):
        streams_impact_index = {
            impact.name: {
                stream.name: input_data[f"{stream.name}.{impact.name}_index"]
                for stream in self.input_streams
            } for impact in self.impacts
        }
        # input streams per production unit
        consumption_per_prod = {
            stream: 1 / input_data[f"{self.name}.{stream.name}.efficiency"]
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
        output_data.update(
            {
                f"{self.name}.indirect.{impact.name}_index":
                    sum(
                        array(
                            [
                                atleast_1d(
                                    stream_consumption *
                                    streams_impact_index[impact.name][stream.name]
                                )
                                for stream, stream_consumption in
                                consumption_per_prod.items()
                            ]
                        )
                    )
                for impact in self.impacts
            }
        )

        # Total impacts index
        output_data.update(
            {
                f"{self.name}.{impact.name}_index":
                    output_data[
                        f"{self.name}.indirect.{impact.name}_index"
                    ] + direct_impact
                for impact, direct_impact in direct_impact_per_prod.items()
            }
        )

        # Final consumption
        output_data.update(
            {
                f"{self.name}.{stream.name}.consumption_index":
                    stream_consumption
                for stream, stream_consumption in consumption_per_prod.items()
            }
        )

        return output_data

    def impact_index_model(self):
        default_values_units = {
            f"{self.name}.direct.{impact.name}_index":
                (0.0, f"{impact.unit}/")
            for impact in self.impacts
        }
        default_values_units.update({
            f"{self.name}.{input_stream.name}.efficiency":
                (1.0, f"/{input_stream.unit}")
            for input_stream in self.input_streams
        })
        output_units = {}
        for impact in self.impacts:
            output_units.update({
                f"{self.name}.{impact.name}_index": f"{impact.unit}/",
                f"{self.name}.indirect.{impact.name}_index":
                    f"{impact.unit}/",
            })
            for input_stream in self.input_streams:
                default_values_units.update({
                    f"{input_stream.name}.{impact.name}_index":
                        (0.0, f"{impact.unit}/{input_stream.unit}")
                })
                output_units.update({
                    f"{self.name}.{input_stream.name}.consumption_index":
                        f"{input_stream.unit}/"
                })
        return JAXModel(
            function=self._impact_index,
            input_names=list(default_values_units.keys()),
            output_names=list(output_units.keys()),
            default_inputs={
                name: value_unit[0] for name, value_unit in default_values_units.items()
            },
            name=f"{self.name} impacts",
        )

    def consumption_model(self):
        default_values_units = {f"{self.name}.production": (0.0, "")}
        default_values_units.update({
            f"{self.name}.{stream.name}.consumption_index":
                (0.0, f"{stream.unit}/")
            for stream in self.input_streams
        })
        output_units = {
            f"{self.name}.{stream.name}.consumption": stream.unit
            for stream in self.input_streams
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
            function=self._consumption,
            input_names=list(default_values_units.keys()),
            output_names=list(output_units.keys()),
            default_inputs={
                name: value_unit[0] for name, value_unit in default_values_units.items()
            },
            name=f"{self.name} production",
            # variable_full_names=fullnames,
            # variable_units=units,
        )

    def _consumption(self, input_data):
        production = input_data[f"{self.name}.production"]
        consumption_index = {
            stream.name: input_data[f"{self.name}.{stream.name}.consumption_index"]
            for stream in self.input_streams
        }
        return {
            f"{self.name}.{stream.name}.consumption":
                production * consumption_index[stream.name]
            for stream in self.input_streams
        }
