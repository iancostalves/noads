from __future__ import annotations

from collections.abc import Mapping

from core.model import JAXModel
from core.model import Model
from core.models.energy.energy import EnergyCarrier

from jax.numpy import log
from jax.numpy import exp
from jax.numpy import where
from jax.numpy import zeros


class PropulsionSystem:
    name: str

    energy_carrier_mix: Mapping[EnergyCarrier: float]

    def __init__(self, name: str, energy_carrier_mix: Mapping[EnergyCarrier: float]):
        self.name = name
        self.energy_carrier_mix = energy_carrier_mix


class AircraftOperation:
    name: str

    propulsion: PropulsionSystem

    energy_per_ask: float

    lifetime: float

    recent: bool

    models: list[Model]

    def __init__(
        self,
        name: str,
        propulsion: PropulsionSystem,
        energy_per_ask: float = 0.87,
        lifetime: float = 25.0,
        recent: bool = False,
    ):
        self.name = name
        self.propulsion = propulsion
        self.energy_per_ask = energy_per_ask
        self.lifetime = lifetime
        self.recent = recent

    @property
    def models(self):
        if self.recent:
            return [self.consumption_model()]
        return [self.share_model(), self.consumption_model()]

    def _sigmoid_fleet_share(self, years_since_introduction, max_share, cut_below=2):
        growth_rate = log(100 / cut_below - 1) / (self.lifetime / 2)
        year_inflection = 0 if self.recent else self.lifetime / 2
        calc_share = 1e2 * max_share / (1 + exp(-growth_rate * (years_since_introduction - year_inflection)))
        calc_share_max = 100 / (1 + exp(-growth_rate * (years_since_introduction - year_inflection)))
        share = 1e-2 * where(
            calc_share_max < cut_below, zeros(calc_share.shape), calc_share
        )
        return share

    def _fleet_share(self, input_data):
        year = input_data["year"]
        entry_into_service = input_data[f"{self.name}.entry_into_service"]
        max_share = input_data[f"{self.name}.max_share"]
        return {f"{self.name}.share": self._sigmoid_fleet_share(
            year - entry_into_service, max_share
        )}

    def share_model(self):
        default_values_units = {
            "year": (2025.0, "year"),
            f"{self.name}.entry_into_service": (2035.0, "year"),
            f"{self.name}.max_share": (0.0, ""),
        }
        output_units = {f"{self.name}.share": ""}
        variables = set(default_values_units.keys()).union(output_units)
        fullnames = {
            name: name.replace(".", " ").replace("_", " ") for name in variables
        }
        units = {
            name: default_values_units[name][1] if name in default_values_units.keys()
            else output_units[name] for name in variables
        }
        return JAXModel(
            function=self._fleet_share,
            input_names=list(default_values_units.keys()),
            output_names=list(output_units.keys()),
            default_inputs={
                name: value_unit[0] for name, value_unit in default_values_units.items()
            },
            name=f"{self.name} ASK share",
            # variable_full_names=fullnames,
            # variable_units=units,
        )

    def _energy_consumption(self, input_data):
        ask = input_data[f"{self.name}.ask"]
        return {
            f"{self.name}.{carrier.name}.consumption":
                carrier_ratio * ask * self.energy_per_ask
            for carrier, carrier_ratio in self.propulsion.energy_carrier_mix.items()
        }

    def consumption_model(self):
        default_values_units = {f"{self.name}.ask": (0.0, "pax km")}
        output_units = {
            f"{self.name}.{carrier.name}.consumption": carrier.unit
            for carrier in self.propulsion.energy_carrier_mix.keys()
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
            function=self._energy_consumption,
            input_names=list(default_values_units.keys()),
            output_names=list(output_units.keys()),
            default_inputs={
                name: value_unit[0] for name, value_unit in default_values_units.items()
            },
            name=f"{self.name} consumption",
            # variable_full_names=fullnames,
            # variable_units=units,
        )
