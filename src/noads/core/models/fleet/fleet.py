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
"""Fleet operations of a mix of aircraft."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo import generate_coupling_graph
from jax.numpy import array
from jax.numpy import power
from jax.numpy import sum as jnp_sum

from noads.core.model import JAXModel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from noads.core.models.energy.energy import EnergyCarrier
    from noads.core.models.fleet.aircraft_operation import AircraftOperation


class Fleet:
    """Fleet operation of a mix of competing aircraft."""

    name: str
    """Fleet name."""

    operating_aircraft: list[AircraftOperation]
    """List of operating aircraft within a fleet."""

    consumed_carriers: list[EnergyCarrier]
    """List of energy carriers consumed in the operation."""

    def __init__(
        self,
        name: str,
        operating_aircraft: Sequence[AircraftOperation],
    ):
        """Initialize Fleet."""
        self.name = name
        self.operating_aircraft = list(operating_aircraft)
        self.consumed_carriers = list({
            carrier
            for aircraft in operating_aircraft
            for carrier in aircraft.propulsion.energy_carrier_mix
        })

    @property
    def models(self):
        """List of models."""
        models = [
            self.demand_avoidance_model(),
            self.last_share_model(),
            self.consumption_model(),
            self.mean_consumption_impacts_model(),
            self.ask_model(),
        ]

        models.extend([
            model for aircraft in self.operating_aircraft for model in aircraft.models
        ])
        return models

    def demand_avoidance_model(self):
        """Demand avoidance model."""
        default_inputs = {
            f"{self.name}.demand_shift_ratio": 0.0,
            f"{self.name}.ask_trend": 0.0,
            "year": 2025.0,
            "discount_rate": 0.03,
            "start_year": 2025.0,
            "price_elasticity": -0.6,
        }
        output_names = [
            f"{self.name}.ask",
            f"{self.name}.ask_avoided",
            f"{self.name}.discounted_ask_avoided",
            f"{self.name}.relative_price_change",
            f"{self.name}.discounted_relative_price_change",
        ]

        return JAXModel(
            function=self._demand_avoidance,
            input_names=list(default_inputs.keys()),
            output_names=output_names,
            default_inputs=default_inputs,
            name=f"{self.name} Demand avoidance",
        )

    def _demand_avoidance(self, input_data):
        demand_shift_ratio = input_data[f"{self.name}.demand_shift_ratio"]
        ask_trend = input_data[f"{self.name}.ask_trend"]
        year = input_data["year"]
        discount_rate = input_data["discount_rate"]
        start_year = input_data["start_year"]
        price_elasticity = input_data["price_elasticity"]

        ask_avoided = ask_trend * demand_shift_ratio
        ask = ask_trend - ask_avoided

        # elast = (dASK/ASKtrend) / (dP/Ptrend)
        # dP/Ptrend = (-SR)/elast
        relative_price_change = power(
            1.0 - demand_shift_ratio,
            1.0 / price_elasticity,
        )

        discount_factor = (1 + discount_rate) ** (start_year - year)
        discounted_relative_price_change = relative_price_change * discount_factor
        discounted_ask_avoided = ask_avoided * discount_factor
        return {
            f"{self.name}.ask": ask,
            f"{self.name}.ask_avoided": ask_avoided,
            f"{self.name}.relative_price_change": relative_price_change,
            f"{self.name}.discounted_relative_price_change": discounted_relative_price_change,  # noqa: E501
            f"{self.name}.discounted_ask_avoided": discounted_ask_avoided,
        }

    def last_share_model(self):
        """Remaining market share of the last operating aircraft."""
        default_inputs = {}
        output_names = []
        for aircraft in self.operating_aircraft:
            if aircraft != self.operating_aircraft[0]:
                default_inputs[f"{aircraft.name}.share"] = 0.0
            else:
                output_names.append(f"{aircraft.name}.share")
        return JAXModel(
            function=self._last_share,
            input_names=list(default_inputs.keys()),
            output_names=output_names,
            default_inputs=default_inputs,
            name=f"{self.operating_aircraft[0].name} ASK share",
        )

    def _last_share(self, input_data):
        all_but_last_shares = array([
            input_data[f"{self.operating_aircraft[i + 1].name}.share"]
            for i in range(len(self.operating_aircraft) - 1)
        ])
        return {
            f"{self.operating_aircraft[0].name}.share": 1.0
            - jnp_sum(all_but_last_shares, axis=0)
        }

    def consumption_model(self):
        """Energy Carrier consumption model."""
        default_inputs = {
            f"{aircraft.name}.{carrier.name}.consumption": 0.0
            for aircraft in self.operating_aircraft
            for carrier in aircraft.propulsion.energy_carrier_mix
        }
        output_names = [
            f"{self.name}.{carrier.name}.consumption"
            for carrier in self.consumed_carriers
        ]
        return JAXModel(
            function=self._energy_consumption,
            input_names=list(default_inputs.keys()),
            output_names=output_names,
            default_inputs=default_inputs,
            name=f"{self.name} consumption",
        )

    def _energy_consumption(self, input_data):
        conso_per_carrier = {
            carrier.name: array([
                input_data[f"{aircraft.name}.{carrier.name}.consumption"]
                if carrier in aircraft.propulsion.energy_carrier_mix
                else 0.0
                for aircraft in self.operating_aircraft
            ])
            for carrier in self.consumed_carriers
        }
        return {
            f"{self.name}.{carrier.name}.consumption": jnp_sum(
                conso_per_carrier[carrier.name]
            )
            for carrier in self.consumed_carriers
        }

    def mean_consumption_impacts_model(self):
        """Mean Energy Carrier consumption and impacts within fleet."""
        default_inputs = {f"{self.name}.ask": 0.0}
        default_inputs.update({
            f"{self.name}.{carrier.name}.consumption": 0.0
            for carrier in self.consumed_carriers
        })
        output_names = [
            f"{self.name}.{carrier.name}.mean_consumption_per_ask"
            for carrier in self.consumed_carriers
        ]
        output_names.append(f"{self.name}.mean_energy_per_ask")
        return JAXModel(
            function=self._mean_consumption_impacts,
            input_names=list(default_inputs.keys()),
            output_names=output_names,
            default_inputs=default_inputs,
            name=f"{self.name} mean consumption and impacts",
        )

    def _mean_consumption_impacts(self, input_data):
        conso_per_carrier = array([
            input_data[f"{self.name}.{carrier.name}.consumption"]
            for carrier in self.consumed_carriers
        ])
        ask = input_data[f"{self.name}.ask"]
        output_data = {
            f"{self.name}.{carrier.name}.mean_consumption_per_ask": conso_per_carrier[i]
            / ask
            for i, carrier in enumerate(self.consumed_carriers)
        }
        output_data[f"{self.name}.mean_energy_per_ask"] = (
            jnp_sum(conso_per_carrier) / ask
        )

        return output_data

    def ask_model(self):
        """Mix of supply among aircraft."""
        default_inputs = {
            f"{aircraft.name}.share": 0.0 for aircraft in self.operating_aircraft
        }
        default_inputs[f"{self.name}.ask"] = 0.0
        output_names = [f"{aircraft.name}.ask" for aircraft in self.operating_aircraft]
        return JAXModel(
            function=self._ask_mix,
            input_names=list(default_inputs.keys()),
            output_names=output_names,
            default_inputs=default_inputs,
            name=f"{self.name} ASK mix",
        )

    def _ask_mix(self, input_data):
        ask = input_data[f"{self.name}.ask"]
        shares = array([
            input_data[f"{aircraft.name}.share"] for aircraft in self.operating_aircraft
        ])
        aircraft_ask = shares * ask
        return {
            f"{aircraft.name}.ask": aircraft_ask[i]
            for i, aircraft in enumerate(self.operating_aircraft)
        }


class FleetAssembly(Fleet):
    """Assembly of a mix of Fleet operations."""

    fleets: list[Fleet]
    """List of Fleets."""

    def __init__(
        self,
        fleets: Sequence[Fleet],
        plot_coupling_graph=False,
    ):
        """Initialize FleetAssembly."""
        self.fleets = list(fleets)
        super().__init__(
            name="Global fleet",
            operating_aircraft=[
                aircraft for fleet in fleets for aircraft in fleet.operating_aircraft
            ],
        )
        if plot_coupling_graph:
            self.plot_pretty_couplings()

    @property
    def models(self):
        """List of models."""
        models = [
            self.demand_avoidance_model(),
            self.last_share_model(),
            self.consumption_model(),
            self.mean_consumption_impacts_model(),
            self.ask_model(),
        ]
        models.extend([model for fleet in self.fleets for model in fleet.models])
        return models

    def plot_pretty_couplings(self):
        """Plot fleet coupling graph."""
        # traffic = AirTraffic()
        # assembly_models = [
        #     self.demand_avoidance_model(),
        #     self.last_share_model(),
        #     self.consumption_model(),
        #     self.mean_consumption_impacts_model(),
        #     self.ask_model(),
        #     traffic,
        # ]
        # assembly_discs = [
        #     JAXChain(
        #         [model.discipline for model in fleet.models],
        #         name=fleet.name,
        #     ) for fleet in self.fleets
        # ]
        # assembly_discs.extend([model.discipline for model in assembly_models])
        # # generate_coupling_graph(
        # #     assembly_discs, "fleet_assembly.pdf"
        # # )
        # assembly_discs.extend([
        #     model.discipline for fleet in self.fleets for model in fleet.models
        # ])
        # generate_coupling_graph(
        #     assembly_discs, "fleet.pdf"
        # )
        generate_coupling_graph(
            [model.discipline for model in self.fleets[0].models],
            f"{self.fleets[0].name}.pdf",
        )
        # for fleet in self.fleets:
        #     generate_coupling_graph(
        #         [model.discipline for model in fleet.models], f"{fleet.name}.pdf"
        #     )

    def demand_avoidance_model(self):
        """Aggregation of demand avoidance among Fleets."""
        default_inputs = {
            "ask_trend": 0.0,
            "year": 2025.0,
            "discount_rate": 0.03,
            "start_year": 2025.0,
            "load_factor": 100.0,
        }
        default_inputs.update({f"{fleet.name}.ask": 0.0 for fleet in self.fleets})
        default_inputs.update({
            f"{fleet.name}.relative_price_change": 0.0 for fleet in self.fleets
        })
        default_inputs.update({f"{fleet.name}.ask_trend": 0.0 for fleet in self.fleets})

        output_names = [
            "rpk",
            "ask",
            "ask_avoided",
            "discounted_ask_avoided",
            "discounted_ask_ratio",
            "discount_factor",
            "relative_price_change",
            "discounted_relative_price_change",
            # "avoidance_burden",
            # "discounted_avoidance_burden",
        ]
        return JAXModel(
            function=self._demand_avoidance,
            input_names=list(default_inputs.keys()),
            output_names=output_names,
            default_inputs=default_inputs,
            name="Global Demand avoidance",
        )

    def _demand_avoidance(self, input_data):
        ask_trend = input_data["ask_trend"]
        year = input_data["year"]
        discount_rate = input_data["discount_rate"]
        start_year = input_data["start_year"]
        load_factor = input_data["load_factor"]

        asks = array([input_data[f"{fleet.name}.ask"] for fleet in self.fleets])
        asks_trend = array([
            input_data[f"{fleet.name}.ask_trend"] for fleet in self.fleets
        ])
        price_changes = array([
            input_data[f"{fleet.name}.relative_price_change"] for fleet in self.fleets
        ])
        # shift_ratios = array(
        #     [
        #         input_data[f"{fleet.name}.demand_shift_ratio"] for fleet in
        #         self.fleets
        #     ]
        # )
        # trend_shares = array(
        #     [
        #         input_data[f"{fleet.name}.share"] for fleet in
        #         self.fleets
        #     ]
        # )

        ask = jnp_sum(asks)
        ask_avoided = ask_trend - ask

        rpk = ask * load_factor * 1e-2

        relative_price_change = jnp_sum(asks_trend * price_changes) / ask_trend
        # avoidance_burden = jnp_sum(trend_shares * shift_ratios)

        discount_factor = (1 + discount_rate) ** (start_year - year)
        discounted_relative_price_change = relative_price_change * discount_factor

        discounted_ask_avoided = ask_avoided * discount_factor
        discounted_ask_ratio = ask_avoided * discount_factor / ask_trend

        # discounted_avoidance_burden = avoidance_burden * discount_factor

        return {
            "rpk": rpk,
            "ask": ask,
            "ask_avoided": ask_avoided,
            "discounted_ask_avoided": discounted_ask_avoided,
            "discounted_ask_ratio": discounted_ask_ratio,
            "discount_factor": discount_factor,
            "relative_price_change": relative_price_change,
            "discounted_relative_price_change": discounted_relative_price_change,
            # "avoidance_burden": avoidance_burden,
            # "discounted_avoidance_burden": discounted_avoidance_burden,
        }

    def last_share_model(self):
        """Remaining ASK share of the last Fleet."""
        default_inputs = {}
        output_names = []
        for fleet in self.fleets:
            if fleet != self.fleets[0]:
                default_inputs[f"{fleet.name}.share"] = 0.0
            else:
                output_names.append(f"{fleet.name}.share")
        return JAXModel(
            function=self._last_share,
            input_names=list(default_inputs.keys()),
            output_names=output_names,
            default_inputs=default_inputs,
            name=f"{self.fleets[0].name} Global share",
        )

    def _last_share(self, input_data):
        all_but_last_shares = array([
            input_data[f"{self.fleets[i + 1].name}.share"]
            for i in range(len(self.fleets) - 1)
        ])
        return {f"{self.fleets[0].name}.share": 1.0 - jnp_sum(all_but_last_shares)}

    def consumption_model(self):
        """Energy Carrier consumption model."""
        default_inputs = {
            f"{fleet.name}.{carrier.name}.consumption": 0.0
            for fleet in self.fleets
            for carrier in fleet.consumed_carriers
        }
        output_names = [
            f"{carrier.name}.consumption" for carrier in self.consumed_carriers
        ]
        return JAXModel(
            function=self._energy_consumption,
            input_names=list(default_inputs.keys()),
            output_names=output_names,
            default_inputs=default_inputs,
            name="Global consumption",
        )

    def _energy_consumption(self, input_data):
        conso_per_carrier = {
            carrier.name: array([
                input_data[f"{fleet.name}.{carrier.name}.consumption"]
                if carrier in fleet.consumed_carriers
                else 0.0
                for fleet in self.fleets
            ])
            for carrier in self.consumed_carriers
        }
        return {
            f"{carrier.name}.consumption": jnp_sum(conso_per_carrier[carrier.name])
            for carrier in self.consumed_carriers
        }

    def ask_model(self):
        """Mix of supply among Fleets."""
        default_inputs = {f"{fleet.name}.share": 0.0 for fleet in self.fleets}
        default_inputs["ask_trend"] = 0.0
        output_names = [f"{fleet.name}.ask_trend" for fleet in self.fleets]
        return JAXModel(
            function=self._ask_mix,
            input_names=list(default_inputs.keys()),
            output_names=output_names,
            default_inputs=default_inputs,
            name="Global ASK trend",
        )

    def _ask_mix(self, input_data):
        ask = input_data["ask_trend"]
        shares = array([input_data[f"{fleet.name}.share"] for fleet in self.fleets])
        fleet_ask = shares * ask
        return {
            f"{fleet.name}.ask_trend": fleet_ask[i]
            for i, fleet in enumerate(self.fleets)
        }

    def mean_consumption_impacts_model(self):
        """Mean Energy Carrier consumption and impacts within all fleets."""
        default_inputs = {
            f"{carrier.name}.consumption": 0.0 for carrier in self.consumed_carriers
        }
        default_inputs["ask"] = 0.0
        output_names = [
            f"{carrier.name}.mean_consumption_per_ask"
            for carrier in self.consumed_carriers
        ]
        output_names.append("mean_energy_per_ask")
        return JAXModel(
            function=self._mean_consumption_impacts,
            input_names=list(default_inputs.keys()),
            output_names=output_names,
            default_inputs=default_inputs,
            name=f"{self.name} mean consumption",
        )

    def _mean_consumption_impacts(self, input_data):
        conso_per_carrier = array([
            input_data[f"{carrier.name}.consumption"]
            for carrier in self.consumed_carriers
        ])
        ask = input_data["ask"]
        output_data = {
            f"{carrier.name}.mean_consumption_per_ask": conso_per_carrier[i] / ask
            for i, carrier in enumerate(self.consumed_carriers)
        }
        output_data["mean_energy_per_ask"] = jnp_sum(conso_per_carrier) / ask

        return output_data
