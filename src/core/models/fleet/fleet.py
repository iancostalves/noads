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

from jax.numpy import array
from jax.numpy import sum

from core.model import JAXModel
from core.model import Model
from core.models.energy.energy import Energy
from core.models.energy.energy import EnergyCarrier
from core.models.fleet.aircraft_design import AircraftDesign

if TYPE_CHECKING:
    from collections.abc import Sequence

    from core.models.fleet.aircraft_operation import AircraftOperation


class Fleet:
    name: str

    operating_aircraft: list[AircraftOperation | AircraftDesign]

    consumed_carriers: list[EnergyCarrier]

    models: list[Model]

    def __init__(
        self,
        name: str,
        operating_aircraft: Sequence[AircraftOperation],
    ):
        self.name = name
        self.operating_aircraft = list(operating_aircraft)
        self.consumed_carriers = list({
            carrier
            for aircraft in operating_aircraft
            for carrier in aircraft.propulsion.energy_carrier_mix
        })

    @property
    def models(self):
        models = [
            self.demand_avoidance_model(),
            self.last_share_model(),
            self.consumption_model(),
            self.mean_consumption_impacts_model(),
            self.ask_model(),
            # self.overshoot_constraint(),
        ]

        models.extend([
            model
            for aircraft in self.operating_aircraft
            for model in aircraft.models
            # if aircraft != self.operating_aircraft[0]
        ])
        return models

    # def overshoot_constraint(self):
    #     default_inputs = {
    #         f"{aircraft.name}.max_share": 0.0 for aircraft in self.operating_aircraft
    #         if aircraft != self.operating_aircraft[0]
    #     }
    #     output_names = [f"{self.name}.sum_max_shares"]
    #     return JAXModel(
    #         function=self.__overshoot_max_shares,
    #         input_names=list(default_inputs.keys()),
    #         output_names=output_names,
    #         default_inputs=default_inputs,
    #         name=f"{self.name} aircraft overshoot",
    #     )
    #
    # def __overshoot_max_shares(self, input_data):
    #     shares = array([
    #         input_data[f"{aircraft.name}.max_share"]
    #         for aircraft in self.operating_aircraft
    #         if aircraft != self.operating_aircraft[0]
    #     ])
    #     return {f"{self.name}.sum_max_shares": sum(shares, axis=0)}

    # def overshoot_constraint(self):
    #     default_inputs = {f"{self.operating_aircraft[0].name}.share": 0.0}
    #     output_names = [f"{self.name}.overshoot"]
    #     return JAXModel(
    #         function=self.__overshoot_max_shares,
    #         input_names=list(default_inputs.keys()),
    #         output_names=output_names,
    #         default_inputs=default_inputs,
    #         name=f"{self.name} aircraft overshoot",
    #     )
    #
    # def __overshoot_max_shares(self, input_data):
    #     share_1st_aircraft = input_data[f"{self.operating_aircraft[0].name}.share"]
    #     return {
    #         f"{self.name}.overshoot":
    #             where(share_1st_aircraft < 0, -share_1st_aircraft, 0.0)
    #     }

    def demand_avoidance_model(self):
        default_values_units = {
            f"{self.name}.demand_shift_ratio": (0.0, ""),
            f"{self.name}.ask_trend": (0.0, "pax km"),
            "year": (2025.0, "year"),
            "discount_rate": (0.03, ""),
            "start_year": (2025.0, "year"),
            "price_elasticity": (-0.6, ""),
        }
        output_units = {
            f"{self.name}.ask": "pax km",
            f"{self.name}.ask_avoided": "pax km",
            f"{self.name}.discounted_ask_avoided": "pax km",
            f"{self.name}.relative_price_change": "",
            f"{self.name}.discounted_relative_price_change": "",
        }

        variables = set(default_values_units.keys()).union(output_units)
        {name: name.replace(".", " ").replace("_", " ") for name in variables}
        {
            name: default_values_units[name][1]
            if name in default_values_units
            else output_units[name]
            for name in variables
        }
        return JAXModel(
            function=self._demand_avoidance,
            input_names=list(default_values_units.keys()),
            output_names=list(output_units.keys()),
            default_inputs={
                name: value_unit[0] for name, value_unit in default_values_units.items()
            },
            name=f"{self.name} Demand avoidance",
            # variable_full_names=fullnames,
            # variable_units=units,
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
        relative_price_change = -demand_shift_ratio / price_elasticity

        discount_factor = (1 + discount_rate) ** (start_year - year)
        discounted_relative_price_change = relative_price_change * discount_factor
        discounted_ask_avoided = ask_avoided * discount_factor
        return {
            f"{self.name}.ask": ask,
            f"{self.name}.ask_avoided": ask_avoided,
            # f"{self.name}.discount_factor": discount_factor,
            f"{self.name}.relative_price_change": relative_price_change,
            f"{self.name}.discounted_relative_price_change": discounted_relative_price_change,
            f"{self.name}.discounted_ask_avoided": discounted_ask_avoided,
        }

    def last_share_model(self):
        default_values_units = {}
        output_units = {}
        for aircraft in self.operating_aircraft:
            if aircraft != self.operating_aircraft[0]:
                default_values_units[f"{aircraft.name}.share"] = (0.0, "")
            else:
                output_units[f"{aircraft.name}.share"] = ""
        variables = set(default_values_units.keys()).union(output_units)
        {name: name.replace(".", " ").replace("_", " ") for name in variables}
        {
            name: default_values_units[name][1]
            if name in default_values_units
            else output_units[name]
            for name in variables
        }
        return JAXModel(
            function=self._last_share,
            input_names=list(default_values_units.keys()),
            output_names=list(output_units.keys()),
            default_inputs={
                name: value_unit[0] for name, value_unit in default_values_units.items()
            },
            name=f"{self.operating_aircraft[0].name} ASK share",
            # variable_full_names=fullnames,
            # variable_units=units,
        )

    def _last_share(self, input_data):
        all_but_last_shares = array([
            input_data[f"{self.operating_aircraft[i + 1].name}.share"]
            for i in range(len(self.operating_aircraft) - 1)
        ])
        return {
            f"{self.operating_aircraft[0].name}.share": 1.0
            - sum(all_but_last_shares, axis=0)
        }

    def consumption_model(self):
        default_values_units = {
            f"{aircraft.name}.{carrier.name}.consumption": (0.0, carrier.unit)
            for aircraft in self.operating_aircraft
            for carrier in aircraft.propulsion.energy_carrier_mix
        }
        output_units = {
            f"{self.name}.{carrier.name}.consumption": carrier.unit
            for carrier in self.consumed_carriers
        }
        variables = set(default_values_units.keys()).union(output_units)
        {name: name.replace(".", " ").replace("_", " ") for name in variables}
        {
            name: default_values_units[name][1]
            if name in default_values_units
            else output_units[name]
            for name in variables
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
            f"{self.name}.{carrier.name}.consumption": sum(
                conso_per_carrier[carrier.name]
            )
            for carrier in self.consumed_carriers
        }

    def mean_consumption_impacts_model(self):
        default_values_units = {
            f"{self.name}.{carrier.name}.consumption": (0.0, carrier.unit)
            for carrier in self.consumed_carriers
        }
        default_values_units[f"{self.name}.ask"] = (0.0, "pax km")
        output_units = {
            f"{self.name}.{carrier.name}.mean_consumption_per_ask": f"{carrier.unit}/ pax km"
            for carrier in self.consumed_carriers
        }
        output_units[f"{self.name}.mean_energy_per_ask"] = f"{Energy.unit}/ pax km"
        variables = set(default_values_units.keys()).union(output_units)
        {name: name.replace(".", " ").replace("_", " ") for name in variables}
        {
            name: default_values_units[name][1]
            if name in default_values_units
            else output_units[name]
            for name in variables
        }
        return JAXModel(
            function=self._mean_consumption_impacts,
            input_names=list(default_values_units.keys()),
            output_names=list(output_units.keys()),
            default_inputs={
                name: value_unit[0] for name, value_unit in default_values_units.items()
            },
            name=f"{self.name} mean consumption and impacts",
            # variable_full_names=fullnames,
            # variable_units=units,
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
        output_data[f"{self.name}.mean_energy_per_ask"] = sum(conso_per_carrier) / ask

        return output_data

    def ask_model(self):
        default_values_units = {
            f"{aircraft.name}.share": (0.0, "") for aircraft in self.operating_aircraft
        }
        default_values_units[f"{self.name}.ask"] = (0.0, "pax km")
        output_units = {
            f"{aircraft.name}.ask": "pax km" for aircraft in self.operating_aircraft
        }
        variables = set(default_values_units.keys()).union(output_units)
        {name: name.replace(".", " ").replace("_", " ") for name in variables}
        {
            name: default_values_units[name][1]
            if name in default_values_units
            else output_units[name]
            for name in variables
        }
        return JAXModel(
            function=self._ask_mix,
            input_names=list(default_values_units.keys()),
            output_names=list(output_units.keys()),
            default_inputs={
                name: value_unit[0] for name, value_unit in default_values_units.items()
            },
            name=f"{self.name} ASK mix",
            # variable_full_names=fullnames,
            # variable_units=units,
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
    design_aircraft: bool

    fleets: list[Fleet]

    def __init__(
        self,
        fleets: Sequence[Fleet],
        design_aircraft: bool = True,
    ):
        self.fleets = list(fleets)
        self.design_aircraft = design_aircraft
        super().__init__(
            name="Global fleet",
            operating_aircraft=[
                aircraft for fleet in fleets for aircraft in fleet.operating_aircraft
            ],
        )
        if self.design_aircraft:
            for aircraft in self.operating_aircraft:
                if aircraft not in [
                    fleet.operating_aircraft[0] for fleet in self.fleets
                ]:
                    assert isinstance(aircraft, AircraftDesign)

    @property
    def models(self):
        models = [
            self.demand_avoidance_model(),
            self.last_share_model(),
            self.consumption_model(),
            self.mean_consumption_impacts_model(),
            self.ask_model(),
        ]
        models.extend([model for fleet in self.fleets for model in fleet.models])
        # if self.design_aircraft:
        #     models.append(self.feasible_designs())
        return models

    # def feasible_designs(self):
    #     default_inputs = {
    #         f"{aircraft.name}.energy_per_ask_relative": 1.0
    #         for aircraft in self.operating_aircraft if aircraft not in [
    #             fleet.operating_aircraft[0] for fleet in self.fleets
    #         ]
    #     }
    #     output_names = [
    #         "minimal_energy_relative.constraint", "maximal_energy_relative.constraint"
    #     ]
    #
    #     return JAXModel(
    #         function=self.__feasible_fleet,
    #         input_names=list(default_inputs.keys()),
    #         output_names=output_names,
    #         default_inputs=default_inputs,
    #         name="Feasible aircraft",
    #     )
    #
    # def __feasible_fleet(self, input_data):
    #     energy_per_ask_relative = array([
    #         atleast_1d(input_data[f"{aircraft.name}.energy_per_ask_relative"])
    #         for aircraft in self.operating_aircraft if aircraft not in [
    #             fleet.operating_aircraft[0] for fleet in self.fleets
    #         ]
    #     ])
    #
    #     min_mass_ratio = min(energy_per_ask_relative, axis=0)
    #     max_mass_ratio = max(energy_per_ask_relative, axis=0)
    #     return {
    #         "minimal_energy_relative.constraint": min_mass_ratio,
    #         "maximal_energy_relative.constraint": max_mass_ratio,
    #     }

    def demand_avoidance_model(self):
        default_values_units = {
            "ask_trend": (0.0, "pax km"),
            "year": (2025.0, "year"),
            "discount_rate": (0.03, ""),
            "start_year": (2025.0, "year"),
            "load_factor": (100.0, ""),
        }
        default_values_units.update({
            f"{fleet.name}.ask": (0.0, "pax km") for fleet in self.fleets
        })
        default_values_units.update({
            f"{fleet.name}.relative_price_change": (0.0, "") for fleet in self.fleets
        })
        # default_values_units.update(
        #     {
        #         f"{fleet.name}.demand_shift_ratio": (0.0, "") for fleet in
        #         self.fleets
        #     }
        # )
        # default_values_units.update(
        #     {
        #         f"{fleet.name}.share": (0.0, "") for fleet in
        #         self.fleets
        #     }
        # )

        output_units = {
            "rpk": "pax km",
            "ask": "pax km",
            "ask_avoided": "pax km",
            "discounted_ask_avoided": "pax km",
            "discounted_ask_ratio": "",
            "discount_factor": "",
            "relative_price_change": "",
            "discounted_relative_price_change": "",
            # "avoidance_burden": "",
            # "discounted_avoidance_burden": "",
        }

        variables = set(default_values_units.keys()).union(output_units)
        {name: name.replace(".", " ").replace("_", " ") for name in variables}
        {
            name: default_values_units[name][1]
            if name in default_values_units
            else output_units[name]
            for name in variables
        }
        return JAXModel(
            function=self._demand_avoidance,
            input_names=list(default_values_units.keys()),
            output_names=list(output_units.keys()),
            default_inputs={
                name: value_unit[0] for name, value_unit in default_values_units.items()
            },
            name="Global Demand avoidance",
            # variable_full_names=fullnames,
            # variable_units=units,
        )

    def _demand_avoidance(self, input_data):
        ask_trend = input_data["ask_trend"]
        year = input_data["year"]
        discount_rate = input_data["discount_rate"]
        start_year = input_data["start_year"]
        load_factor = input_data["load_factor"]

        asks = array([input_data[f"{fleet.name}.ask"] for fleet in self.fleets])
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

        ask = sum(asks)
        ask_avoided = ask_trend - ask

        rpk = ask * load_factor * 1e-2

        relative_price_change = sum(asks * price_changes) / ask
        # avoidance_burden = sum(trend_shares * shift_ratios)

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
        default_values_units = {}
        output_units = {}
        for fleet in self.fleets:
            if fleet != self.fleets[0]:
                default_values_units[f"{fleet.name}.share"] = (0.0, "")
            else:
                output_units[f"{fleet.name}.share"] = ""
        variables = set(default_values_units.keys()).union(output_units)
        {name: name.replace(".", " ").replace("_", " ") for name in variables}
        {
            name: default_values_units[name][1]
            if name in default_values_units
            else output_units[name]
            for name in variables
        }
        return JAXModel(
            function=self._last_share,
            input_names=list(default_values_units.keys()),
            output_names=list(output_units.keys()),
            default_inputs={
                name: value_unit[0] for name, value_unit in default_values_units.items()
            },
            name=f"{self.fleets[0].name} Global share",
            # variable_full_names=fullnames,
            # variable_units=units,
        )

    def _last_share(self, input_data):
        all_but_last_shares = array([
            input_data[f"{self.fleets[i + 1].name}.share"]
            for i in range(len(self.fleets) - 1)
        ])
        return {f"{self.fleets[0].name}.share": 1 - sum(all_but_last_shares)}

    def consumption_model(self):
        default_values_units = {
            f"{fleet.name}.{carrier.name}.consumption": (0.0, carrier.unit)
            for fleet in self.fleets
            for carrier in fleet.consumed_carriers
        }
        output_units = {
            f"{carrier.name}.consumption": carrier.unit
            for carrier in self.consumed_carriers
        }
        variables = set(default_values_units.keys()).union(output_units)
        {name: name.replace(".", " ").replace("_", " ") for name in variables}
        {
            name: default_values_units[name][1]
            if name in default_values_units
            else output_units[name]
            for name in variables
        }
        return JAXModel(
            function=self._energy_consumption,
            input_names=list(default_values_units.keys()),
            output_names=list(output_units.keys()),
            default_inputs={
                name: value_unit[0] for name, value_unit in default_values_units.items()
            },
            name="Global consumption",
            # variable_full_names=fullnames,
            # variable_units=units,
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
            f"{carrier.name}.consumption": sum(conso_per_carrier[carrier.name])
            for carrier in self.consumed_carriers
        }

    def ask_model(self):
        default_values_units = {
            f"{fleet.name}.share": (0.0, "") for fleet in self.fleets
        }
        default_values_units["ask_trend"] = (0.0, "pax km")
        output_units = {f"{fleet.name}.ask_trend": "pax km" for fleet in self.fleets}
        variables = set(default_values_units.keys()).union(output_units)
        {name: name.replace(".", " ").replace("_", " ") for name in variables}
        {
            name: default_values_units[name][1]
            if name in default_values_units
            else output_units[name]
            for name in variables
        }
        return JAXModel(
            function=self._ask_mix,
            input_names=list(default_values_units.keys()),
            output_names=list(output_units.keys()),
            default_inputs={
                name: value_unit[0] for name, value_unit in default_values_units.items()
            },
            name="Global ASK trend",
            # variable_full_names=fullnames,
            # variable_units=units,
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
        default_values_units = {
            f"{carrier.name}.consumption": (0.0, carrier.unit)
            for carrier in self.consumed_carriers
        }
        default_values_units["ask"] = (0.0, "pax km")
        output_units = {
            f"{carrier.name}.mean_consumption_per_ask": f"{carrier.unit}/pax km"
            for carrier in self.consumed_carriers
        }
        output_units["mean_energy_per_ask"] = f"{Energy.unit}/pax km"
        variables = set(default_values_units.keys()).union(output_units)
        {name: name.replace(".", " ").replace("_", " ") for name in variables}
        {
            name: default_values_units[name][1]
            if name in default_values_units
            else output_units[name]
            for name in variables
        }
        return JAXModel(
            function=self._mean_consumption_impacts,
            input_names=list(default_values_units.keys()),
            output_names=list(output_units.keys()),
            default_inputs={
                name: value_unit[0] for name, value_unit in default_values_units.items()
            },
            name=f"{self.name} mean consumption",
            # variable_full_names=fullnames,
            # variable_units=units,
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
        output_data["mean_energy_per_ask"] = sum(conso_per_carrier) / ask

        return output_data
