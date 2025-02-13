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

from core.model import JAXModel
from core.models.fleet.aircraft_operation import AircraftOperation
from gam_jax.models.generic_airplane_model import GAM

if TYPE_CHECKING:
    from collections.abc import Mapping

    from core.models.fleet.aircraft_tech_parameter import AircraftTechParameter


class AircraftDesign(AircraftOperation):
    reference_aircraft: AircraftOperation

    mission: Mapping[str, str | float]

    power_system: Mapping[str, str | float]

    technology_evolution: list[AircraftTechParameter]

    def __init__(
        self,
        name,
        propulsion,
        mission,
        power_system,
        aircraft_tech_params,
        reference_aircraft,
    ):
        self.reference_aircraft = reference_aircraft
        self.mission = mission
        self.power_system = power_system
        self.technology_evolution = aircraft_tech_params
        super().__init__(name, propulsion, 1.5)

    @property
    def models(self):
        return [self.consumption_model(), self.control_model(), self.design_model()]

    def design_model(self):
        default_inputs = {f"{self.name}.entry_into_service": 2035.0}
        output_names = [
            f"{self.name}.energy_per_ask",
            f"{self.name}.mission_energy",
            f"{self.name}.energy_mass",
            f"{self.name}.mtow",
            f"{self.name}.owe",
            f"{self.name}.relative_efficiency_gain",
        ]
        output_names.extend([
            f"{self.name}.{tech_param.name}" for tech_param in self.technology_evolution
        ])

        return JAXModel(
            function=self._gam_design,
            input_names=list(default_inputs.keys()),
            output_names=output_names,
            default_inputs=default_inputs,
            name=f"{self.name} design",
        )

    def _gam_design(self, input_data):
        eis = input_data[f"{self.name}.entry_into_service"]
        tech_params = {
            tech_param.name: tech_param.value_at_entry_into_service(eis)
            for tech_param in self.technology_evolution
        }
        output_data = {
            f"{self.name}.{tech_param_name}": value
            for tech_param_name, value in tech_params.items()
        }
        model = GAM(**tech_params)
        final_design = model.design_airplane(self.power_system, self.mission)

        output_data.update({
            f"{self.name}.energy_per_ask": final_design["enrg_consumption"] * 1.0e-3,
            # J/pax-m -> MJ/pax-km
            f"{self.name}.mission_energy": final_design["mission_enrg"] * 1.0e-6,
            f"{self.name}.energy_mass": final_design["mission_fuel"],
            f"{self.name}.mtow": final_design["mtow"],
            f"{self.name}.owe": final_design["owe"],
            f"{self.name}.relative_efficiency_gain": self.reference_aircraft.energy_per_ask
            / (final_design["enrg_consumption"] * 1.0e-3),
        })
        return output_data

    def control_model(self):
        default_inputs = {
            "start_year": 2025.0,
            f"{self.name}.ramp_up_duration": 5.0,
            f"{self.name}.ramp_down_duration": 5.0,
            f"{self.name}.entry_into_service": 2030.0,
            f"{self.name}.max_share": 0.0,
            f"{self.name}.lifetime": 25.0,
        }
        output_names = [
            f"control.{self.name}.share.times",
            f"control.{self.name}.share.values",
        ]

        return JAXModel(
            function=self._control_share,
            input_names=list(default_inputs.keys()),
            output_names=output_names,
            default_inputs=default_inputs,
            name=f"{self.name} control",
        )

    def _control_share(self, input_data):
        ramp_up = input_data[f"{self.name}.ramp_up_duration"]
        ramp_down = input_data[f"{self.name}.ramp_down_duration"]
        eis = input_data[f"{self.name}.entry_into_service"]
        lifetime = input_data[f"{self.name}.lifetime"]
        max_share = input_data[f"{self.name}.max_share"]

        control_times = (
            array([1.0, 1.0, 1.0, 1.0]) * eis
            + array([0.0, 1.0, 0.0, 0.0]) * ramp_up
            + array([0.0, 0.0, 1.0, 1.0]) * lifetime
            + array([0.0, 0.0, 0.0, 1.0]) * ramp_down
        )
        control_values = array([0.0, 1.0, 1.0, 0.0]) * max_share
        return {
            f"control.{self.name}.share.times": control_times,
            f"control.{self.name}.share.values": control_values,
        }

    def _energy_consumption(self, input_data):
        ask = input_data[f"{self.name}.ask"]
        energy_per_ask = input_data[f"{self.name}.energy_per_ask"]
        return {
            f"{self.name}.{carrier.name}.consumption": carrier_ratio
            * ask
            * energy_per_ask
            for carrier, carrier_ratio in self.propulsion.energy_carrier_mix.items()
        }

    def consumption_model(self):
        default_inputs = {f"{self.name}.ask": 0.0, f"{self.name}.energy_per_ask": 1.0}
        output_names = [
            f"{self.name}.{carrier.name}.consumption"
            for carrier in self.propulsion.energy_carrier_mix
        ]

        return JAXModel(
            function=self._energy_consumption,
            input_names=list(default_inputs.keys()),
            output_names=output_names,
            default_inputs=default_inputs,
            name=f"{self.name} consumption",
        )
