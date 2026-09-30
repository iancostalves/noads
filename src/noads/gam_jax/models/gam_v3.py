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
"""JAX port of the Generic Airplane Model V3.0 (design path only).

Port of ``gam/gam_v3.py`` from https://gitlab.com/m6029/genericairplanemodel
(CADO team, ENAC: Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT,
Yri-Amandine KAMBIRI). Only the airplane sizing path is ported: the cost model,
off-design missions, payload-range and network plug-in are left out.

This is the aircraft model of the NOADS "update" configuration. The paper
configuration keeps using the frozen V2.0 port in ``generic_airplane_model.py``.
Compared with upstream, the constructor exposes the technology parameters used by
NOADS, and the MTOW is solved with a Newton method so that designs are
differentiable with respect to the technology parameters.
"""

# ruff: noqa: E501

from __future__ import annotations

from jax import config
from jax.numpy import array
from jax.numpy import exp
from jax.numpy import interp
from jax.numpy import minimum
from jax.numpy import sqrt
from optimistix import Newton
from optimistix import root_find

from noads.gam_jax.utils import physical_data as phd
from noads.gam_jax.utils import unit

config.update("jax_enable_x64", True)

FUEL_DENSITY = {
    "kerosene": 803.0,
    "petrol": 803.0,
    "e_fuel": 803.0,
    "gasoline": 800.0,
    "liquid_h2": 70.8,
    "liquid_ch4": 422.6,
    "liquid_nh3": 681.0,
    "solid_nh3": 817.0,
    "battery": 2800.0,
}
"""Reference fuel densities (kg/m3), GAM V3.0."""

FUEL_HEAT = {
    "kerosene": 43.1e6,
    "petrol": 43.1e6,
    "e_fuel": 43.1e6,
    "gasoline": 46.41e6,
    "liquid_h2": 121.0e6,
    "compressed_h2": 121.0e6,
    "liquid_ch4": 50.3e6,
    "liquid_nh3": 16.89e6,
    "solid_nh3": 16.89e6,
}
"""Reference fuel lower heating values (J/kg), GAM V3.0."""


def fuel_density(fuel_type, press=101325.0):
    """Reference fuel density (kg/m3)."""
    if fuel_type == "compressed_h2":
        p = press * 1.0e-5
        return (-3.11480362e-05 * p + 7.82320891e-02) * p + 1.03207822e-01
    if fuel_type not in FUEL_DENSITY:
        msg = f"fuel_type {fuel_type} is unknown"
        raise ValueError(msg)
    return FUEL_DENSITY[fuel_type]


def fuel_heat(fuel_type):
    """Reference fuel lower heating value (J/kg)."""
    if fuel_type not in FUEL_HEAT:
        msg = f"fuel_type {fuel_type} is unknown"
        raise ValueError(msg)
    return FUEL_HEAT[fuel_type]


class GAM:
    """Generic Airplane Model V3.0: sizing of an airplane from statistical regressions.

    The design procedure balances the operating empty weight obtained from the
    mission (MTOW - payload - fuel) with the one obtained from the mass
    regressions, for the MTOW.
    """

    def __init__(
        self,
        battery_specific_energy=250.0,
        emotor_specific_power=4.5,
        lh2tank_gravimetric_index=40.0,
        fuelcell_specific_power=1.0,
        lift_to_drag="model",
        struct_weight_factor=100.0,
        fuelcell_efficiency=50.0,
        electronics_specific_power=10.0,
    ):
        """Initialize GAM with the technology parameters.

        The defaults are the upstream GAM V3.0 values.

        Args:
            battery_specific_energy: Battery specific energy (Wh/kg).
            emotor_specific_power: Electric motor specific power (kW/kg).
            lh2tank_gravimetric_index: LH2 tank gravimetric index (%).
            fuelcell_specific_power: Fuel cell system specific power (kW/kg).
            lift_to_drag: ``"model"`` for the L/D regressions, or a forced value.
            struct_weight_factor: Factor on the standard MWE regression (%).
            fuelcell_efficiency: Fuel cell system efficiency (%).
            electronics_specific_power: Power electronics specific power (kW/kg).
        """
        # Categories
        self.general = "general"
        self.commuter = "commuter"
        self.regional = "regional"
        self.business = "business"
        self.short_medium = "short_medium"  # Single aisle
        self.long_range = "long_range"  # Twin aisle

        # Thrusters
        self.propeller = "propeller"
        self.fan = "fan"

        # Engines
        self.turbofan = "turbofan"
        self.turboprop = "turboprop"
        self.piston = "piston"
        self.emotor = "emotor"

        self.power_elec = "power_elec"
        self.fuel_cell = "fuel_cell"

        # Energy storage
        self.petrol = "petrol"
        self.kerosene = "kerosene"
        self.e_fuel = "e_fuel"
        self.gasoline = "gasoline"
        self.gh2 = "compressed_h2"
        self.lh2 = "liquid_h2"
        self.lch4 = "liquid_ch4"
        self.lnh3 = "liquid_nh3"
        self.battery = "battery"

        # Maximum capacity and range per category, with most frequent speed, engine data
        self.category = {
            self.general: {
                "capacity": 6,
                "distance": unit.m_km(500),
                "speed": unit.convert_from("km/h", 300),
            },
            self.commuter: {
                "capacity": 19,
                "distance": unit.m_km(1500),
                "speed": unit.convert_from("km/h", 400),
            },
            self.regional: {"capacity": 80, "distance": unit.m_km(4500), "speed": 0.5},
            self.short_medium: {
                "capacity": 250,
                "distance": unit.m_km(8000),
                "speed": 0.78,
            },
            self.long_range: {
                "capacity": 550,
                "distance": unit.m_km(15000),
                "speed": 0.85,
            },
        }

        # Flight altitudes, [cruise altitude, diversion altitude, holding altitude]
        self.flight_altitudes = {
            self.general: unit.convert_from("ft", [5000, 3000, 1500]),
            self.commuter: unit.convert_from("ft", [10000, 6000, 1500]),
            self.regional: unit.convert_from("ft", [20000, 10000, 1500]),
            self.short_medium: unit.convert_from("ft", [35000, 25000, 1500]),
            self.long_range: unit.convert_from("ft", [35000, 25000, 1500]),
            self.business: unit.convert_from("ft", [35000, 25000, 1500]),
        }

        # Parameters for reserve computation, [flight fuel factor, diversion distance, holding time]
        self.reserve_parameters = {
            self.general: [0.0, 0.0, unit.s_min(30)],
            self.commuter: [0.0, 0.0, unit.s_min(30)],
            self.regional: [0.0, 0.0, unit.s_min(45)],
            self.short_medium: [0.05, unit.m_NM(200), unit.s_min(30)],
            self.long_range: [0.03, unit.m_NM(200), unit.s_min(30)],
            self.business: [0.05, unit.m_NM(200), unit.s_min(30)],
        }

        # Furnishing mass allowance per passenger (kg/pax)
        self.furnishing_dict = {
            self.general: 18,
            self.commuter: 18,
            self.regional: 22,
            self.short_medium: 22,
            self.long_range: 30,
            self.business: 40,
        }

        # Operator items index factor (kg/passenger/distance)
        self.operator_item_index = 5.0e-6

        # Passenger mass allowance (kg/pax)
        self.mpax_dict = {
            self.general: 95,
            self.commuter: 105,
            self.regional: 110,
            self.short_medium: 110,
            self.long_range: 120,
            self.business: 120,
        }
        self.mpax = "model"

        # L/D versus MTOW, distinguishing fan and propeller airplanes
        self.fan_mtow_list = array([200.0, 75000.0, 500000.0, 1.0e6])
        self.fan_lod_list = array([12.0, 18.0, 21.0, 21.0])
        self.prop_mtow_list = array([200.0, 30000.0, 1.0e6])
        self.prop_lod_list = array([13.0, 20.0, 20.0])
        self.lod = lift_to_drag  # Allows to force a value for L/D
        self.lod_factor = 1.0  # Tuning of aerodynamic efficiency

        # Mass factors
        self.max_payload_factor = (
            1.15  # max_payload = nominal_payload * max_payload_factor
        )
        self.max_fuel_factor = 1.25  # max_fuel = nominal_fuel * max_fuel_factor
        self.mlw_factor = 1.07  # MLW = MZFW * mlw_factor
        self.delta_payload = 0.0  # Reference payload tuning

        self.disa = 0.0
        self.take_off_time = unit.s_min(1)  # 30s take off + 30s initial climb

        # Reference power regression, ref_power = a * mtow**2 + b * mtow + c
        self.ref_power_factors = [8.41043794e-05, 2.02675942e02, -9.5e4]
        self.delta_power = 0.0  # Reference power tuning

        # Efficiencies
        self.prop_eff = 0.80  # Propeller efficiency
        self.fan_eff = 0.82  # Propeller-like fan efficiency
        self.emotor_eff = 0.90  # Electric motor efficiency (MAGNIX)
        self.fuel_cell_eff = (
            1e-2 * fuelcell_efficiency
        )  # System level (Horizon Fuel Cell)
        self.fuel_energy_ratio = 2.28  # eta_fan / (eta_prop * eta_thermal)
        self.propu_eff_factor = 1.0  # Propulsion system efficiency factor

        # PSFC of reciprocating engines (Lycoming IO-720)
        self.psfc_piston = unit.convert_from("kg/kW/h", 0.25)
        self.fhv_piston = fuel_heat(self.gasoline)

        # PSFC of turboshafts versus max power, kg/kW/h = a + b / (kW)**c
        self.psfc_turboshaft_coef = [0.17, 12.0, 0.63]
        self.fhv_turboshaft = fuel_heat(self.kerosene)

        # Turbofan thermal efficiency, and scale factor accounting for the
        # efficiency decrease of small turbomachines
        self.eff_th_turbofan = 0.474
        self.fuel_mixture = 0.020  # Fuel to air ratio
        self.turbofan_scale_coef = [0.2, 5.0e6, 2]  # 1 - (1-a) / (1+(max_power/b))**c

        # Propulsion component power densities (W/kg)
        self.power_density = {
            self.turbofan: unit.W_kW(4.3),
            self.turboprop: unit.W_kW(4.3),
            self.piston: unit.W_kW(1.1),  # Lycoming IO-720
            self.emotor: unit.W_kW(emotor_specific_power),
            self.power_elec: unit.W_kW(electronics_specific_power),
            self.fuel_cell: unit.W_kW(fuelcell_specific_power),
            self.propeller: unit.W_kW(10),
            self.fan: unit.W_kW(15),
        }

        # Energy storage
        self.battery_enrg_density = unit.J_Wh(battery_specific_energy)

        # 1000 bar.L/kg, Source AFHYPAC, Fiche 4.2 from CEA document
        self.tank_efficiency_factor = unit.convert_from("bar", 661e-3)
        self.initial_gh2_pressure = unit.convert_from("bar", 700)
        self.initial_lnh3_pressure = unit.convert_from("bar", 30)

        self.lh2_density = fuel_density(self.lh2)
        self.lch4_density = fuel_density(self.lch4)
        # kg_LH2 / (kg_LH2 + kg_Tank)
        self.lh2_tank_gravimetric_index = 1e-2 * lh2tank_gravimetric_index

        # Standard airframe MWE regression, standard_af_mwe = a * mtow**2 + b*mtow + c
        self.standard_af_mwe_factors = [-3.06255540e-07, 4.18303322e-01, -35]
        self.stdm_shift = 0.0  # Mass delta on standard MWE
        self.stdm_factor = 1e-2 * struct_weight_factor  # Mass factor on standard MWE

    # ------------------------------------------------------------------------------
    # Low level sub-models
    # ------------------------------------------------------------------------------
    def flight_altitude(self, airplane_type, cruise_altp=None):
        """Return the cruise, diversion and holding altitudes."""
        mz, dz, hz = self.flight_altitudes[airplane_type]
        if cruise_altp is not None:
            mz = cruise_altp
        return {
            "airplane_type": airplane_type,
            "mission": mz,
            "diversion": dz,
            "holding": hz,
        }

    def reserve_data(self, airplane_type):
        """Return the mission fuel factor, diversion leg and holding time."""
        ff, dl, ht = self.reserve_parameters[airplane_type]
        return {
            "airplane_type": airplane_type,
            "fuel_factor": ff,
            "diversion_leg": dl,
            "holding_time": ht,
        }

    def get_lod(self, mtow, power_system):
        """Return L/D estimation or user input."""
        if self.lod != "model":
            return self.lod
        if power_system["thruster_type"] == self.fan:
            lod_ref = interp(mtow, self.fan_mtow_list, self.fan_lod_list)
        elif power_system["thruster_type"] == self.propeller:
            lod_ref = interp(mtow, self.prop_mtow_list, self.prop_lod_list)
        else:
            msg = "thruster_type can only be 'fan' or 'propeller'"
            raise ValueError(msg)
        return lod_ref * self.lod_factor

    def get_piston_eff(self):
        """Return reciprocating engine overall efficiency, including propeller."""
        return (self.prop_eff * self.propu_eff_factor) / (
            self.fhv_piston * self.psfc_piston
        )

    def get_turboprop_eff(self, max_power):
        """Return turboprop engine overall efficiency, including propeller."""
        a, b, c = self.psfc_turboshaft_coef
        psfc_ref = unit.convert_from("kg/kW/h", a + b / unit.kW_W(max_power) ** c)
        return (self.prop_eff * self.propu_eff_factor) / (
            self.fhv_turboshaft * psfc_ref
        )

    def get_scale_factor(self, max_power):
        """Efficiency factor of turbomachines, decreasing with their size."""
        a, b, c = self.turbofan_scale_coef
        return 1 - (1 - a) / (1 + (max_power / b)) ** c

    def get_turbofan_eff(self, tas, bpr, max_power, energy_type):
        """Return turbofan engine overall efficiency, including fan."""
        fhv = fuel_heat(energy_type)
        eff_th = self.eff_th_turbofan
        alpha = self.fuel_mixture
        eff_pr = 1 / (
            0.5 + sqrt(0.25 + ((alpha * eff_th * fhv) / (2 * (1 + bpr) * tas**2)))
        )
        k_eff = self.get_scale_factor(max_power)
        return eff_th * eff_pr * k_eff * self.propu_eff_factor

    def get_emotor_eff(self, energy_type, thruster_type, max_power):
        """Return electric motor overall efficiency, including thruster and fuel cell."""
        if thruster_type == self.propeller:
            thruster_eff = self.prop_eff
        elif thruster_type == self.fan:
            thruster_eff = self.fan_eff
        else:
            msg = "thruster type is unknown"
            raise ValueError(msg)
        if energy_type == self.battery:
            return thruster_eff * self.emotor_eff * self.propu_eff_factor
        if energy_type in [self.gh2, self.lh2]:
            return (
                thruster_eff
                * self.emotor_eff
                * self.fuel_cell_eff
                * self.propu_eff_factor
            )
        msg = "energy type is unknown"
        raise ValueError(msg)

    def get_engine_eff(self, power_system, tas, max_power):
        """Overall efficiency of the propulsion system of the given architecture."""
        engine_type = power_system["engine_type"]
        if engine_type == self.piston:
            return self.get_piston_eff()
        if engine_type == self.turboprop:
            return self.get_turboprop_eff(max_power)
        if engine_type == self.turbofan:
            return self.get_turbofan_eff(
                tas, power_system["bpr"], max_power, power_system["energy_type"]
            )
        if engine_type == self.emotor:
            return self.get_emotor_eff(
                power_system["energy_type"], power_system["thruster_type"], max_power
            )
        msg = "power system, engine type is unknown"
        raise ValueError(msg)

    def ref_power(self, mtow):
        """Required total power for an airplane with a given MTOW."""
        a, b, c = self.ref_power_factors
        return (a * mtow + b) * mtow + c + self.delta_power

    def take_off_energy(self, total_power):
        """Energy required for take off."""
        return total_power * self.take_off_time

    def climb_energy(self, mass, alt):
        """Energy required to climb to the cruise altitude."""
        return mass * alt * phd.gravity()

    def standard_mass(self, mtow):
        """Standard MWE (MWE - furnishing - operator items) for a given MTOW."""
        a, b, c = self.standard_af_mwe_factors
        return (a * mtow + b) * mtow + c

    def furnishing(self, npax, category):
        """Semi empirical furnishing mass."""
        return self.furnishing_dict[category] * npax

    def op_item(self, npax, distance):
        """Semi empirical mass for operator items."""
        return self.operator_item_index * npax * distance

    def get_pax_allowance(self, category):
        """Passenger mass allowance."""
        return self.mpax_dict[category] if self.mpax == "model" else self.mpax

    # ------------------------------------------------------------------------------
    # Mass model components
    # ------------------------------------------------------------------------------
    def fuel_cell_system_mass(self, power_system, total_power):
        """Mass of the fuel cell system powering the electric motors."""
        return (total_power / self.emotor_eff) / self.power_density[self.fuel_cell]

    def propulsion_mass(self, power_system, total_power):
        """Mass of the propulsion system and of the fuel cell system, if any."""
        energy_type = power_system["energy_type"]
        engine_type = power_system["engine_type"]
        thruster_type = power_system["thruster_type"]

        fuel_cell_system_mass = 0.0
        if engine_type in [self.piston, self.turboprop]:
            propulsion_mass = total_power / self.power_density[engine_type]
            propulsion_mass += total_power / self.power_density[self.propeller]
        elif engine_type == self.turbofan:
            propulsion_mass = total_power / self.power_density[engine_type]
        elif engine_type == self.emotor:
            propulsion_mass = self.emotor_mass(power_system, total_power)
            propulsion_mass += total_power / self.power_density[self.power_elec]
            if thruster_type == self.fan:
                propulsion_mass += total_power / self.power_density[self.fan]
            elif thruster_type == self.propeller:
                propulsion_mass += total_power / self.power_density[self.propeller]
            else:
                msg = "target power system - thruster type is unknown"
                raise ValueError(msg)

            if energy_type in [self.gh2, self.lh2]:
                fuel_cell_system_mass = self.fuel_cell_system_mass(
                    power_system, total_power
                )
            elif energy_type != self.battery:
                msg = "target power system - energy_type is unknown"
                raise ValueError(msg)
        else:
            msg = "engine type is unknown"
            raise ValueError(msg)

        return propulsion_mass, fuel_cell_system_mass

    def emotor_mass(self, power_system, total_power):
        """Mass of the electric motors."""
        return total_power / self.power_density[self.emotor]

    def lh2_storage(self, power_system, max_fuel):
        """LH2 tank mass and realized gravimetric indices (tank-only and system)."""
        ratio = 1.0 / self.lh2_tank_gravimetric_index - 1.0
        return max_fuel * ratio, {
            "gi_tank": self.lh2_tank_gravimetric_index,
            "gi_system": self.lh2_tank_gravimetric_index,
        }

    def energy_storage_mass(self, power_system, max_fuel, max_enrg):
        """Fuel and/or energy storage mass, fuel density and storage data."""
        energy_type = power_system["energy_type"]
        storage_data = {}
        energy_storage_mass = 0.0
        if energy_type in [self.kerosene, self.e_fuel, self.gasoline, self.petrol]:
            density = fuel_density(energy_type)
        elif energy_type == self.gh2:
            density = fuel_density(self.gh2, press=self.initial_gh2_pressure)
            gh2_gi = 1 / (
                1 + self.initial_gh2_pressure / (self.tank_efficiency_factor * density)
            )
            energy_storage_mass += max_fuel * (1.0 / gh2_gi - 1.0)
        elif energy_type == self.lh2:
            density = fuel_density(self.lh2)
            energy_storage_mass, storage_data = self.lh2_storage(power_system, max_fuel)
        elif energy_type == self.lch4:
            density = fuel_density(self.lch4)
            # Extrapolation from liquid H2 tanks
            lh2_ratio = 1.0 / self.lh2_tank_gravimetric_index - 1.0
            energy_storage_mass += (
                max_fuel * (self.lh2_density / self.lch4_density) * lh2_ratio
            )
        elif energy_type == self.lnh3:
            density = fuel_density(self.lnh3)
            lnh3_gi = 1 / (
                1 + self.initial_lnh3_pressure / (self.tank_efficiency_factor * density)
            )
            energy_storage_mass += max_fuel * (1.0 / lnh3_gi - 1.0)
        elif energy_type == self.battery:
            density = fuel_density(self.battery)
            energy_storage_mass += max_enrg / self.battery_enrg_density
        else:
            msg = "energy_type is unknown"
            raise ValueError(msg)

        return energy_storage_mass, density, storage_data

    def owe_structure(
        self,
        category,
        npax,
        mtow,
        distance,
        total_power,
        max_fuel,
        max_enrg,
        power_system,
    ):
        """Compute OWE from the point of view of structures."""
        furnishing = self.furnishing(npax, category)
        operator_items = self.op_item(npax, distance)
        standard_mass = self.standard_mass(mtow)
        propulsion_mass, fuel_cell_system_mass = self.propulsion_mass(
            power_system, total_power
        )
        energy_storage_mass, density, storage_data = self.energy_storage_mass(
            power_system, max_fuel, max_enrg
        )

        basic_mwe = standard_mass * self.stdm_factor + self.stdm_shift
        std_mwe = (
            basic_mwe + propulsion_mass + energy_storage_mass + fuel_cell_system_mass
        )
        mwe = std_mwe + furnishing
        owe = mwe + operator_items

        return {
            "owe": owe,
            "op_item": operator_items,
            "mwe": mwe,
            "furnishing": furnishing,
            "std_mwe": std_mwe,
            "propulsion_mass": propulsion_mass,
            "fuel_cell_system_mass": fuel_cell_system_mass,
            "energy_storage_mass": energy_storage_mass,
            "fuel_density": density,
            "basic_mwe": basic_mwe,
            "stdm_factor": self.stdm_factor,
            "stdm_shift": self.stdm_shift,
            "storage_data": storage_data,
        }

    # ------------------------------------------------------------------------------
    # Mission fuel
    # ------------------------------------------------------------------------------
    def get_tas(self, tamb, speed, speed_type):
        """True air speed and Mach number."""
        vsnd = phd.sound_speed(tamb)
        if speed_type == "mach":
            return speed * vsnd, speed
        return speed, speed / vsnd

    def _burn(self, power_system, start_mass, g, distance, eff, lod):
        """Fuel mass (Breguet) and energy over a distance at the given efficiency."""
        if power_system["energy_type"] == self.battery:
            return 0.0, start_mass * g * distance / (eff * lod)  # Constant mass
        fhv = fuel_heat(power_system["energy_type"])
        fuel = start_mass * (1.0 - exp(-(g * distance) / (eff * fhv * lod)))
        return fuel, fuel * fhv

    def leg_fuel(
        self,
        start_mass,
        distance,
        altp,
        speed,
        speed_type,
        mtow,
        max_power,
        power_system,
    ):
        """Compute fuel and/or energy over a given distance."""
        _pamb, tamb, g = phd.atmosphere_g(altp, self.disa)
        tas, _mach = self.get_tas(tamb, speed, speed_type)
        time = distance / tas
        lod = self.get_lod(mtow, power_system)
        eff = self.get_engine_eff(power_system, tas, max_power)
        fuel, enrg = self._burn(power_system, start_mass, g, distance, eff, lod)
        return fuel, enrg, lod, eff, time

    def holding_fuel(
        self, start_mass, time, altp, speed, speed_type, mtow, max_power, power_system
    ):
        """Compute fuel and/or energy for a given holding time."""
        _pamb, tamb, g = phd.atmosphere_g(altp, self.disa)
        tas, _mach = self.get_tas(tamb, speed, speed_type)
        lod = self.get_lod(mtow, power_system)
        eff = self.get_engine_eff(power_system, tas, max_power)
        fuel, enrg = self._burn(power_system, start_mass, g, tas * time, eff, lod)
        return fuel, enrg, lod

    def total_fuel(
        self,
        tow,
        distance,
        cruise_speed,
        mtow,
        total_power,
        power_system,
        altitude_data,
        reserve_data,
    ):
        """Compute the total fuel (kg) and energy (J) required for a mission."""
        speed_type = "tas" if cruise_speed > 1 else "mach"
        max_power = total_power / power_system["engine_count"]
        cruise_altp = altitude_data["mission"]
        is_battery = power_system["energy_type"] == self.battery

        mission_fuel = 0.0
        mission_enrg = self.take_off_energy(total_power) + self.climb_energy(
            tow, cruise_altp
        )
        if not is_battery:
            mission_fuel += mission_enrg * (
                self.fuel_energy_ratio / fuel_heat(power_system["energy_type"])
            )

        fuel, enrg, mission_lod, global_eff, mission_time = self.leg_fuel(
            tow,
            distance,
            cruise_altp,
            cruise_speed,
            speed_type,
            mtow,
            max_power,
            power_system,
        )
        mission_fuel += fuel
        mission_enrg += enrg

        ldw = tow if is_battery else tow - mission_fuel

        reserve_fuel = 0.0
        reserve_enrg = 0.0
        if reserve_data["fuel_factor"] > 0:
            reserve_fuel += reserve_data["fuel_factor"] * mission_fuel
            reserve_enrg += reserve_data["fuel_factor"] * mission_enrg
        if reserve_data["diversion_leg"] > 0:
            lf, le, _lod, _eff, _time = self.leg_fuel(
                ldw,
                reserve_data["diversion_leg"],
                altitude_data["diversion"],
                cruise_speed,
                speed_type,
                mtow,
                max_power,
                power_system,
            )
            reserve_fuel += lf
            reserve_enrg += le
        if reserve_data["holding_time"] > 0:
            hf, he, _lod = self.holding_fuel(
                ldw,
                reserve_data["holding_time"],
                altitude_data["holding"],
                cruise_speed,
                speed_type,
                mtow,
                max_power,
                power_system,
            )
            reserve_fuel += hf
            reserve_enrg += he

        return {
            "tow": tow,
            "distance": distance,
            "total_fuel": mission_fuel + reserve_fuel,
            "mission_fuel": mission_fuel,
            "reserve_fuel": reserve_fuel,
            "total_enrg": mission_enrg + reserve_enrg,
            "mission_enrg": mission_enrg,
            "reserve_enrg": reserve_enrg,
            "mission_lod": mission_lod,
            "global_eff": global_eff,
            "mission_time": mission_time,
        }

    def owe_performance(
        self,
        payload,
        mtow,
        distance,
        cruise_speed,
        total_power,
        power_system,
        altitude_data,
        reserve_data,
    ):
        """Compute OWE from the point of view of the mission."""
        data = self.total_fuel(
            mtow,
            distance,
            cruise_speed,
            mtow,
            total_power,
            power_system,
            altitude_data,
            reserve_data,
        )
        return {
            "owe": mtow - payload - data["total_fuel"],
            "total_energy": data["total_enrg"],
            "total_fuel": data["total_fuel"],
            "mission_fuel": data["mission_fuel"],
            "reserve_fuel": data["reserve_fuel"],
            "max_fuel": data["total_fuel"]
            * self.max_fuel_factor,  # Tanks are sized for max fuel
            "mission_enrg": data["mission_enrg"],
            "reserve_enrg": data["reserve_enrg"],
            "max_energy": data["total_enrg"]
            * self.max_fuel_factor,  # Battery sized for max energy
            "payload": payload,
            "aerodynamic_efficiency": data["mission_lod"],
            "propulsion_system_efficiency": data["global_eff"],
        }

    # ------------------------------------------------------------------------------
    # Airplane sizing
    # ------------------------------------------------------------------------------
    def get_category_data(self, mission):
        """Retrieve design range, payload, category, speed and altitude of a mission."""
        design_range = mission["range"]
        category = mission.get("category")
        if category is None:
            if "npax" not in mission:
                msg = "Keys 'npax' AND 'range' must be valued in category guess mode"
                raise ValueError(msg)
            for cat, data in self.category.items():
                if (
                    mission["npax"] <= data["capacity"]
                    and design_range <= data["distance"]
                ):
                    category = cat
                    break
            else:
                msg = f"Could not find category for npax={mission['npax']} and range={design_range}"
                raise ValueError(msg)

        cruise_speed = mission.get(
            "speed", self.category.get(category, {}).get("speed")
        )
        cruise_altp = mission.get("altitude", self.flight_altitudes[category][0])

        if "npax" in mission and "payload" not in mission:
            npax = mission["npax"]
            mpax = self.get_pax_allowance(category)
            payload = npax * mpax + self.delta_payload
        elif "npax" in mission and "payload" in mission:
            npax = mission["npax"]
            payload = mission["payload"]
            mpax = payload / npax
        elif "payload" in mission:
            npax = 0
            mpax = self.get_pax_allowance(category)
            payload = mission["payload"]
        else:
            msg = "Key 'npax' or/and key 'payload' must be present in mission input dictionary"
            raise ValueError(msg)
        return design_range, npax, mpax, payload, category, cruise_speed, cruise_altp

    def design_airplane(self, power_system, mission):
        """Design the airplane for a mission, returning a dictionary of its characteristics.

        Args:
            power_system: ``energy_type``, ``engine_count``, ``engine_type``,
                ``thruster_type`` and ``bpr`` (turbofan only). The combination of
                ``"emotor"`` with ``"liquid_h2"`` or ``"compressed_h2"`` implies fuel
                cells.
            mission: ``category``, ``npax`` (or ``payload``), ``speed`` (m/s or
                Mach), ``range`` (m) and ``altitude`` (m).
        """
        power_system = {"bpr": None, **power_system}
        design_range, npax, mpax, payload, category, cruise_speed, cruise_altp = (
            self.get_category_data(mission)
        )
        altitude_data = self.flight_altitude(category, cruise_altp)
        reserve_data = self.reserve_data(category)

        def sizing(mtow):
            total_power = self.ref_power(mtow)
            dict_p = self.owe_performance(
                payload,
                mtow,
                design_range,
                cruise_speed,
                total_power,
                power_system,
                altitude_data,
                reserve_data,
            )
            dict_s = self.owe_structure(
                category,
                npax,
                mtow,
                design_range,
                total_power,
                dict_p["max_fuel"],
                dict_p["max_energy"],
                power_system,
            )
            return total_power, dict_p, dict_s

        def mass_mission_balance(mtow, args):
            _, dict_p, dict_s = sizing(mtow)
            return dict_p["owe"] - dict_s["owe"]

        mtow_ini = 0.9e-3 * (payload / mpax) * design_range
        sol = root_find(
            fn=mass_mission_balance,
            solver=Newton(rtol=1e-6, atol=1e-6),
            y0=mtow_ini,
            max_steps=500,
        )
        mtow = sol.value
        total_power, dict_p, dict_s = sizing(mtow)
        max_power = total_power / power_system["engine_count"]

        empty_mission = self.total_fuel(
            mtow,
            0.0,
            cruise_speed,
            mtow,
            total_power,
            power_system,
            altitude_data,
            reserve_data,
        )
        payload_max = minimum(
            mtow - dict_s["owe"] - empty_mission["total_fuel"],
            dict_p["payload"] * self.max_payload_factor,
        )
        mzfw = dict_s["owe"] + payload_max
        mlw = mzfw * self.mlw_factor

        design = self.design_dict(
            npax, mpax, design_range, cruise_speed, category, max_power, total_power, mtow, mzfw, mlw, payload_max,
            power_system, mission, altitude_data, reserve_data, dict_p, dict_s,
        )  # fmt: skip
        design.update(dict_s["storage_data"])
        return design

    def design_dict(
        self, npax, mpax, nominal_range, cruise_speed, category, max_power, total_power, mtow, mzfw, mlw, payload_max,
        power_system, mission, altitude_data, reserve_data, dict_p, dict_s,
    ):  # fmt: skip
        """Centralize the airplane characteristics into a dictionary."""
        speed_type = "tas" if cruise_speed > 1 else "mach"
        _pamb, tamb, _g = phd.atmosphere_g(altitude_data["mission"], self.disa)
        tas, _mach = self.get_tas(tamb, cruise_speed, speed_type)
        return {
            "airplane_type": category,
            "npax": npax,
            "mpax": mpax,
            "delta_payload": self.delta_payload,
            "payload": dict_p["payload"],
            "mission": mission,
            "nominal_range": nominal_range,
            "cruise_speed": cruise_speed,
            "cruise_tas": tas,
            "nominal_time": nominal_range / tas,
            "altitude_data": altitude_data,
            "reserve_data": reserve_data,
            "mission_fuel": dict_p["mission_fuel"],
            "reserve_fuel": dict_p["reserve_fuel"],
            "total_fuel": dict_p["total_fuel"],
            "fuel_consumption": (dict_p["mission_fuel"] / dict_s["fuel_density"])
            / npax
            / nominal_range,
            "mission_enrg": dict_p["mission_enrg"],
            "reserve_enrg": dict_p["reserve_enrg"],
            "total_energy": dict_p["total_energy"],
            "enrg_consumption": dict_p["mission_enrg"] / npax / nominal_range,
            "n_engine": power_system["engine_count"],
            "by_pass_ratio": power_system["bpr"],
            "max_power": max_power,
            "total_power": total_power,
            "power_system": power_system,
            "mtow": mtow,
            "mlw": mlw,
            "mzfw": mzfw,
            "payload_max": payload_max,
            "max_payload_factor": self.max_payload_factor,
            "owe": dict_s["owe"],
            "op_item": dict_s["op_item"],
            "mwe": dict_s["mwe"],
            "furnishing": dict_s["furnishing"],
            "std_mwe": dict_s["std_mwe"],
            "propulsion_mass": dict_s["propulsion_mass"],
            "energy_storage_mass": dict_s["energy_storage_mass"],
            "fuel_cell_system_mass": dict_s["fuel_cell_system_mass"],
            "basic_mwe": dict_s["basic_mwe"],
            "stdm_factor": dict_s["stdm_factor"],
            "stdm_shift": dict_s["stdm_shift"],
            "storage_energy_density": dict_p["total_energy"]
            / (dict_s["energy_storage_mass"] + dict_p["total_fuel"]),
            "propulsion_power_density": total_power
            / (dict_s["propulsion_mass"] + dict_s["fuel_cell_system_mass"]),
            "aero_eff_factor": self.lod_factor,
            "aerodynamic_efficiency": dict_p["aerodynamic_efficiency"],
            "propulsion_system_efficiency": dict_p["propulsion_system_efficiency"],
            "structural_factor": dict_s["owe"] / mtow,
            "pk_o_mass": npax * nominal_range / dict_s["owe"],
            "pk_o_enrg": npax * nominal_range / dict_p["total_energy"],
        }
