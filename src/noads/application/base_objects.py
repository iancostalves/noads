# Copyright 2025 ISAE-SUPAERO, https://www.isae-supaero.fr/en/
# Copyright 2025 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""Initialize object-oriented model structure."""

from numpy import array

from noads.core.models.energy.energy import Energy
from noads.core.models.energy.energy import ProducedEnergy
from noads.core.models.energy.energy import ProducedEnergyCarrier
from noads.core.models.energy.energy_mix import EnergyMix
from noads.core.models.energy.production_pathway import ProductionPathway
from noads.core.models.energy.streams import Impact
from noads.core.models.fleet.aircraft_design import AircraftDesign
from noads.core.models.fleet.aircraft_operation import AircraftOperation
from noads.core.models.fleet.aircraft_operation import PropulsionSystem
from noads.core.models.fleet.aircraft_tech_parameter import AircraftTechParameter
from noads.core.models.fleet.fleet import Fleet
from noads.core.models.fleet.fleet import FleetAssembly

category_conso = {
    "general": (2.7355931281666916, 1.9140569910448313, 1.4709811219423268),
    "commuter": (1.2471681199434796, 1.014373482550286, 0.8806485123652082),
    "regional": (0.8721753476178873, 0.804966734517787, 0.7304364463650426),
    "short_medium": (1.0041690806234553, 0.8710601781283808, 0.8223594455080704),
    "long_range": (1.0371264261013249, 0.9195081834700178, 0.8283365677650192),
}

category_lifetime = {
    # 3rd quartile, median, 1st quartile
    # "general": (33.8, 22.2, 11.2),
    # "commuter": (26.5, 21.9, 18.6),
    # "regional": (20.5, 15.3, 9.2),
    # "short_medium": (33.8, 24.8, 12.4),
    # "long_range": (29.5, 23.6, 18.7),--------------------------------------------
    # 3rd quartile, (half-way between), median
    # This avoids too low lifetime values for the 1st quartile
    "general": (33.8, (33.8 + 22.2) / 2, 22.2),
    "commuter": (26.5, (26.5 + 21.9) / 2, 21.9),
    "regional": (20.5, (20.5 + 15.3) / 2, 15.3),
    "short_medium": (33.8, (33.8 + 24.8) / 2, 24.8),
    "long_range": (29.5, (29.5 + 23.6) / 2, 23.6),
}

categories_mission = {
    "general": {"npax": 19, "range": 500e3},
    "commuter": {"npax": 50, "range": 1500e3},
    "regional": {"npax": 80, "range": 4500e3},
    "short_medium": {"npax": 120, "range": 8000e3},
    "long_range": {"npax": 250, "range": 15000e3},
}

propulsion_mission = {
    "JetA-GasTurbine": {"speed": 0.8 * 340, "altitude": 28000 * 0.3048},
    "Battery-Electric": {"speed": 0.5 * 340, "altitude": 20000 * 0.3048},
    "lH2-FuelCell": {"speed": 0.5 * 340, "altitude": 20000 * 0.3048},
    "lH2-GasTurbine": {"speed": 0.8 * 340, "altitude": 28000 * 0.3048},
    # "lCH4-GasTurbine": {"speed": 0.8 * 340, "altitude": 28000 * 0.3048},
}

propulsion_architectures = {
    "JetA-GasTurbine": {
        "engine_count": 2,
        "engine_type": "turbofan",
        "thruster_type": "fan",
        "energy_type": "kerosene",
        "bpr": 12.0,
    },
    "Battery-Electric": {
        "engine_count": 2,
        "engine_type": "emotor",
        "thruster_type": "propeller",
        "energy_type": "battery",
    },
    "lH2-FuelCell": {
        "engine_count": 2,
        "engine_type": "emotor",
        "thruster_type": "propeller",
        "energy_type": "liquid_h2",
    },
    "lH2-GasTurbine": {
        "engine_count": 2,
        "engine_type": "turbofan",
        "thruster_type": "fan",
        "energy_type": "liquid_h2",
        "bpr": 12.0,
    },
    # "lCH4-GasTurbine": {
    #     "engine_count": 2,
    #     "engine_type": "turbofan",
    #     "thruster_type": "fan",
    #     "energy_type": "liquid_ch4",
    #     "bpr": 12.0,
    # },
}

tech_params_lower_mid_upper_2020_2040_2060 = {
    "battery_specific_energy": (
        array([200, 350, 600]),
        0.5 * (array([200, 350, 600]) + array([200, 800, 1500])),
        array([200, 800, 1500]),
    ),
    "emotor_specific_power": (
        array([2, 10, 15]),
        0.5 * (array([2, 10, 15]) + array([2, 20, 28])),
        array([2, 20, 28]),
    ),
    "electronics_specific_power": (
        array([2, 15, 20]),
        0.5 * (array([2, 15, 20]) + array([2, 25, 32])),
        array([2, 25, 32]),
    ),
    "fuelcell_specific_power": (
        array([1, 2, 3]),
        0.5 * (array([1, 2, 3]) + array([1, 3, 6])),
        array([1, 3, 6]),
    ),
    "lh2tank_gravimetric_index": (
        array([0.2, 0.3, 0.35]) * 1e2,
        0.5 * (array([0.2, 0.3, 0.35]) * 1e2 + array([0.2, 0.65, 0.8]) * 1e2),
        array([0.2, 0.65, 0.8]) * 1e2,
    ),
    "fuelcell_efficiency": (
        array([0.40, 0.45, 0.5]) * 1e2,
        0.5 * (array([0.40, 0.45, 0.5]) * 1e2 + array([0.40, 0.55, 0.65]) * 1e2),
        array([0.40, 0.55, 0.65]) * 1e2,
    ),
    "struct_weight_factor": (
        array([1, 0.9, 0.85]) * 1e2,
        0.5 * (array([1, 0.9, 0.85]) * 1e2 + array([1, 0.66, 0.55]) * 1e2),
        array([1, 0.66, 0.55]) * 1e2,
    ),
}


def initialize_base_objects(drop_in_only=False, technology_index=0):
    """Initialize base model objects."""
    if technology_index < 0 or technology_index > 2:
        msg = "Please enter 0, 1 or 2 as technology index (low, mid, up)"
        raise RuntimeError(msg)

    co2 = Impact(name="CO2", unit="gCO2", budget=900e15)

    # ENERGY MIX ______________________________________________________________________
    # Non-produced energies ............................................................
    oil = Energy("OIL")
    biomass = Energy("BIOMASS")
    electricity = Energy("ELECTRICITY")
    # natural_gas = Energy("NATURAL_GAS")

    # Production Pathways and their associated Secondary energies ......................

    # Kerosene from oil -------------------
    refinery = ProductionPathway(
        "Refinery",
        impacts=[co2],
        input_streams=[oil],
    )
    kerosene = ProducedEnergy(
        "KEROSENE",
        pathways=[refinery],
    )

    bio_hefa = ProductionPathway(
        "HEFA",
        impacts=[co2],
        input_streams=[biomass],
    )
    bio_ft = ProductionPathway(
        "FT",
        impacts=[co2],
        input_streams=[biomass],
    )
    bio_atj = ProductionPathway(
        "ATJ",
        impacts=[co2],
        input_streams=[biomass],
    )
    biofuel = ProducedEnergy(
        "BIOFUEL",
        pathways=[bio_ft, bio_atj, bio_hefa],
    )

    # Gas-Hydrogen
    electrolysis = ProductionPathway(
        "Electrolysis",
        impacts=[],
        input_streams=[electricity],
    )
    gas = ProductionPathway(
        "Gas_reforming",
        impacts=[co2],
        input_streams=[],
    )
    gh2 = ProducedEnergy(
        "GAS-H2",
        pathways=[electrolysis, gas],
    )
    # E-fuel
    ptl = ProductionPathway(
        "Power_to_liquid",
        impacts=[],
        input_streams=[gh2, electricity],
    )
    e_fuel = ProducedEnergy(
        "E-FUEL",
        pathways=[ptl],
    )

    # # Gas methane
    # fossil_ch4 = ProductionPathway(
    #     "GM_fossil",
    #     impacts=[],
    #     input_streams=[natural_gas],
    # )
    # ptg = ProductionPathway(
    #     "GM_methanation",
    #     impacts=[],
    #     input_streams=[gh2],
    # )
    # biogas = ProductionPathway(
    #     "GM_biogas",
    #     impacts=[co2],
    #     input_streams=[biomass],
    # )
    # gch4 = ProducedEnergy(
    #     "GAS-CH4",
    #     pathways=[ptg, biogas, fossil_ch4]
    # )

    # Final Energy Carriers ............................................................
    # Drop-in
    e_drop_in = ProductionPathway(
        "Electrofuel",
        impacts=[],
        input_streams=[e_fuel],
    )
    bio_drop_in = ProductionPathway(
        "Biofuel",
        impacts=[],
        input_streams=[biofuel],
    )
    fossil_drop_in = ProductionPathway(
        "Fossil",
        impacts=[],
        input_streams=[kerosene],
    )
    drop_in = ProducedEnergyCarrier(
        "JET-A",
        pathways=[e_drop_in, bio_drop_in, fossil_drop_in],
        density=33.8 / 44.1,
        specific_energy=44.1,
    )
    # Liquid-Hydrogen
    gh2_liquefaction = ProductionPathway(
        "H2_liquefaction",
        impacts=[],
        input_streams=[electricity, gh2],
    )

    lh2 = ProducedEnergyCarrier(
        "LIQUID-H2",
        pathways=[gh2_liquefaction],
        density=8.49 / 120.0,
        specific_energy=120.0,
    )

    # # Liquid-methane
    # gch4_liquefaction = ProductionPathway(
    #     "LM_liquefaction",
    #     impacts=[],
    #     input_streams=[electricity, gch4],
    # )
    #
    # lch4 = ProducedEnergyCarrier(
    #     "LIQUID-CH4",
    #     pathways=[gch4_liquefaction],
    #     density=22.2 / 53.6,
    #     specific_energy=53.6,
    # )

    # Batteries
    charging = ProductionPathway(
        "Charging",
        impacts=[],
        input_streams=[electricity],
    )
    battery = ProducedEnergyCarrier(
        "BATTERY",
        pathways=[charging],
        density=2.32 / 1.29,
        specific_energy=1.29,
    )

    # Energy Mix and aircraft propulsion ...............................................
    energies = [kerosene, biofuel, e_fuel, drop_in, gh2]
    prop_systems = {
        "JetA-GasTurbine": PropulsionSystem("turbofan", {drop_in: 1.0}),
    }
    if not drop_in_only:
        energies.extend([lh2, battery])
        prop_systems.update({
            "lH2-FuelCell": PropulsionSystem("lh2_fuel_cell", {lh2: 1.0}),
            "lH2-GasTurbine": PropulsionSystem("lh2_burn", {lh2: 1.0}),
            "Battery-Electric": PropulsionSystem("electric_propulsion", {battery: 1.0}),
        })
        #     energies.extend([gch4, lch4])
        #     prop_systems.update(
        #         {
        #             "lCH4-GasTurbine": PropulsionSystem("lch4_burn", {lch4: 1.0}),
        #         }
        #     )

    energy_mix = EnergyMix(energies, inputs_to_constrain=[electricity, biomass])

    # Aircraft technology evolution parameters
    aircraft_tech_params = []
    for (
        param_name,
        lower_mid_upper,
    ) in tech_params_lower_mid_upper_2020_2040_2060.items():
        param_values = lower_mid_upper[technology_index]
        aircraft_tech_params.append(
            AircraftTechParameter(
                param_name, (param_values[0], param_values[1], param_values[2])
            )
        )

    fleets = []
    for _cat_i, (cat_name, _cat_mission) in enumerate(categories_mission.items()):
        # List of aircraft within category
        aircraft = []

        aircraft_reference_2019 = AircraftOperation(
            name=f"Current_fleet_{cat_name}",
            propulsion=prop_systems["JetA-GasTurbine"],
            energy_per_ask=category_conso[cat_name][technology_index],
            recent=True,
            lifetime=20.0,
        )
        aircraft.append(aircraft_reference_2019)

        # Add new aircraft to be entering the fleet
        for prop_name in prop_systems:
            mission = {
                **categories_mission[cat_name],
                **propulsion_mission[prop_name],
                "category": cat_name,
            }
            if "Electric" in prop_name:
                if cat_name == "general" or (
                    cat_name == "commuter" and technology_index > 0
                ):
                    # Electric aircraft always included for general market, commuter
                    # market only if not lower technology
                    aircraft.append(
                        AircraftDesign(
                            name=f"{prop_name}_{cat_name}",
                            propulsion=prop_systems[prop_name],
                            mission=mission,
                            power_system=propulsion_architectures[prop_name],
                            aircraft_tech_params=aircraft_tech_params,
                            reference_aircraft=aircraft_reference_2019,
                        )
                    )
            elif "JetA-GasTurbine" in prop_name:
                aircraft.extend((
                    AircraftDesign(
                        name=f"{prop_name}-v1_{cat_name}",
                        propulsion=prop_systems[prop_name],
                        mission=mission,
                        power_system=propulsion_architectures[prop_name],
                        aircraft_tech_params=aircraft_tech_params,
                        reference_aircraft=aircraft_reference_2019,
                    ),
                    AircraftDesign(
                        name=f"{prop_name}-v2_{cat_name}",
                        propulsion=prop_systems[prop_name],
                        mission=mission,
                        power_system=propulsion_architectures[prop_name],
                        aircraft_tech_params=aircraft_tech_params,
                        reference_aircraft=aircraft_reference_2019,
                    ),
                ))
            else:
                aircraft.append(
                    AircraftDesign(
                        name=f"{prop_name}_{cat_name}",
                        propulsion=prop_systems[prop_name],
                        mission=mission,
                        power_system=propulsion_architectures[prop_name],
                        aircraft_tech_params=aircraft_tech_params,
                        reference_aircraft=aircraft_reference_2019,
                    )
                )

        # Create the fleet object assembling all aircraft
        fleets.append(Fleet(cat_name, operating_aircraft=aircraft))
    fleet = FleetAssembly(fleets)

    return energy_mix, fleet
