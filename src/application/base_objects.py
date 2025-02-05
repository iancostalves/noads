from numpy import array

from AeroMAX.models.energy.energy import ProducedEnergy
from AeroMAX.models.energy.energy import ProducedEnergyCarrier
from AeroMAX.models.energy.energy_mix import EnergyMix
from AeroMAX.models.energy.production_pathway import ProductionPathway
from AeroMAX.models.energy.energy import Energy
from AeroMAX.models.energy.streams import Impact
from AeroMAX.models.fleet.aircraft_design import AircraftDesign
from AeroMAX.models.fleet.aircraft_operation import AircraftOperation
from AeroMAX.models.fleet.aircraft_operation import PropulsionSystem
from AeroMAX.models.fleet.aircraft_tech_parameter import AircraftTechParameter
from AeroMAX.models.fleet.fleet import Fleet
from AeroMAX.models.fleet.fleet import FleetAssembly

from gam_fleet import aeroscope_category_conso

categories_mission = {
    "general": {"npax": 19, "range": 500e3},
    "commuter": {"npax": 50, "range": 1500e3},
    "regional": {"npax": 80, "range": 4500e3},
    "short_medium": {"npax": 120, "range": 8000e3},
    "long_range": {"npax": 250, "range": 15000e3},
}

propulsion_mission = {
    "JetA-GasTurbine": {"speed": 0.8 * 340, "altitude": 28000 * 0.3048},
    # "Turboprop": {"speed": 0.5 * 340, "altitude": 20000 * 0.3048},
    "Battery-Electric": {"speed": 0.5 * 340, "altitude": 20000 * 0.3048},
    "lH2-FuelCell": {"speed": 0.5 * 340, "altitude": 20000 * 0.3048},
    "lH2-GasTurbine": {"speed": 0.8 * 340, "altitude": 28000 * 0.3048},
    "lCH4-GasTurbine": {"speed": 0.8 * 340, "altitude": 28000 * 0.3048},
}

propulsion_architectures = {
    "JetA-GasTurbine": {
        "engine_count": 2,
        "engine_type": "turbofan",
        "thruster_type": "fan",
        "energy_type": "kerosene",
        "bpr": 12.0,
    },
    # "Turboprop": {
    #     "engine_count": 2,
    #     "engine_type": "turboprop",
    #     "thruster_type": "propeller",
    #     "energy_type": "kerosene",
    # },
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
    "lCH4-GasTurbine": {
        "engine_count": 2,
        "engine_type": "turbofan",
        "thruster_type": "fan",
        "energy_type": "liquid_ch4",
        "bpr": 12.0,
    },
}

tech_params_lower_upper_2020_2035_2050 = {
    "battery_specific_energy": (
        array([200, 350, 600]),
        0.5 * (array([200, 350, 600]) + array([200, 700, 1300])),
        array([200, 700, 1300]),
    ),
    "emotor_specific_power": (
        array([2, 15, 20]),
        0.5 * (array([1, 15, 20]) + array([1, 20, 26])),
        array([1, 20, 26]),
    ),
    "electronics_specific_power": (
        array([2, 15, 20]),
        0.5 * (array([1, 15, 20]) + array([1, 22, 30])),
        array([1, 22, 30]),
    ),
    "fuelcell_specific_power": (
        array([1, 2, 3]),
        0.5 * (array([1, 2, 3]) + array([1, 3, 5])),
        array([1, 3, 5]),
    ),
    "lh2tank_gravimetric_index": (
        array([0.2, 0.3, 0.4]) * 1e2,
        0.5 * (array([0.2, 0.3, 0.4]) * 1e2 + array([0.2, 0.6, 0.75]) * 1e2),
        array([0.2, 0.6, 0.75]) * 1e2,
    ),
    "fuelcell_efficiency": (
        array([0.40, 0.45, 0.5]) * 1e2,
        0.5 * (array([0.40, 0.45, 0.5]) * 1e2 + array([0.40, 0.50, 0.60]) * 1e2),
        array([0.40, 0.50, 0.60]) * 1e2,
    ),
    "struct_weight_factor": (
        array([1, 0.85, 0.8]) * 1e2,
        0.5 * (array([1, 0.85, 0.8]) * 1e2 + array([1, 0.7, 0.6]) * 1e2),
        array([1, 0.7, 0.6]) * 1e2,
    ),
}


def initialize_aeromax_objects(
    drop_in_only=False, include_methane=False, technology_index=0,
):
    if technology_index < 0 or technology_index > 2:
        raise RuntimeError("Please enter 0, 1 or 2 as technology index (low, mid, up)")
    if include_methane:
        raise NotImplementedError("Sorry, methane aircraft still not modeled")

    co2 = Impact(name="CO2", unit="gCO2", budget=900e15)

    ## ENERGY MIX ______________________________________________________________________
    # Non-produced energies ............................................................
    oil = Energy("OIL")
    biomass = Energy("BIOMASS")
    electricity = Energy("ELECTRICITY")
    natural_gas = Energy("NATURAL_GAS")

    # Production Pathways and their associated Secondary energies ......................

    # Kerosene from oil -------------------
    refinery = ProductionPathway(
        "Refinery",
        impacts=[],
        input_streams=[oil],
    )
    kerosene = ProducedEnergy(
        "KEROSENE",
        pathways=[refinery],
    )

    # Biofuel
    # bio_hefa_fog = ProductionPathway(
    #     "HEFA-fog",
    #     impacts=[co2],
    #     input_streams=[biomass],
    # )
    bio_hefa = ProductionPathway(
        "HEFA",
        impacts=[co2],
        input_streams=[biomass],
    )
    # bio_ft_others = ProductionPathway(
    #     "FT-others",
    #     impacts=[co2],
    #     input_streams=[biomass],
    # )
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
    # gas_ccs = ProductionPathway(
    #     "GH_gas_ccs",
    #     impacts=[co2],
    #     input_streams=[],
    # )
    # coal_ccs = ProductionPathway(
    #     "GH_coal_ccs",
    #     impacts=[co2],
    #     input_streams=[],
    # )
    gas = ProductionPathway(
        "Gas_reforming",
        impacts=[co2],
        input_streams=[],
    )
    # coal = ProductionPathway(
    #     "Coal_gasification",
    #     impacts=[co2],
    #     input_streams=[],
    # )
    gh2 = ProducedEnergy(
        "GAS-H2",
        pathways=[electrolysis, gas],
    )  # gas_ccs, coal_ccs,
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
    # Gas methane
    fossil_ch4 = ProductionPathway(
        "GM_fossil",
        impacts=[],
        input_streams=[natural_gas],
    )
    ptg = ProductionPathway(
        "GM_methanation",
        impacts=[],
        input_streams=[gh2],
    )
    biogas = ProductionPathway(
        "GM_biogas",
        impacts=[co2],
        input_streams=[biomass],
    )
    gch4 = ProducedEnergy(
        "GAS-CH4",
        pathways=[ptg, biogas, fossil_ch4]
    )

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

    # Liquid-methane
    gch4_liquefaction = ProductionPathway(
        "LM_liquefaction",
        impacts=[],
        input_streams=[electricity, gch4],
    )

    lch4 = ProducedEnergyCarrier(
        "LIQUID-CH4",
        pathways=[gch4_liquefaction],
        density=22.2 / 53.6,
        specific_energy=53.6,
    )

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
        # "Turboprop": PropulsionSystem("turboprop", {drop_in: 1.0}),
    }
    if not drop_in_only:
        energies.extend([lh2, battery])
        prop_systems.update(
            {
                "lH2-FuelCell": PropulsionSystem("lh2_fuel_cell", {lh2: 1.0}),
                "lH2-GasTurbine": PropulsionSystem("lh2_burn", {lh2: 1.0}),
                "Battery-Electric": PropulsionSystem("electric_propulsion", {battery: 1.0}),
            }
        )
        if include_methane:
            energies.extend([gch4, lch4])
            prop_systems.update(
                {
                    "lCH4-GasTurbine": PropulsionSystem("lch4_burn", {lch4: 1.0}),
                }
            )

    energy_mix = EnergyMix(energies, inputs_to_constrain=[electricity, biomass])

    # Aircraft technology evolution parameters
    aircraft_tech_params = []
    for param_name, lower_mid_upper in tech_params_lower_upper_2020_2035_2050.items():
        param_values = lower_mid_upper[technology_index]
        aircraft_tech_params.append(
            AircraftTechParameter(
                param_name, (param_values[0], param_values[1], param_values[2])
            )
        )

    fleets = []
    for cat_i, (cat_name, cat_mission) in enumerate(categories_mission.items()):
        # List of aircraft within category
        aircraft = []

        aircraft_reference_2019 = AircraftOperation(
            name=f"Current_fleet_{cat_name}",
            propulsion=prop_systems["JetA-GasTurbine"],
            energy_per_ask=aeroscope_category_conso[cat_name],
            recent=True,
            lifetime=20.,
        )
        aircraft.append(aircraft_reference_2019)

        # Add new aircraft to be entering the fleet
        for prop_name, prop_sys in prop_systems.items():
            mission = {
                **categories_mission[cat_name], **propulsion_mission[prop_name],
                "category": cat_name,
            }
            if "Electric" in prop_name:
                if cat_name == "general" or (
                        cat_name == "commuter" and technology_index > 0
                ):
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
    fleet = FleetAssembly(fleets, design_aircraft=True)

    return energy_mix, fleet
