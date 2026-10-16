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

"""Instantiation of the scenario building blocks of the paper.

This module declares the market segmentation (missions, current-fleet consumption
quartiles, and lifetimes per distance band), the propulsion architectures and their
cruise conditions, the aircraft component technology parameters per technology
scenario, and the energy production graph (resources, pathways, and carriers).
:func:`initialize_base_objects` assembles them into the
:class:`~noads.core.models.energy.energy_mix.EnergyMix` and
:class:`~noads.core.models.fleet.fleet.FleetAssembly` used by
:mod:`noads.application.scenario_setup`. To change the fleet partitioning or add
aircraft, pathways, or resources, see the "Extending the analysis" section of the
documentation.
"""

from numpy import array

from noads.core.models.energy.energy import Energy
from noads.core.models.energy.energy import ProducedEnergy
from noads.core.models.energy.energy import ProducedEnergyCarrier
from noads.core.models.energy.energy_mix import EnergyMix
from noads.core.models.energy.production_pathway import ProductionPathway
from noads.core.models.energy.streams import Impact
from noads.core.models.fleet.aircraft_design import AIRCRAFT_MODELS
from noads.core.models.fleet.aircraft_design import UPDATE_GAM_OPTIONS
from noads.core.models.fleet.aircraft_design import AircraftDesign
from noads.core.models.fleet.aircraft_operation import AircraftOperation
from noads.core.models.fleet.aircraft_operation import PropulsionSystem
from noads.core.models.fleet.aircraft_tech_parameter import AircraftTechParameter
from noads.core.models.fleet.aircraft_tech_parameter import LogisticTechParameter
from noads.core.models.fleet.aircraft_tech_parameter import ScenarioTechParameter
from noads.core.models.fleet.fleet import Fleet
from noads.core.models.fleet.fleet import FleetAssembly
from noads.gam_jax.models import gam_v3

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

# UPDATE AIRCRAFT MODEL ________________________________________________________________
# Used with ``aircraft_model="update"``: GAM V3.0 port, powertrain scale effects and
# size-dependent LH2 tanks. See the "Update paper" section of the documentation.

# Technology parameters of the update model, for the Lower, Mid and Upper scenarios
# at 2020, 2040, 2060 and 2080 (ScenarioTechParameter: monotone cubic interpolation,
# constant after 2080). All scenarios share their 2020 value (the Mid one), and the
# gaps of Lower and Upper to Mid never shrink, so that the Upper-to-Lower band
# starts at zero and only widens:
# - 2040 and 2060 values are those of the paper (above) or of the powertrain
#   calibration (aircraft_tech_data/powertrain/: powertrain_tech_data.csv,
#   calibrate.py, calibrated_params.json), except the fuel cell and TMS values,
#   revised downwards from the 2026 literature review
#   (aircraft_tech_data/literature_data.csv);
# - 2080 values extend the calibrated curves where they exist (e-motor specific power
#   and efficiency), otherwise they add half of the 2040-2060 increase;
# - where a calibrated gap to Mid shrinks in time, it is held at its largest value.
# Specific powers and efficiencies are given at a reference power of 1 MW per
# propulsor, and fuelcell_specific_power then excludes heat rejection, which is sized
# separately from the rated heat load.
update_tech_params_lower_mid_upper_2020_2040_2060_2080 = {
    # Wh/kg
    "battery_specific_energy": (
        (200.0, 350.0, 600.0, 725.0),
        (200.0, 575.0, 1050.0, 1287.5),
        (200.0, 800.0, 1500.0, 1850.0),
    ),
    # kW/kg at 1 MW per propulsor. Pastra et al. logistic shifted by a 10/8/6-year
    # delay for Lower and Mid; Upper capped at 35 kW/kg, as 15-25 kW/kg is the highest
    # credible long-term value found and 50 kW/kg a curve fit
    "emotor_specific_power": (
        (2.7, 9.2, 11.2, 11.4),
        (2.7, 14.8, 24.7, 26.3),
        (2.7, 20.0, 30.0, 35.0),
    ),
    # %, motor x inverter x distribution at 1 MW
    "emotor_efficiency": (
        (91.33, 93.51, 94.63, 95.19),
        (91.33, 94.64, 96.27, 97.085),
        (91.33, 95.95, 97.68, 98.545),
    ),
    # kW/kg
    "electronics_specific_power": (
        (2.0, 15.0, 20.0, 22.5),
        (2.0, 20.0, 26.0, 29.0),
        (2.0, 25.0, 32.0, 35.5),
    ),
    # kW/kg, stack + BoP without heat rejection. 2020: flown aviation systems at
    # 0.2-0.6 kW/kg, automotive systems 0.6-0.86 kW/kg; 2030 targets about 2 kW/kg
    # (Clean Hydrogen Partnership); no system target above about 2-3 kW/kg found
    "fuelcell_specific_power": (
        (0.75, 1.5, 2.0, 2.25),
        (0.75, 2.0, 2.75, 3.25),
        (0.75, 2.5, 3.25, 4.0),
    ),
    # %, system at cruise. 2020: about 39-40 % at cruise (compressor losses at
    # altitude); DOE peak targets (65-72 %) are low-load values, not cruise
    "fuelcell_efficiency": (
        (40.0, 44.0, 46.0, 48.0),
        (40.0, 46.0, 49.0, 51.0),
        (40.0, 48.0, 51.0, 54.0),
    ),
    # kW heat/kg, full TMS sized at hot-day take-off: 1.5-3 today, about 5 for
    # FlyZero, 5-15 with HT-PEM or two-phase cooling, about 20 as long-term ceiling
    "fuelcell_tms_heat_rejection": (
        (2.0, 3.5, 5.0, 6.0),
        (2.0, 5.0, 8.0, 10.0),
        (2.0, 7.0, 12.0, 15.0),
    ),
    # -, TMS parasitic power per unit heat: about 0.2-0.27 today (no-fan heat
    # exchanger adds 27 % to cruise power), 0.10-0.13 for optimized cruise designs
    "fuelcell_tms_power_loss": (
        (0.225, 0.2, 0.18, 0.16),
        (0.225, 0.17, 0.14, 0.12),
        (0.225, 0.14, 0.10, 0.08),
    ),
    # MW per propulsor, interpolated in log space. Demonstrated 0.7-1 MW; projections
    # stop at 2 MW (ZEROe) and 2-5 MW (ZA2000, NASA 5 MW concept): Mid is capped at
    # 5 MW, and Upper goes beyond it as an exploratory bound
    "max_unit_power": (
        (0.06, 1.5, 2.3, 2.8),
        (0.06, 2.5, 4.0, 5.0),
        (0.06, 4.0, 6.5, 9.0),
    ),
    # %, overall (thermal x propulsive) efficiency of thermal engines relative to
    # today's (LEAP / GTF generation): LEAP gained 15 % fuel burn over the previous
    # generation, UltraFan targets about 10 % and CFM RISE (open rotor) more than 20 %
    # for a mid-2030s EIS; today's large turbofans reach about 40 % overall at
    # cruise, about 50 % is achievable and 67 % is the ideal limit (Grönstedt 2019).
    "engine_efficiency_factor": (
        (100.0, 108.0, 115.0, 118.0),
        (100.0, 115.0, 125.0, 130.0),
        (100.0, 125.0, 135.0, 140.0),
    ),
    # %, standard empty mass relative to today. About 10-20 % further savings are
    # supported by sources (787/A350 banked 20 % of the structure already), not 34-50 %
    "struct_weight_factor": (
        (100.0, 94.0, 91.0, 90.0),
        (100.0, 90.0, 86.0, 85.0),
        (100.0, 86.0, 81.0, 78.0),
    ),
    # SP ~ P^-beta, from the Mid value in 2020 to the scenario value in 2040
    "emotor_power_exponent": (
        (0.1, 0.25, 0.25, 0.25),
        (0.1, 0.1, 0.1, 0.1),
        (0.1, 0.0, 0.0, 0.0),
    ),
    # 1 - eta ~ P^-gamma
    "emotor_loss_exponent": (
        (0.1, 0.0, 0.0, 0.0),
        (0.1, 0.1, 0.1, 0.1),
        (0.1, 0.25, 0.25, 0.25),
    ),
    # core SP ~ P^+exponent
    "fuelcell_power_exponent": (
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.15, 0.15, 0.15),
    ),
}

LOG_SCALE_TECH_PARAMS = {"max_unit_power"}
"""Technology parameters interpolated in log space."""

# LH2 tank technology, as logistic curves (v_0 in 2020, v_inf, t_50, tau) of the EIS
# year: tank mass factor k on the size law of present-day aluminium tanks, and fuel
# system mass per kg of LH2 s. The scenarios share their timing and differ by their
# final technology. Fitted by noads.application.lh2_tank_calibration on
# aircraft_tech_data/lh2_tank_gi/lh2_tank_gi_dataset.csv.
lh2_tank_tech_params_lower_mid_upper = {
    "lh2tank_mass_factor": (
        (5.341, 1.5, 2038.0, 5.0),
        (5.341, 1.094, 2038.0, 5.0),
        (5.341, 0.798, 2038.0, 5.0),
    ),
    "lh2_fuel_system_ratio": (
        (0.271, 0.201, 2052.1, 5.0),
        (0.271, 0.153, 2052.1, 5.0),
        (0.271, 0.115, 2052.1, 5.0),
    ),
}

# Number of LH2 tanks per aircraft, per market (tank size effect)
lh2_tank_count = {
    "general": 2,
    "commuter": 2,
    "regional": 2,
    "short_medium": 2,
    "long_range": 2,
}

# Propulsors per fuel cell aircraft, as in credible concepts: 2 up to about 80 seats
# (ZeroAvia ZA600 and ZA2000, ICCT ATR 72 retrofit), 4 for about 100 seats and more
# (Airbus ZEROe 2025: 4 x 2 MW). 8 or more only appear in NASA research concepts.
fuelcell_engine_count = {
    "general": 2,
    "commuter": 2,
    "regional": 2,
    "short_medium": 4,
    "long_range": 4,
}

# Turboprop architectures, compared with the fleet architectures in the update
# prospective aircraft figures (GAM V3.0 turboshaft efficiency grows with size).
# They are not part of the optimized fleets.
update_comparison_architectures = {
    "JetA-Turboprop": {
        "engine_count": 2,
        "engine_type": "turboprop",
        "thruster_type": "propeller",
        "energy_type": "kerosene",
    },
    "lH2-Turboprop": {
        "engine_count": 2,
        "engine_type": "turboprop",
        "thruster_type": "propeller",
        "energy_type": "liquid_h2",
    },
}
update_comparison_mission = {
    "JetA-Turboprop": {"speed": 0.5 * 340, "altitude": 20000 * 0.3048},
    "lH2-Turboprop": {"speed": 0.5 * 340, "altitude": 20000 * 0.3048},
}


def aircraft_tech_params(technology_index, aircraft_model="paper"):
    """Aircraft technology parameters of a technology scenario.

    Args:
        technology_index: The technology scenario (0: Lower, 1: Mid, 2: Upper).
        aircraft_model: ``"paper"`` or ``"update"``.

    Returns:
        The time-evolving technology parameters.
    """
    if aircraft_model == "paper":
        return [
            AircraftTechParameter(name, tuple(values[technology_index]))
            for name, values in tech_params_lower_mid_upper_2020_2040_2060.items()
        ]
    update_params = update_tech_params_lower_mid_upper_2020_2040_2060_2080
    params = [
        ScenarioTechParameter(
            name,
            values,
            technology_index,
            log_scale=name in LOG_SCALE_TECH_PARAMS,
        )
        for name, values in update_params.items()
    ]
    params.extend(
        LogisticTechParameter(name, values[technology_index])
        for name, values in lh2_tank_tech_params_lower_mid_upper.items()
    )
    return params


def update_power_system(architecture, category):
    """Power system of an architecture for a market with the update aircraft model."""
    power_system = dict(
        propulsion_architectures.get(architecture)
        or update_comparison_architectures[architecture]
    )
    if architecture == "lH2-FuelCell":
        power_system["engine_count"] = fuelcell_engine_count[category]
    if power_system["energy_type"] == "liquid_h2":
        power_system["tank_count"] = lh2_tank_count[category]
    return power_system


UNIT_POWER_RATIO_MARGIN = 0.98
"""Largest unit power ratio at the last EIS for a design to enter the update fleet."""


def update_design_feasible(
    prop_name, cat_name, technology_index, last_entry_into_service
):
    """Whether an aircraft of the update model can be designed by the last EIS.

    Technology only improves with the entry-into-service, so the design at the last
    EIS is the best case: it must close, have its power per propulsor available, and
    be at least as energy efficient as the current fleet of its market, as required
    by the optimization constraints.

    Args:
        prop_name: The propulsion architecture.
        cat_name: The market.
        technology_index: The technology scenario (0: Lower, 1: Mid, 2: Upper).
        last_entry_into_service: The last entry-into-service year.

    Returns:
        Whether the design is feasible at the last entry-into-service.
    """
    mission = {
        **categories_mission[cat_name],
        **propulsion_mission[prop_name],
        "category": cat_name,
    }
    gam = gam_v3.GAM(
        **{
            param.name: param.value_at_entry_into_service(last_entry_into_service)
            for param in aircraft_tech_params(technology_index, "update")
        },
        **UPDATE_GAM_OPTIONS,
    )
    design = gam.design_airplane(update_power_system(prop_name, cat_name), mission)
    energy_per_ask = 1.0e-3 * float(design["enrg_consumption"])
    return (
        float(design["closed"]) > 0.5
        and float(design["unit_power_ratio"]) <= UNIT_POWER_RATIO_MARGIN
        and energy_per_ask <= category_conso[cat_name][technology_index]
    )


def initialize_base_objects(
    drop_in_only=False,
    technology_index=0,
    aircraft_model="paper",
    last_entry_into_service=2060.0,
):
    """Build the energy mix and global fleet of the paper's scenarios.

    Args:
        drop_in_only: Whether to restrict the system to drop-in (Jet-A) aircraft,
            excluding hydrogen and battery-electric architectures and their energy
            supply chains.
        technology_index: The aircraft technology scenario (0: Lower, 1: Mid,
            2: Upper), selecting the component technology parameters, the
            current-fleet consumption quartile, and the fleet lifetimes.
        aircraft_model: The aircraft design model: ``"paper"`` (GAM V2.0, as in
            the paper) or ``"update"`` (GAM V3.0 with powertrain scale effects,
            fuel cell propulsor counts, maximum power per propulsor, and
            size-dependent LH2 tanks).
        last_entry_into_service: The last entry-into-service of new aircraft. With
            the update model, the electric and fuel cell aircraft of a market are
            included only if they are feasible by then
            (:func:`update_design_feasible`).

    Returns:
        The energy mix and the fleet assembly.

    Raises:
        RuntimeError: If the technology index is not 0, 1 or 2.
        ValueError: If the aircraft model is unknown.
    """
    if technology_index < 0 or technology_index > 2:
        msg = "Please enter 0, 1 or 2 as technology index (low, mid, up)"
        raise RuntimeError(msg)
    if aircraft_model not in AIRCRAFT_MODELS:
        msg = f"aircraft_model must be one of {AIRCRAFT_MODELS}"
        raise ValueError(msg)

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
    tech_params = aircraft_tech_params(technology_index, aircraft_model)

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
            if aircraft_model == "update":
                power_system = update_power_system(prop_name, cat_name)
            else:
                power_system = propulsion_architectures[prop_name]
            if aircraft_model == "update" and (
                "Electric" in prop_name or "FuelCell" in prop_name
            ):
                included = update_design_feasible(
                    prop_name, cat_name, technology_index, last_entry_into_service
                )
            elif "Electric" in prop_name:
                # Electric aircraft always included for general market, commuter
                # market only if not lower technology
                included = cat_name == "general" or (
                    cat_name == "commuter" and technology_index > 0
                )
            else:
                included = True
            if not included:
                continue
            if "JetA-GasTurbine" in prop_name:
                aircraft.extend((
                    AircraftDesign(
                        name=f"{prop_name}-v1_{cat_name}",
                        propulsion=prop_systems[prop_name],
                        mission=mission,
                        power_system=power_system,
                        aircraft_tech_params=tech_params,
                        aircraft_model=aircraft_model,
                        reference_aircraft=aircraft_reference_2019,
                    ),
                    AircraftDesign(
                        name=f"{prop_name}-v2_{cat_name}",
                        propulsion=prop_systems[prop_name],
                        mission=mission,
                        power_system=power_system,
                        aircraft_tech_params=tech_params,
                        aircraft_model=aircraft_model,
                        reference_aircraft=aircraft_reference_2019,
                    ),
                ))
            else:
                aircraft.append(
                    AircraftDesign(
                        name=f"{prop_name}_{cat_name}",
                        propulsion=prop_systems[prop_name],
                        mission=mission,
                        power_system=power_system,
                        aircraft_tech_params=tech_params,
                        aircraft_model=aircraft_model,
                        reference_aircraft=aircraft_reference_2019,
                    )
                )

        # Create the fleet object assembling all aircraft
        fleets.append(Fleet(cat_name, operating_aircraft=aircraft))
    fleet = FleetAssembly(fleets)

    return energy_mix, fleet
