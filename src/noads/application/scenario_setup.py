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
"""Setup model structure, default inputs and design space for optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo import create_design_space
from gemseo_jax.jax_discipline import JAXDiscipline
from numpy import array as np_array
from numpy import ones as np_ones

from noads.application.background_scenario_data import get_ar6_data
from noads.application.base_objects import initialize_base_objects
from noads.core.models.interpolation import interpolate_data
from noads.core.models.traffic import AirTraffic
from noads.core.scenarios.multiscenario import MultiScenario
from noads.core.scenarios.temporalscenario import TemporalScenario

if TYPE_CHECKING:
    from collections.abc import Sequence

refuel_eu_years = [2025, 2030, 2035, 2040, 2045, 2050, 2055, 2060]
refuel_eu_biofuel = [0.02, 0.04, 0.15, 0.24, 0.27, 0.35, 0.7, 1.0]
refuel_eu_efuel = [0.01, 0.02, 0.05, 0.10, 0.15, 0.35, 0.7, 1.0]


def single_scenario_setup(
    scenario_name: str,
    start_year=2025,
    end_year=2075,
    time_step=1.0,
    interp_step=5.0,
    technology_index=0,
    integrate_constraints=False,
    demand_aversion=False,
    drop_in_only=False,
    preferential_energy=False,
    plot_scenario_data=False,
    compile_jit=True,
):
    """Setup decarbonization scenario based on a single objective."""
    resources_fair_share = 8.6e-2 if preferential_energy else 5.0e-2
    ar6_data, years_data = get_ar6_data(plot_data=plot_scenario_data)
    energy_mix, fleet = initialize_base_objects(drop_in_only, technology_index)

    temporal_constraints = [
        f"{stream.name}.constraint" for stream in energy_mix.constrained_inputs
    ]
    temporal_constraints.extend([
        f"{energy.name}.controls_constraint"
        for energy in energy_mix.produced_energies
        if len(energy.pathways) > 1
    ])
    temporal_constraints.extend([
        f"{fleet_i.name}.controls_constraint" for fleet_i in fleet.fleets
    ])

    time_integrated_outputs = ["ask", "rpk"]
    time_integrated_outputs.extend([impact.name for impact in energy_mix.impacts])
    if integrate_constraints:
        for i in range(len(temporal_constraints)):
            temporal_constraints[i] += "_violation"
        time_integrated_outputs.extend(list(temporal_constraints))
    if demand_aversion:
        time_integrated_outputs.extend([
            "discounted_relative_price_change",
        ])

    constants = {
        "load_factor_end_year": 92.0,
        "gdp_per_capita_covid_end": float(
            interpolate_data(
                x=2024.0,
                x_data=np_array(years_data),
                y_data=np_array(ar6_data["gdp_per_capita"][scenario_name]),
                cubic=True,
            )
        ),
        "discount_rate": 0.04,
        "start_year": start_year,
        "end_year": end_year,
        "price_elasticity": -0.6,
        "ELECTRICITY.fair_share": resources_fair_share,
        "BIOMASS.fair_share": resources_fair_share,
        "commuter.share": 1e-2 * 22.133471276729583,
        "regional.share": 1e-2 * 38.55054251368191,
        "short_medium.share": 1e-2 * 18.32169512462106,
        "long_range.share": 1e-2 * 18.253503304539215,
        "OIL.CO2_index": 73.2 * 0.865,
        # This is made just to match 73.2 gCO2/MJ as emission factor for kerosene
        # (without accounting for lifecycle)
        "BIOMASS.CO2_index": 0.0,
        "Refinery.direct.CO2_index": 88.7 - 73.2,
        # https://doi.org/10.1021/es2017522 uses 73.2 as CO2 emission index, yet
        # https://doi.org/10.1038/s41467-022-35392-1 shows that accounting for lifecycle
        # this value goes up to 88.7 in a global mean (varies with location).
        "HEFA.direct.CO2_index": 62.75,
        # mean between HEFA-PO and HEFA-JO https://doi.org/10.1016/j.fuproc.2017.09.022
        "FT.direct.CO2_index": 35.3,
        # mean BtL-Wi and BtL-WS
        "ATJ.direct.CO2_index": 51.55,
        # ATJ-WG and AtJ-WS
        "Electrolysis.direct.CO2_index": 0.0,
        "Gas_reforming.direct.CO2_index": 100.0,
        "Power_to_liquid.direct.CO2_index": 0.0,
        "Electrofuel.direct.CO2_index": 0.0,
        "Biofuel.direct.CO2_index": 0.0,
        "Fossil.direct.CO2_index": 0.0,
        "Refinery.OIL.efficiency": 0.865,
        # from https://publications.anl.gov/anlpubs/2011/01/69026.pdf
        "HEFA.BIOMASS.efficiency": 0.59,  # mean between HEFA-PO and HEFA-JO
        "FT.BIOMASS.efficiency": 0.20,  # BtL-Wi and BtL-WS
        "ATJ.BIOMASS.efficiency": 0.30,  # ATJ-WG and AtJ-WS
        "Biofuel.BIOFUEL.efficiency": 1.0,
        "Electrofuel.E-FUEL.efficiency": 1.0,
        "Fossil.KEROSENE.efficiency": 1.0,
    }
    # interpolated values are from https://doi.org/10.1016/j.joule.2024.07.012.
    interpolated_2025_2035_2050 = {
        "Electrolysis.ELECTRICITY.efficiency": (0.71, 0.71 * 1.03, 0.71 * 1.06),
        "Power_to_liquid.ELECTRICITY.efficiency": (1.53, 1.53 * 1.08, 1.53 * 1.16),
        "Power_to_liquid.GAS-H2.efficiency": (0.53, 0.53 * 1.06, 0.53 * 1.12),
    }
    if not drop_in_only:
        constants.update({
            "H2_liquefaction.direct.CO2_index": 0.0,
            "H2_liquefaction.GAS-H2.efficiency": 1.0,
            "Charging.direct.CO2_index": 0.0,
            "Charging.ELECTRICITY.efficiency": 0.98,
        })
        interpolated_2025_2035_2050.update({
            "H2_liquefaction.ELECTRICITY.efficiency": (4.54, 4.54 * 1.2, 4.54 * 1.4),
        })
        # if include_methane:
        #     constants.update({
        #         "NATURAL_GAS.CO2_index": 67.6,
        #         "GM_methanation.direct.CO2_index": 0.0,
        #         "GM_biogas.direct.CO2_index": 14.3,
        #         "GM_fossil.direct.CO2_index": 0.0,
        #         "LM_liquefaction.direct.CO2_index": 0.0,
        #         "GM_fossil.NATURAL_GAS.efficiency": 1.0,
        #         "GM_methanation.GAS-H2.efficiency": 0.89,
        #         "LM_liquefaction.ELECTRICITY.efficiency": 20.0,
        #         "LM_liquefaction.GAS-CH4.efficiency": 1.0,
        #     })
        #     interpolated_2025_2035_2050.update({
        #         "GM_biogas.BIOMASS.efficiency": (0.7, 0.75, 0.8)
        #     })

    controls_delay_times = {}
    constrained_control_groups = {}
    custom_controls = []
    for energy in energy_mix.produced_energies:
        controls_in_group = []
        for pathway in energy.pathways:
            if pathway != energy.pathways[-1]:
                controls_delay_times.update({f"{pathway.name}.share": 5.0})
                controls_in_group.append(f"{pathway.name}.share")
        if controls_in_group:
            constrained_control_groups.update({energy.name: controls_in_group})
    fleet_lifetimes = [15.0, 20.0, 25.0, 25.0, 25.0]
    # https://aviation.report/Resources/Whitepapers/c7ca1e8f-fd11-4a96-9500-85609082abf7_whitepaper%201.pdf
    for i, fleet_i in enumerate(fleet.fleets):
        controls_delay_times.update({f"{fleet_i.name}.demand_shift_ratio": 7.5})
        controls_in_group = []
        for aircraft in fleet_i.operating_aircraft:
            if aircraft != fleet_i.operating_aircraft[0]:
                aircraft_tau = fleet_lifetimes[i] / 4.0
                constants.update({
                    f"{aircraft.name}.max_share": 0.0,
                    f"{aircraft.name}.entry_into_service": 2050.0,
                    f"{aircraft.name}.ramp_up_duration": 2 * aircraft_tau,
                    f"{aircraft.name}.ramp_down_duration": 2 * aircraft_tau,
                    f"{aircraft.name}.lifetime": fleet_lifetimes[i],
                })
                controls_delay_times.update({f"{aircraft.name}.share": aircraft_tau})
                controls_in_group.append(f"{aircraft.name}.share")
                custom_controls.append(f"{aircraft.name}.share")
        if controls_in_group:
            constrained_control_groups.update({fleet_i.name: controls_in_group})

    models = [AirTraffic()]
    models.extend(energy_mix.models)
    models.extend(fleet.models)

    temporal_scenario = TemporalScenario(
        name="Single scenario",
        models=models,
        constant_inputs=list(constants.keys()),
        control_delay_times=controls_delay_times,
        constrained_control_groups=constrained_control_groups,
        custom_controls=custom_controls,
        interpolated_inputs=list(interpolated_2025_2035_2050.keys()),
        time_integrated_outputs=time_integrated_outputs,
        start_year=start_year,
        end_year=end_year,
        time_step=time_step,
        interp_step=interp_step,
        adjoint_method=TemporalScenario.AdjointMethod.BACKSOLVE
        if integrate_constraints
        else TemporalScenario.AdjointMethod.DIRECT,
        differentiation_method=JAXDiscipline.DifferentiationMethod.REVERSE
        if integrate_constraints
        else JAXDiscipline.DifferentiationMethod.FORWARD,
    )
    scenario_inputs = {"year": temporal_scenario.time_vector}
    scenario_inputs.update({
        f"constant.{name}": np_array([value]) for name, value in constants.items()
    })
    scenario_inputs.update({
        f"interpolate.{name}": np_array(
            interpolate_data(
                x=temporal_scenario.time_interpolation,
                x_data=np_array([2025.0, 2035.0, 2050.0]),
                y_data=np_array([tup[0], tup[1], tup[2]]),
            )
        )
        for name, tup in interpolated_2025_2035_2050.items()
    })
    scenario_inputs.update({
        f"control.{name}": 0.0 * np_ones(temporal_scenario.time_interpolation.shape)
        for name in controls_delay_times
        if name not in custom_controls
    })
    scenario_inputs.update({
        name: np_array(
            interpolate_data(
                x=temporal_scenario.time_vector,
                x_data=np_array(years_data),
                y_data=np_array(ar6_data[name][scenario_name]),
                cubic=True,
            )
        )
        for name in ar6_data
    })
    temporal_scenario.discipline.default_input_data = scenario_inputs
    if compile_jit:
        temporal_scenario.discipline.compile_jit(pre_run=False)

    design_space = create_design_space()
    for energy in energy_mix.produced_energies:
        for pathway in energy.pathways:
            if pathway != energy.pathways[-1]:
                if pathway.name == "Biofuel":
                    upper_value = interpolate_data(
                        x=temporal_scenario.time_interpolation,
                        x_data=np_array(refuel_eu_years),
                        y_data=np_array(refuel_eu_biofuel),
                    )
                    value = 0.1 * upper_value
                elif pathway.name == "Electrofuel":
                    upper_value = interpolate_data(
                        x=temporal_scenario.time_interpolation,
                        x_data=np_array(refuel_eu_years),
                        y_data=np_array(refuel_eu_efuel),
                    )
                    value = 0.1 * upper_value
                else:
                    n_pathways = len(energy.pathways)
                    upper_value = 1.0
                    value = 1.0 / n_pathways

                design_space.add_variable(
                    f"control.{pathway.name}.share",
                    size=temporal_scenario.time_interpolation.size,
                    lower_bound=0.0,
                    upper_bound=upper_value,
                    value=value,
                )
    for i, fleet_i in enumerate(fleet.fleets):
        if demand_aversion:
            design_space.add_variable(
                f"control.{fleet_i.name}.demand_shift_ratio",
                size=temporal_scenario.time_interpolation.size,
                lower_bound=0.0,
                upper_bound=0.6,
                # * (np_ones(
                #     temporal_scenario.time_interpolation.shape
                # ) - np_eye(1, temporal_scenario.time_interpolation.size)[0]),
                value=0.4,
                # * (np_ones(
                #     temporal_scenario.time_interpolation.shape
                # ) - np_eye(1, temporal_scenario.time_interpolation.size)[0]),
            )
        for aircraft in fleet_i.operating_aircraft:
            fleet_tau = fleet_lifetimes[i] / 4.0
            if aircraft != fleet_i.operating_aircraft[0]:
                design_space.add_variable(
                    f"constant.{aircraft.name}.max_share",
                    lower_bound=0.0,
                    upper_bound=1.0,
                    value=0.1,
                )
                design_space.add_variable(
                    f"constant.{aircraft.name}.entry_into_service",
                    lower_bound=2035.0,
                    upper_bound=2060.0,
                    value=2060.0,
                )
                design_space.add_variable(
                    f"constant.{aircraft.name}.lifetime",
                    lower_bound=3.0 * fleet_tau,
                    upper_bound=16.0 * fleet_tau,
                    value=13.0 * fleet_tau,
                )
                design_space.add_variable(
                    f"constant.{aircraft.name}.ramp_up_duration",
                    lower_bound=2.0 * fleet_tau,
                    upper_bound=4.0 * fleet_tau,
                    value=4.0 * fleet_tau,
                )
                design_space.add_variable(
                    f"constant.{aircraft.name}.ramp_down_duration",
                    lower_bound=2.0 * fleet_tau,
                    upper_bound=4.0 * fleet_tau,
                    value=4.0 * fleet_tau,
                )

    optimization_constraints = {
        f"cumulative.{name}" if integrate_constraints else name: (0.0, False)
        for name in temporal_constraints
    }
    for aircraft in fleet.operating_aircraft:
        if "Electric" in aircraft.name:
            optimization_constraints.update({
                f"{aircraft.name}.relative_efficiency_gain": (1.0, True),
            })

    return temporal_scenario, design_space, optimization_constraints, energy_mix, fleet


def multi_scenario_setup(
    scenario_names: Sequence[str],
    start_year=2025,
    end_year=2075,
    time_step=2.0,
    interp_step=5.0,
    technology_index=0,
    aggregate_constraints=False,
    integrate_constraints=False,
    demand_aversion=False,
    drop_in_only=False,
    preferential_energy=False,
    plot_scenario_data=False,
):
    """Setup decarbonization scenario robust to several background scenarios."""
    temporal_scenario, _, constraints, energy_mix, fleet = single_scenario_setup(
        scenario_name=scenario_names[0],
        start_year=start_year,
        end_year=end_year,
        time_step=time_step,
        interp_step=interp_step,
        technology_index=technology_index,
        integrate_constraints=integrate_constraints,
        demand_aversion=demand_aversion,
        drop_in_only=drop_in_only,
        preferential_energy=preferential_energy,
        plot_scenario_data=plot_scenario_data,
    )
    meaned = [
        f"cumulative.{name}" for name in temporal_scenario.time_integrated_outputs
    ]
    meaned.extend(temporal_scenario.time_integrated_outputs)
    if aggregate_constraints:
        meaned.extend([name for name in constraints if name not in meaned])

    fixed = ["year"]
    fixed.extend([
        f"constant.{name}"
        for name in temporal_scenario.constant_inputs
        if name != "gdp_per_capita_covid_end"
    ])
    fixed.extend([
        f"interpolate.{name}" for name in temporal_scenario.interpolated_inputs
    ])
    # all except controls

    ar6_data, years_data = get_ar6_data()

    multi_scenario = MultiScenario(
        temporal_scenario=temporal_scenario,
        mean_outputs=meaned,
        scenario_names=scenario_names,
        fixed_inputs=fixed,
    )
    multi_scenario_inputs = {
        f"fixed.{name}": temporal_scenario.discipline.default_input_data[name]
        for name in fixed
    }
    multi_scenario_inputs.update({
        f"{scenario}.{name}": temporal_scenario.discipline.default_input_data[name]
        for scenario in scenario_names
        for name in temporal_scenario.discipline.input_grammar.names
        if name not in fixed
    })
    # reupdate inputs with ar6 data
    multi_scenario_inputs.update({
        f"{scenario}.{name}": np_array(
            interpolate_data(
                x=temporal_scenario.time_vector,
                x_data=np_array(years_data),
                y_data=np_array(ar6_data[name][scenario]),
                cubic=True,
            )
        )
        for scenario in scenario_names
        for name, scenario_dict in ar6_data.items()
    })
    multi_scenario_inputs.update({
        f"{scenario}.constant.gdp_per_capita_covid_end": np_array([
            interpolate_data(
                x=2024.0,
                x_data=np_array(years_data),
                y_data=np_array(ar6_data["gdp_per_capita"][scenario]),
            )
        ])
        for scenario in scenario_names
    })
    multi_scenario.discipline.default_input_data = multi_scenario_inputs
    multi_scenario.discipline.compile_jit(pre_run=False)

    design_space = create_design_space()
    for scenario in scenario_names:
        for energy in energy_mix.produced_energies:
            for pathway in energy.pathways:
                if pathway != energy.pathways[-1]:
                    if pathway.name == "Biofuel":
                        upper_value = interpolate_data(
                            x=temporal_scenario.time_interpolation,
                            x_data=np_array(refuel_eu_years),
                            y_data=np_array(refuel_eu_biofuel),
                        )
                        value = 0.5 * upper_value
                    elif pathway.name == "Electrofuel":
                        upper_value = interpolate_data(
                            x=temporal_scenario.time_interpolation,
                            x_data=np_array(refuel_eu_years),
                            y_data=np_array(refuel_eu_efuel),
                        )
                        value = 0.5 * upper_value
                    else:
                        n_pathways = len(energy.pathways)
                        upper_value = 1.0
                        value = 1.0 / n_pathways

                    design_space.add_variable(
                        f"{scenario}.control.{pathway.name}.share",
                        size=temporal_scenario.time_interpolation.size,
                        lower_bound=0.0,
                        upper_bound=upper_value,
                        value=value,
                    )

        for fleet_i in fleet.fleets:
            if demand_aversion:
                design_space.add_variable(
                    f"{scenario}.control.{fleet_i.name}.demand_shift_ratio",
                    size=temporal_scenario.time_interpolation.size,
                    lower_bound=0.0,
                    upper_bound=0.6,
                    # * (np_ones(
                    #     temporal_scenario.time_interpolation.shape
                    # ) - np_eye(1, temporal_scenario.time_interpolation.size)[0]),
                    value=0.4,
                    # * (np_ones(
                    #     temporal_scenario.time_interpolation.shape
                    # ) - np_eye(1, temporal_scenario.time_interpolation.size)[0]),
                )
    fleet_lifetimes = [15.0, 20.0, 25.0, 25.0, 25.0]
    for i, fleet_i in enumerate(fleet.fleets):
        fleet_tau = fleet_lifetimes[i] / 4.0
        for aircraft in fleet_i.operating_aircraft:
            if aircraft != fleet_i.operating_aircraft[0]:
                design_space.add_variable(
                    f"fixed.constant.{aircraft.name}.entry_into_service",
                    lower_bound=2035.0,
                    upper_bound=2060.0,
                    value=2060.0,
                )
                design_space.add_variable(
                    f"fixed.constant.{aircraft.name}.max_share",
                    lower_bound=0.0,
                    upper_bound=1.0,
                    value=0.1,
                )
                design_space.add_variable(
                    f"fixed.constant.{aircraft.name}.lifetime",
                    lower_bound=3.0 * fleet_tau,
                    upper_bound=16.0 * fleet_tau,
                    value=3.0 * fleet_tau,
                )
                design_space.add_variable(
                    f"fixed.constant.{aircraft.name}.ramp_up_duration",
                    lower_bound=2.0 * fleet_tau,
                    upper_bound=4.0 * fleet_tau,
                    value=2.0 * fleet_tau,
                )
                design_space.add_variable(
                    f"fixed.constant.{aircraft.name}.ramp_down_duration",
                    lower_bound=2.0 * fleet_tau,
                    upper_bound=4.0 * fleet_tau,
                    value=2.0 * fleet_tau,
                )

    optimization_constraints = {}
    for name, value in constraints.items():
        if aggregate_constraints:
            optimization_constraints[f"mean.{name}"] = value
        else:
            for scenario in scenario_names:
                optimization_constraints[f"{scenario}.{name}"] = value

    return multi_scenario, design_space, optimization_constraints, energy_mix, fleet
