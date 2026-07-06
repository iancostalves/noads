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

"""Setup model structure, default inputs and design space for optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo import create_design_space
from gemseo_jax.jax_discipline import JAXDiscipline
from numpy import array as np_array
from numpy import ones as np_ones

from noads.application.background_scenario_data import get_ar6_input_data
from noads.application.base_objects import category_lifetime
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
    name: str,
    background_scenario_name: str,
    start_year=2025,
    end_year=2075,
    time_step=1.0,
    interp_step=2.5,
    technology_index=0,
    integrate_constraints=False,
    demand_aversion=False,
    drop_in_only=False,
    fossil_kerosene_only=False,
    preferential_energy=False,
    plot_scenario_data=False,
    compile_jit=True,
):
    """Set up a single-objective decarbonization scenario.

    Builds the fleet and energy-mix objects, loads the AR6 background data of the
    chosen scenario, assembles everything into a
    :class:`~noads.core.scenarios.temporalscenario.TemporalScenario`, fills in the
    default inputs (pathway efficiencies and emission indices, market shares, fair
    shares, demand-model constants), and creates the design space (aircraft
    entry-into-service and maximum market shares, energy pathway shares, and, with
    ``demand_aversion``, the market supply-shift ratios) together with the
    optimization constraints.

    Args:
        name: The name of the scenario.
        background_scenario_name: The AR6 background scenario (e.g. ``"SSP2-26"``).
        start_year: The first simulated year.
        end_year: The last simulated year.
        time_step: The simulation time step in years.
        interp_step: The spacing in years of the control knots.
        technology_index: The aircraft technology scenario (0: Lower, 1: Mid,
            2: Upper).
        integrate_constraints: Whether to time-integrate constraint violations
            instead of imposing path-wise constraints.
        demand_aversion: Whether to use the low-demand formulation (supply caps and
            discounted price-increase burden).
        drop_in_only: Whether to restrict the fleet to drop-in (Jet-A) aircraft.
        fossil_kerosene_only: Whether to restrict Jet-A to fossil kerosene.
        preferential_energy: Whether to use the preferential (8.6 %) instead of the
            conservative (5.0 %) fair share of biomass and electricity.
        plot_scenario_data: Whether to plot the AR6 background data.
        compile_jit: Whether to JIT-compile the assembled discipline.

    Returns:
        The temporal scenario, the design space, the constraints (mapping each
        constraint name to its bound and sign), the energy mix, and the fleet.
    """
    resources_fair_share = 8.6e-2 if preferential_energy else 5.0e-2
    ar6_data, years_data = get_ar6_input_data(plot_data=plot_scenario_data)
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

    capacity_factor = 1.0
    if "SSP5" in background_scenario_name:
        capacity_factor = 1.5
    elif "SSP1" in background_scenario_name:
        capacity_factor = 0.9
    constants = {
        "capacity_factor": capacity_factor,
        "load_factor_end_year": 92.0,
        "gdp_per_capita_2019": float(
            interpolate_data(
                x=2019.0,
                x_data=np_array(years_data),
                y_data=np_array(ar6_data["gdp_per_capita"][background_scenario_name]),
                cubic=True,
            )
        ),
        "gdp_per_capita_covid_end": float(
            interpolate_data(
                x=2024.0,
                x_data=np_array(years_data),
                y_data=np_array(ar6_data["gdp_per_capita"][background_scenario_name]),
                cubic=True,
            )
        ),
        "discount_rate": 3.0e-2,
        "start_year": start_year,
        "end_year": end_year,
        "price_elasticity": -0.25587624,
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
    fleet_lifetimes = [
        tech_tup[technology_index] for tech_tup in category_lifetime.values()
    ]
    # https://aviation.report/Resources/Whitepapers/c7ca1e8f-fd11-4a96-9500-85609082abf7_whitepaper%201.pdf
    final_rates = []
    for i, fleet_i in enumerate(fleet.fleets):
        controls_delay_times.update({f"{fleet_i.name}.demand_shift_ratio": 10.0})
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
            else:
                final_rates.append(f"{aircraft.name}.share")
        if controls_in_group:
            constrained_control_groups.update({fleet_i.name: controls_in_group})

    models = [AirTraffic()]
    models.extend(energy_mix.models)
    models.extend(fleet.models)

    temporal_scenario = TemporalScenario(
        name=name,
        models=models,
        constant_inputs=list(constants.keys()),
        control_delay_times=controls_delay_times,
        constrained_control_groups=constrained_control_groups,
        custom_controls=custom_controls,
        final_rates=final_rates,
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
                y_data=np_array(ar6_data[name][background_scenario_name]),
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
                    upper_value = (
                        0.0
                        if fossil_kerosene_only
                        else interpolate_data(
                            x=temporal_scenario.time_interpolation,
                            x_data=np_array(refuel_eu_years),
                            y_data=np_array(refuel_eu_biofuel),
                        )
                    )
                    value = 0.1 * upper_value
                elif pathway.name == "Electrofuel":
                    upper_value = (
                        0.0
                        if fossil_kerosene_only
                        else interpolate_data(
                            x=temporal_scenario.time_interpolation,
                            x_data=np_array(refuel_eu_years),
                            y_data=np_array(refuel_eu_efuel),
                        )
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
                upper_bound=0.9,
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
                    lower_bound=(
                        2032.0 if "JetA-GasTurbine-v1" in aircraft.name else 2035.0
                    ),
                    upper_bound=(
                        2047.5 if "JetA-GasTurbine-v1" in aircraft.name else 2060.0
                    ),
                    # value=2060.0,
                    value=2035.0
                    if (
                        "JetA-GasTurbine-v1" in aircraft.name
                        or "lH2-GasTurbine" in aircraft.name
                    )
                    else 2060.0,
                )
                design_space.add_variable(
                    f"constant.{aircraft.name}.lifetime",
                    lower_bound=3.0 * fleet_tau,
                    upper_bound=45.0,
                    value=4.0 * fleet_tau,
                )
                design_space.add_variable(
                    f"constant.{aircraft.name}.ramp_up_duration",
                    lower_bound=2.0 * fleet_tau,
                    upper_bound=3.0 * fleet_tau,
                    value=2.0 * fleet_tau,
                )
                design_space.add_variable(
                    f"constant.{aircraft.name}.ramp_down_duration",
                    lower_bound=2.0 * fleet_tau,
                    upper_bound=3.0 * fleet_tau,
                    value=2.0 * fleet_tau,
                )

    optimization_constraints = {
        f"cumulative.{name}" if integrate_constraints else name: (0.0, False)
        for name in temporal_constraints
    }
    optimization_constraints.update({
        f"final_rate.{name_i}": (0.0, False) for name_i in final_rates
    })
    for aircraft in fleet.operating_aircraft:
        if "Electric" in aircraft.name:
            optimization_constraints.update({
                f"{aircraft.name}.relative_efficiency_gain": (1.0, True),
            })

    return temporal_scenario, design_space, optimization_constraints, energy_mix, fleet


def multi_scenario_setup(
    name: str,
    background_scenario_names: Sequence[str],
    start_year=2025,
    end_year=2075,
    time_step=2.0,
    interp_step=5.0,
    technology_index=0,
    aggregate_constraints=False,
    integrate_constraints=False,
    demand_aversion=False,
    fossil_kerosene_only=False,
    drop_in_only=False,
    preferential_energy=False,
    plot_scenario_data=False,
):
    """Set up a decarbonization scenario robust to several background scenarios.

    Same as :func:`single_scenario_setup`, but the temporal scenario is batched
    across the given ensemble of background scenarios with a
    :class:`~noads.core.scenarios.multiscenario.MultiScenario`: the aircraft mix
    variables are shared across the ensemble (``fixed.`` prefix) while the energy
    mix variables are scenario-specific, and ensemble-mean outputs (``mean.``
    prefix) are exposed for the robust objective.
    """
    temporal_scenario, _, constraints, energy_mix, fleet = single_scenario_setup(
        name=name,
        background_scenario_name=background_scenario_names[0],
        start_year=start_year,
        end_year=end_year,
        time_step=time_step,
        interp_step=interp_step,
        technology_index=technology_index,
        integrate_constraints=integrate_constraints,
        demand_aversion=demand_aversion,
        fossil_kerosene_only=fossil_kerosene_only,
        drop_in_only=drop_in_only,
        preferential_energy=preferential_energy,
        plot_scenario_data=plot_scenario_data,
    )
    final_rates = temporal_scenario.final_rates
    meaned = [
        f"cumulative.{name}" for name in temporal_scenario.time_integrated_outputs
    ]
    meaned.extend(temporal_scenario.time_integrated_outputs)
    meaned.extend(final_rates)
    if aggregate_constraints:
        meaned.extend([name for name in constraints if name not in meaned])
        meaned.extend([
            f"final_rate.{name}"
            for name in final_rates
            if f"final_rate.{name}" not in meaned
        ])

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

    ar6_data, years_data = get_ar6_input_data(plot_data=False)

    multi_scenario = MultiScenario(
        name=name,
        temporal_scenario=temporal_scenario,
        mean_outputs=meaned,
        scenario_names=background_scenario_names,
        fixed_inputs=fixed,
    )
    multi_scenario_inputs = {
        f"fixed.{name}": temporal_scenario.discipline.default_input_data[name]
        for name in fixed
    }
    multi_scenario_inputs.update({
        f"{scenario}.{name}": temporal_scenario.discipline.default_input_data[name]
        for scenario in background_scenario_names
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
        for scenario in background_scenario_names
        for name in ar6_data
    })
    multi_scenario_inputs.update({
        f"{scenario}.constant.gdp_per_capita_covid_end": np_array([
            interpolate_data(
                x=2024.0,
                x_data=np_array(years_data),
                y_data=np_array(ar6_data["gdp_per_capita"][scenario]),
            )
        ])
        for scenario in background_scenario_names
    })
    multi_scenario.discipline.default_input_data = multi_scenario_inputs
    multi_scenario.discipline.compile_jit(pre_run=False)

    design_space = create_design_space()
    for scenario in background_scenario_names:
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
                    upper_bound=0.9,
                    # * (np_ones(
                    #     temporal_scenario.time_interpolation.shape
                    # ) - np_eye(1, temporal_scenario.time_interpolation.size)[0]),
                    value=0.4,
                    # * (np_ones(
                    #     temporal_scenario.time_interpolation.shape
                    # ) - np_eye(1, temporal_scenario.time_interpolation.size)[0]),
                )
    fleet_lifetimes = [
        tech_tup[technology_index] for tech_tup in category_lifetime.values()
    ]
    for i, fleet_i in enumerate(fleet.fleets):
        fleet_tau = fleet_lifetimes[i] / 4.0
        for aircraft in fleet_i.operating_aircraft:
            if aircraft != fleet_i.operating_aircraft[0]:
                design_space.add_variable(
                    f"fixed.constant.{aircraft.name}.entry_into_service",
                    lower_bound=(
                        2032.0 if "JetA-GasTurbine-v1" in aircraft.name else 2035.0
                    ),
                    upper_bound=2060.0,
                    value=2035.0
                    if (
                        "JetA-GasTurbine-v1" in aircraft.name
                        or "lH2-GasTurbine" in aircraft.name
                    )
                    else 2060.0,
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
                    upper_bound=45.0,
                    value=4.0 * fleet_tau,
                )
                design_space.add_variable(
                    f"fixed.constant.{aircraft.name}.ramp_up_duration",
                    lower_bound=2.0 * fleet_tau,
                    upper_bound=3.0 * fleet_tau,
                    value=2.0 * fleet_tau,
                )
                design_space.add_variable(
                    f"fixed.constant.{aircraft.name}.ramp_down_duration",
                    lower_bound=2.0 * fleet_tau,
                    upper_bound=3.0 * fleet_tau,
                    value=2.0 * fleet_tau,
                )

    optimization_constraints = {}
    for name, value in constraints.items():
        if aggregate_constraints:
            optimization_constraints[f"mean.{name}"] = value
        else:
            for scenario in background_scenario_names:
                optimization_constraints[f"{scenario}.{name}"] = value

    return multi_scenario, design_space, optimization_constraints, energy_mix, fleet
