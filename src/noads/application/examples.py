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

"""Utilities for launching scenario optimization examples."""

from json import dump
from json import load
from logging import getLogger
from os import environ
from pathlib import Path

from gemseo import configure_logger
from gemseo import create_scenario
from numpy import array

from noads.application.background_scenario_data import co2_budget_2p0deg_66percent
from noads.application.scenario_setup import multi_scenario_setup
from noads.application.scenario_setup import single_scenario_setup
from noads.application.visualization import plot_multi_scenario_result
from noads.application.visualization import plot_single_scenario_result

LOGGER = getLogger(__name__)


def _results_dir() -> Path:
    """Return the directory storing optimization results.

    Set the ``NOADS_RESULTS_DIR`` environment variable to share pre-computed
    results across working directories; defaults to ``results`` in the current
    working directory.
    """
    return Path(environ.get("NOADS_RESULTS_DIR", "results"))


def single_policy_scenario_optimization(
    global_scenario_name: str,
    carbon_budget_percent=3.0,
    technology_index=0,
    drop_in_only=False,
    fossil_kerosene_only=False,
    low_demand_formulation=False,
    preferential_energy=False,
    load_optimum=False,
    plot_optimum=True,
    save_optimum=False,
    save_history_view=False,
    save_figs=False,
    plot_computational_graphs=False,
):
    """Optimal decarbonization scenario based on a single objective.

    Args:
        global_scenario_name: Name of the background scenario.
        carbon_budget_percent: Percentage of CO2 budget to allocate.
        technology_index: Technology level (0=lower, 1=mid, 2=upper).
        drop_in_only: Use only drop-in fuels.
        fossil_kerosene_only: Use only fossil kerosene.
        low_demand_formulation: Use low-demand optimization formulation.
        preferential_energy: Apply preferential energy availability constraint.
        load_optimum: Load previously saved optimization results instead of re-running.
        plot_optimum: Plot the optimization results.
        save_optimum: Save the optimization results to file.
        save_history_view: Save optimization history visualization.
        save_figs: Save generated figures to files.
        plot_computational_graphs: Plot computational dependency graphs.

    Returns:
        Dictionary containing optimal output values.
    """
    # Build scenario name with suffixes
    scenario_name = global_scenario_name
    if fossil_kerosene_only:
        scenario_name += "-Fossil"
    elif drop_in_only:
        scenario_name += "-DropIn"

    if low_demand_formulation:
        scenario_name += "-LowDemand"
    if preferential_energy:
        scenario_name += "-Availability"

    if technology_index == 0:
        scenario_name += "-lowTech"
    elif technology_index == 1:
        scenario_name += "-midTech"
    elif technology_index == 2:
        scenario_name += "-upTech"

    configure_logger()

    results_folder = _results_dir()

    # If loading existing results, read from file and return
    if load_optimum:
        result_path = results_folder / scenario_name / "opt_result.json"
        if result_path.exists():
            with result_path.open() as f:
                result = load(f)
                input_optimal = {
                    name: array(value) for name, value in result["inputs"].items()
                }
                output_optimal = {
                    name: array(value) for name, value in result["outputs"].items()
                }

                # Optionally plot loaded results
                if plot_optimum:
                    start_year = 2025.0
                    end_year = 2075.0
                    _, _, _, energy_mix, fleet = single_scenario_setup(
                        name=scenario_name,
                        background_scenario_name=global_scenario_name,
                        start_year=start_year,
                        end_year=end_year,
                        technology_index=technology_index,
                        plot_scenario_data=False,
                        integrate_constraints=False,
                        demand_aversion=low_demand_formulation,
                        drop_in_only=drop_in_only,
                        fossil_kerosene_only=fossil_kerosene_only,
                        preferential_energy=preferential_energy,
                    )
                    plot_single_scenario_result(
                        scenario_name=scenario_name,
                        output_optimal={**input_optimal, **output_optimal},
                        energy_mix=energy_mix,
                        fleet=fleet,
                        low_demand=low_demand_formulation,
                        save_figs=save_figs,
                        directory_path=str(results_folder / scenario_name),
                    )

                return output_optimal
        else:
            LOGGER.warning(
                "No saved results found at %s - . Running optimization.",
                result_path,
            )

    # Standard optimization execution (existing code)
    start_year = 2025.0
    end_year = 2075.0
    aeromax_scenario, design_space, constraints, energy_mix, fleet = (
        single_scenario_setup(
            name=scenario_name,
            background_scenario_name=global_scenario_name,
            start_year=start_year,
            end_year=end_year,
            technology_index=technology_index,
            plot_scenario_data=False,
            integrate_constraints=False,
            demand_aversion=low_demand_formulation,
            drop_in_only=drop_in_only,
            fossil_kerosene_only=fossil_kerosene_only,
            preferential_energy=preferential_energy,
        )
    )

    gemseo_scenario = create_scenario(
        disciplines=aeromax_scenario.discipline,
        formulation_name="DisciplinaryOpt",
        objective_name=(
            "cumulative.discounted_relative_price_change"
            if low_demand_formulation
            else "cumulative.CO2"
        ),
        design_space=design_space,
    )
    time_period = end_year - start_year
    allocated_co2_budget = co2_budget_2p0deg_66percent * carbon_budget_percent * 1.0e-2
    if low_demand_formulation:
        gemseo_scenario.formulation.optimization_problem.objective *= 1.0 / time_period
        gemseo_scenario.add_constraint(
            "cumulative.CO2", "ineq", value=allocated_co2_budget
        )
        gemseo_scenario.formulation.optimization_problem.constraints[-1] *= (
            1.0 / allocated_co2_budget
        )
    else:
        gemseo_scenario.formulation.optimization_problem.objective *= (
            1.0 / allocated_co2_budget
        )

    for name, (value, positive) in constraints.items():
        gemseo_scenario.add_constraint(name, "ineq", value=value, positive=positive)
        if "cumulative" in name or "Electric" in name:
            gemseo_scenario.formulation.optimization_problem.constraints[-1] *= 0.1

    gemseo_scenario.execute(
        max_iter=2000,
        algo_name="NLOPT_SLSQP",
        ftol_rel=1e-15,
        ftol_abs=1e-15,
        ineq_tolerance=1e-4,
        xtol_rel=1e-12,
        xtol_abs=1e-12,
    )

    input_optimal = gemseo_scenario.optimization_result.x_opt_as_dict
    output_optimal = aeromax_scenario.discipline.execute(input_optimal)

    if save_history_view:
        gemseo_scenario.post_process(
            post_name="OptHistoryView",
            show=False,
            save=True,
            directory_path=str(results_folder / scenario_name),
            fig_size=(11.0, 22.0),
        )

    if save_optimum:
        (results_folder / scenario_name).mkdir(parents=True, exist_ok=True)
        result = {
            "inputs": {name: value.tolist() for name, value in input_optimal.items()}
        }
        result.update({
            "outputs": {name: value.tolist() for name, value in output_optimal.items()}
        })
        with (results_folder / scenario_name / "opt_result.json").open("w") as r_file:
            dump(result, r_file)

    if plot_optimum:
        plot_single_scenario_result(
            scenario_name=scenario_name,
            output_optimal={**input_optimal, **output_optimal},
            energy_mix=energy_mix,
            fleet=fleet,
            low_demand=low_demand_formulation,
            save_figs=save_figs,
            directory_path=str(results_folder / scenario_name),
        )

    return output_optimal


def single_policy_robust_scenario_optimization(
    scenario_name,
    global_scenario_names,
    carbon_budget_percent=3.0,
    technology_index=0,
    drop_in_only=False,
    fossil_kerosene_only=False,
    low_demand_formulation=False,
    preferential_energy=False,
    load_optimum=False,
    plot_optimum=True,
    save_optimum=False,
    save_history_view=False,
    save_figs=False,
):
    """Optimal decarbonization scenario robust to several background scenarios."""
    if fossil_kerosene_only:
        scenario_name += "-Fossil"
    elif drop_in_only:
        scenario_name += "-DropIn"

    if low_demand_formulation:
        scenario_name += "-LowDemand"
    if preferential_energy:
        scenario_name += "-Availability"

    if technology_index == 0:
        scenario_name += "-lowTech"
    elif technology_index == 1:
        scenario_name += "-midTech"
    elif technology_index == 2:
        scenario_name += "-upTech"

    configure_logger()

    results_folder = _results_dir()

    # If loading existing results, read from file and return
    if load_optimum:
        result_path = results_folder / scenario_name / "opt_result.json"
        if result_path.exists():
            with result_path.open() as f:
                result = load(f)
                input_optimal = {
                    name: array(value) for name, value in result["inputs"].items()
                }
                output_optimal = {
                    name: array(value) for name, value in result["outputs"].items()
                }

                # Optionally plot loaded results
                if plot_optimum:
                    start_year = 2025.0
                    end_year = 2075.0
                    _, _, _, energy_mix, fleet = multi_scenario_setup(
                        scenario_name,
                        background_scenario_names=global_scenario_names,
                        start_year=start_year,
                        end_year=end_year,
                        technology_index=technology_index,
                        plot_scenario_data=False,
                        integrate_constraints=False,
                        aggregate_constraints=False,
                        demand_aversion=low_demand_formulation,
                        drop_in_only=drop_in_only,
                        preferential_energy=preferential_energy,
                    )
                    plot_multi_scenario_result(
                        scenario_names=global_scenario_names,
                        mean_outputs=[],
                        output_optimal={**input_optimal, **output_optimal},
                        energy_mix=energy_mix,
                        fleet=fleet,
                        year_endplots=2075.0,
                        low_demand=low_demand_formulation,
                        save_figs=save_figs,
                        directory_path=str(results_folder / scenario_name),
                    )

                return output_optimal
        else:
            LOGGER.warning(
                "No saved results found at %s - . Running optimization.",
                result_path,
            )

    start_year = 2025.0
    end_year = 2075.0
    aeromax_scenario, design_space, constraints, energy_mix, fleet = (
        multi_scenario_setup(
            scenario_name,
            background_scenario_names=global_scenario_names,
            start_year=start_year,
            end_year=end_year,
            technology_index=technology_index,
            plot_scenario_data=False,
            integrate_constraints=False,
            aggregate_constraints=False,
            demand_aversion=low_demand_formulation,
            fossil_kerosene_only=fossil_kerosene_only,
            drop_in_only=drop_in_only,
            preferential_energy=preferential_energy,
        )
    )

    gemseo_scenario = create_scenario(
        disciplines=aeromax_scenario.discipline,
        formulation_name="DisciplinaryOpt",
        objective_name=(
            "mean.cumulative.discounted_relative_price_change"
            if low_demand_formulation
            else "mean.cumulative.CO2"
        ),
        design_space=design_space,
    )

    allocated_co2_budget = co2_budget_2p0deg_66percent * carbon_budget_percent * 1.0e-2

    if low_demand_formulation:
        for global_scenario in global_scenario_names:
            gemseo_scenario.add_constraint(
                f"{global_scenario}.cumulative.CO2", "ineq", value=allocated_co2_budget
            )
            gemseo_scenario.formulation.optimization_problem.constraints[-1] *= (
                1.0 / allocated_co2_budget
            )
    else:
        gemseo_scenario.formulation.optimization_problem.objective *= (
            1.0 / allocated_co2_budget
        )

    for name, (value, positive) in constraints.items():
        gemseo_scenario.add_constraint(name, "ineq", value=value, positive=positive)
        if "cumulative" in name or "Electric" in name:
            gemseo_scenario.formulation.optimization_problem.constraints[-1] *= 0.1

    gemseo_scenario.execute(
        max_iter=5000,
        algo_name="NLOPT_SLSQP",
        ftol_rel=1e-14,
        ftol_abs=1e-14,
        ineq_tolerance=1e-3,
        xtol_rel=1e-11,
        xtol_abs=1e-11,
    )

    input_optimal = gemseo_scenario.optimization_result.x_opt_as_dict
    output_optimal = aeromax_scenario.discipline.execute(input_optimal)

    if save_history_view:
        gemseo_scenario.post_process(
            post_name="OptHistoryView",
            show=False,
            save=True,
            directory_path=str(results_folder / scenario_name),
            fig_size=(11.0, 44.0),
        )

    if save_optimum:
        (results_folder / scenario_name).mkdir(parents=True, exist_ok=True)
        result = {
            "inputs": {name: value.tolist() for name, value in input_optimal.items()}
        }
        result.update({
            "outputs": {name: value.tolist() for name, value in output_optimal.items()}
        })
        with (results_folder / scenario_name / "opt_result.json").open("w") as r_file:
            dump(result, r_file)

    if plot_optimum:
        plot_multi_scenario_result(
            scenario_names=global_scenario_names,
            mean_outputs=aeromax_scenario.mean_outputs,
            output_optimal={**input_optimal, **output_optimal},
            energy_mix=energy_mix,
            fleet=fleet,
            year_endplots=2075.0,
            low_demand=low_demand_formulation,
            save_figs=save_figs,
            directory_path=str(results_folder / scenario_name),
        )
    return output_optimal
