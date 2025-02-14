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

from json import dump
from json import load
from os import mkdir
from os import walk
from os.path import isdir

from aviation_scenarios.scenario_data import co2_budget_2p0deg_66percent
from aviation_scenarios.scenario_setup import single_scenario_setup
from aviation_scenarios.visualization import plot_single_scenario_result
from gemseo import MDODiscipline
from gemseo import configure_logger
from gemseo import create_scenario
from numpy import array as np_array
from numpy import interp
from numpy import sum as np_sum
from numpy import trapz
from numpy import vstack


def single_policy_scenario_optimization(
    global_scenario_name,
    carbon_budget_percent=3.0,
    technology_index=0,
    drop_in_only=False,
    low_demand_formulation=False,
    preferential_energy=False,
    plot_optimum=True,
):
    scenario_name = global_scenario_name
    if drop_in_only:
        scenario_name += "-DropIn"

    if low_demand_formulation:
        scenario_name += "-LD"
    if preferential_energy:
        scenario_name += "-E"

    if technology_index == 0:
        scenario_name += "-low"
    elif technology_index == 1:
        scenario_name += "-mid"
    elif technology_index == 2:
        scenario_name += "-up"

    configure_logger()

    formulation_name = "single_policy"

    if not isdir(formulation_name):
        mkdir(formulation_name)
    if not isdir(f"{formulation_name}/{scenario_name}"):
        mkdir(f"{formulation_name}/{scenario_name}")
    start_year = 2025.0
    end_year = 2075.0
    aeromax_scenario, design_space, constraints, energy_mix, fleet = (
        single_scenario_setup(
            global_scenario_name,
            start_year=start_year,
            end_year=end_year,
            technology_index=technology_index,
            plot_scenario_data=True,
            integrate_constraints=False,
            demand_aversion=low_demand_formulation,
            drop_in_only=drop_in_only,
            preferential_energy=preferential_energy,
        )
    )

    # if gemseo_scenario is None:
    gemseo_scenario = create_scenario(
        disciplines=aeromax_scenario.discipline,
        formulation="DisciplinaryOpt",
        objective_name=(
            "cumulative.discounted_relative_price_change"
            if low_demand_formulation
            else "cumulative.CO2"
        ),
        design_space=design_space,
        grammar_type=MDODiscipline.GrammarType.SIMPLE,
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

    # else:
    #     gemseo_scenario.formulation._disciplines = \
    #         aeromax_scenario.discipline.default_inputs

    gemseo_scenario.execute({
        "max_iter": 700,
        "algo": "NLOPT_SLSQP",
        "algo_options": {
            "ftol_rel": 1e-14,
            "ftol_abs": 1e-14,
            "ineq_tolerance": 1e-6,
            "xtol_rel": 1e-11,
            "xtol_abs": 1e-11,
        },
    })
    gemseo_scenario.post_process(
        "OptHistoryView",
        save=True,
        show=False,
        directory_path=f"{formulation_name}/{scenario_name}",
    )

    input_optimal = gemseo_scenario.optimization_result.x_opt_as_dict
    result = {"inputs": {name: value.tolist() for name, value in input_optimal.items()}}
    output_optimal = aeromax_scenario.discipline.execute(input_optimal)
    result.update({
        "outputs": {name: value.tolist() for name, value in output_optimal.items()}
    })
    dump(result, open(f"{formulation_name}/{scenario_name}/opt_result.json", "w"))

    if plot_optimum:
        plot_single_scenario_result(
            scenario_name=scenario_name,
            folder_name=f"./{formulation_name}/{scenario_name}",
            output_optimal={**input_optimal, **output_optimal},
            energy_mix=energy_mix,
            fleet=fleet,
            low_demand=low_demand_formulation,
        )
    return output_optimal  # , gemseo_scenario


def single_scenario_synthesis():
    directory = "./single_policy"
    variables_units_conversion = {
        "rpk": (1e-12, "trillion pax-km / yr"),
        "ask": (1e-12, "trillion pax-km / yr"),
        "CO2": (1e-15, "Gt CO2 / yr"),
        "ELECTRICITY.consumption": (1e-12, "EJ / yr"),
        "BIOMASS.consumption": (1e-12, "EJ / yr"),
        "OIL.consumption": (1e-12, "EJ / yr"),
    }
    aircraft_agregation_names = [
        "Current",
        "JetA",
        "Battery",
        "lH2",
    ]
    observed_years = np_array([2030, 2050, 2070])

    for _path, folders, _files in walk(directory):
        for folder_name in folders:
            with open(f"{directory}/{folder_name}/opt_result.json") as f:
                result = load(f)
                io_dict = {**result["inputs"], **result["outputs"]}
                simulation_years = io_dict["year"]
                for _i, (var, (factor, _unit)) in enumerate(
                    variables_units_conversion.items()
                ):
                    interp(observed_years, simulation_years, io_dict[var]) * factor
                    trapz(io_dict[var], simulation_years) * factor
                for aircraft_name in aircraft_agregation_names:
                    if any(".ask" in var and aircraft_name in var for var in io_dict):
                        asks = vstack([
                            np_array(value)
                            for var, value in io_dict.items()
                            if ".ask" in var and aircraft_name in var
                        ])
                        sum_ask = np_sum(asks, axis=0)
                        interp(
                            observed_years, simulation_years, sum_ask
                        ) * variables_units_conversion["rpk"][0]
                        trapz(sum_ask, simulation_years) * variables_units_conversion[
                            "rpk"
                        ][0]


def single_policy_robust_scenario_optimization(
    scenario_name,
    global_scenario_names,
    carbon_budget_percent=3.0,
    technology_index=0,
    drop_in_only=False,
    low_demand_formulation=False,
    preferential_energy=False,
    plot_optimum=True,
):
    if drop_in_only:
        scenario_name += "-DropIn"

    if low_demand_formulation:
        scenario_name += "-LD"
    if preferential_energy:
        scenario_name += "-E"

    if technology_index == 0:
        scenario_name += "-low"
    elif technology_index == 1:
        scenario_name += "-mid"
    elif technology_index == 2:
        scenario_name += "-up"

    configure_logger()

    formulation_name = "robust_single_policy"

    if not isdir(formulation_name):
        mkdir(formulation_name)
    if not isdir(f"{formulation_name}/{scenario_name}"):
        mkdir(f"{formulation_name}/{scenario_name}")
    start_year = 2025.0
    end_year = 2075.0
    aeromax_scenario, design_space, constraints, energy_mix, fleet = (
        multi_scenario_setup(
            global_scenario_names,
            start_year=start_year,
            end_year=end_year,
            technology_index=technology_index,
            plot_scenario_data=True,
            integrate_constraints=False,
            aggregate_constraints=False,
            demand_aversion=low_demand_formulation,
            drop_in_only=drop_in_only,
            preferential_energy=preferential_energy,
        )
    )

    gemseo_scenario = create_scenario(
        disciplines=aeromax_scenario.discipline,
        formulation="DisciplinaryOpt",
        objective_name=(
            "mean.cumulative.discounted_relative_price_change"
            if low_demand_formulation
            else "mean.cumulative.CO2"
        ),
        design_space=design_space,
        grammar_type=MDODiscipline.GrammarType.SIMPLE,
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

    gemseo_scenario.execute({
        "max_iter": 1500,
        "algo": "NLOPT_SLSQP",
        "algo_options": {
            "ftol_rel": 1e-14,
            "ftol_abs": 1e-14,
            "ineq_tolerance": 1e-6,
            "xtol_rel": 1e-11,
            "xtol_abs": 1e-11,
        },
    })
    gemseo_scenario.post_process(
        "OptHistoryView",
        save=True,
        show=False,
        directory_path=f"{formulation_name}/{scenario_name}",
    )

    output_optimal = aeromax_scenario.discipline.execute(
        gemseo_scenario.optimization_result.x_opt_as_dict
    )
    output_optimal.update(aeromax_scenario.discipline.default_inputs)
    if plot_optimum:
        plot_multi_scenario_result(
            scenario_names=global_scenario_names,
            mean_outputs=aeromax_scenario.mean_outputs,
            folder_name=f"./{formulation_name}/{scenario_name}",
            figure_name="robust_scenario",
            output_optimal=output_optimal,
            energy_mix=energy_mix,
            fleet=fleet,
            year_endplots=2075.0,
            low_demand=low_demand_formulation,
        )
    return output_optimal


def multi_policy_scenario_optimization(
    global_scenario_name,
    n_sub_optim=10,
    # min_carbon_budget_percent=2.5,
    technology_index=0,
    drop_in_only=False,
    preferential_energy=False,
    plot_optimum=True,
):
    scenario_name = global_scenario_name
    if technology_index == 0:
        scenario_name += "-low"
    elif technology_index == 1:
        scenario_name += "-mid"
    elif technology_index == 2:
        scenario_name += "-up"

    if drop_in_only:
        scenario_name += "-DropIn"
    if preferential_energy:
        scenario_name += "-E"

    configure_logger()

    formulation_name = "multi_policy"

    if not isdir(formulation_name):
        mkdir(formulation_name)
    if not isdir(f"{formulation_name}/{scenario_name}"):
        mkdir(f"{formulation_name}/{scenario_name}")
    start_year = 2025.0
    end_year = 2075.0
    aeromax_scenario, design_space, constraints, energy_mix, fleet = (
        single_scenario_setup(
            global_scenario_name,
            start_year=start_year,
            end_year=end_year,
            time_step=2.0,
            interp_step=5.0,
            technology_index=technology_index,
            plot_scenario_data=True,
            integrate_constraints=False,
            demand_aversion=False,
            drop_in_only=drop_in_only,
            preferential_energy=preferential_energy,
        )
    )

    min_co2_trend_traffic_scenario = create_scenario(
        disciplines=aeromax_scenario.discipline,
        formulation="DisciplinaryOpt",
        objective_name="cumulative.CO2",
        design_space=design_space,
        grammar_type=MDODiscipline.GrammarType.SIMPLE,
    )
    co2_budget_ref = 3.0e-2 * co2_budget_2p0deg_66percent
    min_co2_trend_traffic_scenario.formulation.optimization_problem.objective *= (
        1.0 / co2_budget_ref
    )
    for name, (value, positive) in constraints.items():
        min_co2_trend_traffic_scenario.add_constraint(
            name, "ineq", value=value, positive=positive
        )
        if "cumulative" in name or "Electric" in name:
            min_co2_trend_traffic_scenario.formulation.optimization_problem.constraints[
                -1
            ] *= 0.1

    if not isdir(f"{formulation_name}/{scenario_name}/0"):
        mkdir(f"{formulation_name}/{scenario_name}/0")

    min_co2_trend_traffic_scenario.execute({
        "max_iter": 700,
        "algo": "NLOPT_SLSQP",
        "algo_options": {
            "ftol_rel": 1e-14,
            "ftol_abs": 1e-14,
            "ineq_tolerance": 1e-6,
            "xtol_rel": 1e-11,
            "xtol_abs": 1e-11,
        },
    })
    min_co2_trend_traffic_scenario.post_process(
        "OptHistoryView",
        save=True,
        show=False,
        directory_path=f"{formulation_name}/{scenario_name}/0",
    )

    input_optimal = min_co2_trend_traffic_scenario.optimization_result.x_opt_as_dict
    result = {"inputs": {name: value.tolist() for name, value in input_optimal.items()}}
    output_optimal = aeromax_scenario.discipline.execute(input_optimal)
    input_optimal.update({
        key: value
        for key, value in aeromax_scenario.discipline.default_inputs.items()
        if key not in input_optimal
    })
    max_budget = output_optimal["cumulative.CO2"]
    result.update({
        "outputs": {name: value.tolist() for name, value in output_optimal.items()}
    })
    dump(result, open(f"{formulation_name}/{scenario_name}/0/opt_result.json", "w"))

    inout_optimal = {**input_optimal, **output_optimal}
    pareto_outputs = {f"0.{name}": value for name, value in inout_optimal.items()}

    if plot_optimum:
        plot_single_scenario_result(
            scenario_name=scenario_name,
            folder_name=f"./{formulation_name}/{scenario_name}/0",
            output_optimal=inout_optimal,
            energy_mix=energy_mix,
            fleet=fleet,
            low_demand=True,
        )

    time_period = end_year - start_year

    aeromax_scenario, design_space, constraints, energy_mix, fleet = (
        single_scenario_setup(
            global_scenario_name,
            start_year=start_year,
            end_year=end_year,
            time_step=2.0,
            interp_step=5.0,
            technology_index=technology_index,
            plot_scenario_data=True,
            integrate_constraints=False,
            demand_aversion=True,
            drop_in_only=drop_in_only,
            preferential_energy=preferential_energy,
        )
    )
    min_co2_scenario = create_scenario(
        disciplines=aeromax_scenario.discipline,
        formulation="DisciplinaryOpt",
        objective_name="cumulative.CO2",
        design_space=design_space,
        grammar_type=MDODiscipline.GrammarType.SIMPLE,
    )
    co2_budget_ref = 3.0e-2 * co2_budget_2p0deg_66percent
    min_co2_scenario.formulation.optimization_problem.objective *= 1.0 / co2_budget_ref
    for name, (value, positive) in constraints.items():
        min_co2_scenario.add_constraint(name, "ineq", value=value, positive=positive)
        if "cumulative" in name or "Electric" in name:
            min_co2_scenario.formulation.optimization_problem.constraints[-1] *= 0.1

    if not isdir(f"{formulation_name}/{scenario_name}/{n_sub_optim}"):
        mkdir(f"{formulation_name}/{scenario_name}/{n_sub_optim}")

    min_co2_scenario.execute({
        "max_iter": 700,
        "algo": "NLOPT_SLSQP",
        "algo_options": {
            "ftol_rel": 1e-14,
            "ftol_abs": 1e-14,
            "ineq_tolerance": 1e-6,
            "xtol_rel": 1e-11,
            "xtol_abs": 1e-11,
        },
    })
    min_co2_scenario.post_process(
        "OptHistoryView",
        save=True,
        show=False,
        directory_path=f"{formulation_name}/{scenario_name}/{n_sub_optim}",
    )

    input_optimal = min_co2_scenario.optimization_result.x_opt_as_dict
    result = {"inputs": {name: value.tolist() for name, value in input_optimal.items()}}
    output_optimal = aeromax_scenario.discipline.execute(input_optimal)
    input_optimal.update({
        key: value
        for key, value in aeromax_scenario.discipline.default_inputs.items()
        if key not in input_optimal
    })
    min_budget = output_optimal["cumulative.CO2"]
    result.update({
        "outputs": {name: value.tolist() for name, value in output_optimal.items()}
    })
    dump(
        result,
        open(f"{formulation_name}/{scenario_name}/{n_sub_optim}/opt_result.json", "w"),
    )

    inout_optimal = {**input_optimal, **output_optimal}
    pareto_outputs.update({
        f"{n_sub_optim}.{name}": value for name, value in inout_optimal.items()
    })

    if plot_optimum:
        plot_single_scenario_result(
            scenario_name=scenario_name,
            folder_name=f"./{formulation_name}/{scenario_name}/{n_sub_optim}",
            output_optimal=inout_optimal,
            energy_mix=energy_mix,
            fleet=fleet,
            low_demand=True,
        )
    carbon_budget_constraints = linspace(
        max_budget,
        min_budget,
        n_sub_optim + 1,
    )
    for i in range(1, n_sub_optim):
        if not isdir(f"{formulation_name}/{scenario_name}/{i}"):
            mkdir(f"{formulation_name}/{scenario_name}/{i}")

        # design_space.set_current_value(input_optimal)
        sub_scenario = create_scenario(
            disciplines=aeromax_scenario.discipline,
            formulation="DisciplinaryOpt",
            objective_name="cumulative.discounted_relative_price_change",
            design_space=design_space,
            grammar_type=MDODiscipline.GrammarType.SIMPLE,
        )

        sub_scenario.formulation.optimization_problem.objective *= 1.0 / time_period
        for name, (value, positive) in constraints.items():
            sub_scenario.add_constraint(name, "ineq", value=value, positive=positive)
            if "cumulative" in name or "Electric" in name:
                sub_scenario.formulation.optimization_problem.constraints[-1] *= 0.1

        allocated_co2_budget = carbon_budget_constraints[i]
        sub_scenario.add_constraint(
            "cumulative.CO2", "ineq", value=allocated_co2_budget
        )
        sub_scenario.formulation.optimization_problem.constraints[-1] *= (
            1.0 / allocated_co2_budget
        )

        sub_scenario.execute({
            "max_iter": 700,
            "algo": "NLOPT_SLSQP",
            "algo_options": {
                "ftol_rel": 1e-14,
                "ftol_abs": 1e-14,
                "ineq_tolerance": 1e-6,
                "xtol_rel": 1e-11,
                "xtol_abs": 1e-11,
            },
        })
        sub_scenario.post_process(
            "OptHistoryView",
            save=True,
            show=False,
            directory_path=f"{formulation_name}/{scenario_name}/{i}",
        )

        input_optimal = sub_scenario.optimization_result.x_opt_as_dict
        result = {
            "inputs": {name: value.tolist() for name, value in input_optimal.items()}
        }
        output_optimal = aeromax_scenario.discipline.execute(input_optimal)
        input_optimal.update({
            key: value
            for key, value in aeromax_scenario.discipline.default_inputs.items()
            if key not in input_optimal
        })
        result.update({
            "outputs": {name: value.tolist() for name, value in output_optimal.items()}
        })
        dump(
            result, open(f"{formulation_name}/{scenario_name}/{i}/opt_result.json", "w")
        )

        inout_optimal = {**input_optimal, **output_optimal}
        pareto_outputs.update({
            f"{i}.{name}": value for name, value in inout_optimal.items()
        })

        if plot_optimum:
            plot_single_scenario_result(
                scenario_name=scenario_name,
                folder_name=f"./{formulation_name}/{scenario_name}/{i}",
                output_optimal=inout_optimal,
                energy_mix=energy_mix,
                fleet=fleet,
                low_demand=True,
            )
    if plot_optimum:
        plot_multi_scenario_result(
            scenario_names=[f"{i}" for i in range(n_sub_optim + 1)],
            mean_outputs=[],
            folder_name=f"./{formulation_name}/{scenario_name}",
            figure_name="pareto_front_comparison",
            output_optimal=pareto_outputs,
            energy_mix=energy_mix,
            fleet=fleet,
            year_endplots=end_year,
            low_demand=True,
        )
    return pareto_outputs
