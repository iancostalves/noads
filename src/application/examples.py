from json import load
from os import mkdir
from os import walk
from os.path import isdir
from json import dump

from gemseo import MDODiscipline
from gemseo import configure_logger
from gemseo import create_scenario
from numpy import argmax

from numpy import array as np_array
from numpy import sum as np_sum
from numpy import interp
from numpy import trapz
from numpy import vstack

from aviation_scenarios.scenario_data import co2_budget_2p0deg_66percent
from aviation_scenarios.scenario_setup import single_scenario_setup
from aviation_scenarios.visualization import plot_single_scenario_result
from aviation_scenarios.visualization import plot_tech_scenario_comparison


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
    aeromax_scenario, design_space, constraints, energy_mix, fleet = \
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

    # if gemseo_scenario is None:
    gemseo_scenario = create_scenario(
        disciplines=aeromax_scenario.discipline,
        formulation="DisciplinaryOpt",
        objective_name=(
            "cumulative.discounted_relative_price_change"
            if low_demand_formulation else "cumulative.CO2"
        ),
        design_space=design_space,
        grammar_type=MDODiscipline.GrammarType.SIMPLE,
    )
    time_period = end_year - start_year
    allocated_co2_budget = co2_budget_2p0deg_66percent * carbon_budget_percent * 1.0e-2
    if low_demand_formulation:
        gemseo_scenario.formulation.optimization_problem.objective *= \
            1.0 / time_period
        gemseo_scenario.add_constraint(
            "cumulative.CO2", "ineq", value=allocated_co2_budget
        )
        gemseo_scenario.formulation.optimization_problem.constraints[-1] *= \
            1.0 / allocated_co2_budget
    else:
        gemseo_scenario.formulation.optimization_problem.objective *= \
            1.0 / allocated_co2_budget

    for name, (value, positive) in constraints.items():
        gemseo_scenario.add_constraint(name, "ineq", value=value, positive=positive)
        if "cumulative" in name or "Electric" in name:
            gemseo_scenario.formulation.optimization_problem.constraints[-1] *= 0.1


    # else:
    #     gemseo_scenario.formulation._disciplines = \
    #         aeromax_scenario.discipline.default_inputs

    gemseo_scenario.execute({
        "max_iter": 700, "algo": "NLOPT_SLSQP",
        "algo_options": {
            "ftol_rel": 1e-14, "ftol_abs": 1e-14, "ineq_tolerance": 1e-6,
            "xtol_rel": 1e-11, "xtol_abs": 1e-11,
        },
    })
    gemseo_scenario.post_process(
        "OptHistoryView", save=True, show=False,
        directory_path=f"{formulation_name}/{scenario_name}",
    )

    input_optimal = gemseo_scenario.optimization_result.x_opt_as_dict
    result = {"inputs": {
        name: value.tolist() for name, value in input_optimal.items()
    }}
    output_optimal = aeromax_scenario.discipline.execute(input_optimal)
    result.update({"outputs": {
        name: value.tolist() for name, value in output_optimal.items()
    }})
    dump(result, open(f"{formulation_name}/{scenario_name}/opt_result.json", 'w'))

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
    directory = f"./single_policy"
    variables_units_conversion = {
        "rpk": (1e-12, "trillion pax-km / yr"),
        "ask": (1e-12, "trillion pax-km / yr"),
        "CO2": (1e-15, "Gt CO2 / yr"),
        "ELECTRICITY.consumption": (1e-12, "EJ / yr"),
        "BIOMASS.consumption": (1e-12, "EJ / yr"),
        "OIL.consumption": (1e-12, "EJ / yr"),
    }
    aircraft_agregation_names = [
        "Current", "JetA", "Battery", "lH2",
    ]
    observed_years = np_array([2030, 2050, 2070])

    for path, folders, files in walk(directory):
        for folder_name in folders:
            print(folder_name)
            with open(f"{directory}/{folder_name}/opt_result.json", 'r') as f:
                result = load(f)
                io_dict = {**result["inputs"], **result["outputs"]}
                simulation_years = io_dict["year"]
                for i, (var, (factor, unit)) in enumerate(variables_units_conversion.items()):
                    print("     ", var, "[", unit, "]")
                    observed_variable = interp(
                        observed_years, simulation_years, io_dict[var]
                    ) * factor
                    print("         decades:", observed_variable)
                    cumulative_variable = trapz(io_dict[var], simulation_years) * factor
                    print("         cumulative:", cumulative_variable)
                    print("         max:", max(io_dict[var]) * factor)
                    print("         y max:", simulation_years[argmax(io_dict[var])])
                for aircraft_name in aircraft_agregation_names:
                    if any([
                        ".ask" in var and aircraft_name in var for var in io_dict.keys()
                    ]):
                        print(
                            "     ", aircraft_name, "ASK [",
                            variables_units_conversion["rpk"][1], "]"
                        )
                        asks = vstack([
                            np_array(value) for var, value in io_dict.items()
                            if ".ask" in var and aircraft_name in var
                        ])
                        sum_ask = np_sum(asks, axis=0)
                        decadal_ask = interp(
                            observed_years, simulation_years, sum_ask
                        ) * variables_units_conversion["rpk"][0]
                        cumulative_ask = trapz(sum_ask, simulation_years) * \
                                         variables_units_conversion["rpk"][0]
                        print("         decades:", decadal_ask)
                        print("         cumulative:", cumulative_ask)