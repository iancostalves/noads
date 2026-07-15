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
"""Utilities for handling background scenario from the AR6 database."""

from matplotlib.pyplot import subplots
from numpy import argwhere
from numpy import array
from numpy import isnan
from numpy import ndarray
from numpy import ravel
from pandas import read_csv

from noads._data import data_file

scenario_to_model = {
    # "SSP1-34": "MESSAGE-GLOBIOM 1.0",
    "SSP1-19": "WITCH-GLOBIOM 3.1",
    # "SSP1-45": "IMAGE 3.0.1",
    # "SSP1-26": "IMAGE 3.0.1",
    # "SSP2-45": "MESSAGE-GLOBIOM 1.0",
    "SSP2-19": "MESSAGE-GLOBIOM 1.0",
    "SSP2-26": "MESSAGE-GLOBIOM 1.0",
    "SSP2-34": "MESSAGE-GLOBIOM 1.0",
    # "SSP3-45": "MESSAGE-GLOBIOM 1.0",
    # "SSP3-34": "MESSAGE-GLOBIOM 1.0",
    # "SSP4-45": "AIM/CGE 2.0",
    # "SSP4-34": "AIM/CGE 2.0",
    # "SSP4-26": "AIM/CGE 2.0",
    "SSP5-45": "REMIND-MAgPIE 1.5",
    # "SSP5-34": "REMIND-MAgPIE 1.5",
    # "SSP2-34": "MESSAGE-GLOBIOM 1.0",
    # "SSP2-19": "MESSAGE-GLOBIOM 1.0",
    # "SSP2-45": "MESSAGE-GLOBIOM 1.0",
    # "SSP3-34": "MESSAGE-GLOBIOM 1.0",
    # "SSP3-45": "MESSAGE-GLOBIOM 1.0",
    # "SSP4-26": "GCAM 4.2",
    # "SSP4-19": "WITCH-GLOBIOM 3.1",
    # "SSP4-34": "AIM/CGE 2.0",
    # "SSP4-26": "AIM/CGE 2.0",
    # "SSP4-45": "AIM/CGE 2.0",
    # "SSP4-45": "GCAM 4.2",
    # "SSP5-34": "REMIND-MAgPIE 1.5",
    # "SSP5-26": "REMIND-MAgPIE 1.5",
    # "SSP5-45": "REMIND-MAgPIE 1.5",
    # "SSP5-45": "REMIND-MAgPIE 1.5",
}

model_to_colors = {
    "IMAGE 3.2": "lightseagreen",
    "REMIND-MAgPIE 2.1-4.3": "chocolate",
    "REMIND-Transport 2.1": "crimson",
    "AIM/Hub-Global 2.0": "navy",
    "MESSAGEix-GLOBIOM_GEI 1.0": "gold",
    "MESSAGEix-GLOBIOM_1.1": "red",
    "MESSAGEix-GLOBIOM 1.0": "darkviolet",
}

co2_budget_2p5deg_66percent = 1890e15
co2_budget_2p2deg_66percent = 1326e15
co2_budget_2p0deg_66percent = 944e15
co2_budget_1p8deg_66percent = 571e15
co2_budget_1p5deg_66percent = 60e15


def lines_gen():
    """Linestyle generator."""
    yield "-"
    for i in range(1, 12):
        yield 0, (i, 1, 1, 1)


def get_scenario_color(scenario_name: str):
    """Get scenario color from SSP name."""
    name_split = scenario_name.split("SSP")
    ssp_idx = int(name_split[1][0])
    if ssp_idx == 1:
        return "darkgreen"
    if ssp_idx == 2:
        return "royalblue"
    if ssp_idx == 3:
        return "firebrick"
    if ssp_idx == 4:
        return "darkorange"
    return "darkviolet"


def get_ar6_input_data(start_year=2010, end_year=2080, plot_data=True):
    """Get relevant aviation input data from AR6 scenarios."""
    population_data = read_csv(
        data_file("noads.application", "ar6_scenarios_data", "population.csv"),
        index_col=1,
    )
    gdp_data = read_csv(
        data_file("noads.application", "ar6_scenarios_data", "gdp.csv"),
        index_col=1,
    )
    electricity_emissions_data = read_csv(
        data_file(
            "noads.application", "ar6_scenarios_data", "electricity_emissions.csv"
        ),
        index_col=1,
    )
    final_electricity_data = read_csv(
        data_file("noads.application", "ar6_scenarios_data", "final_electricity.csv"),
        index_col=1,
    )
    biomass_data = read_csv(
        data_file("noads.application", "ar6_scenarios_data", "biomass_total.csv"),
        index_col=1,
    )

    all_years = [
        int(year)
        for year in list(electricity_emissions_data.keys())[5:]
        if start_year <= int(year) <= end_year
    ]
    var_units_name_convert = {
        "population": ("million hab.", "Population", 1e6),
        "gdp_per_capita": ("thousand US$2010/(hab. year)", "GDP (PPP) per capita", 1e3),
        "gdp": ("billion US$2010/year", "Gross Domestic Product (PPP)", 1e9),
        "ELECTRICITY.CO2_index": ("g CO2 / MJ", "Electricity Emission Factor", 1),
        "ELECTRICITY.global_production": ("EJ / year", "Electricity Production", 1e12),
        "BIOMASS.global_production": ("EJ / year", "Biomass Production", 1e12),
    }

    ar6_data = {
        "population": {},
        "gdp_per_capita": {},
        "gdp": {},
        "ELECTRICITY.CO2_index": {},
        "ELECTRICITY.global_production": {},
        "BIOMASS.global_production": {},
    }
    years = []
    for scenario, model in scenario_to_model.items():
        for variable in ar6_data:
            ar6_data[variable][scenario] = []
        for year in all_years:
            array_or_pandas = population_data[str(year)][scenario][
                population_data["Model"][scenario] == model
            ]
            if isinstance(array_or_pandas, ndarray):
                pop = population_data[str(year)][scenario][
                    population_data["Model"][scenario] == model
                ]
                gdp = gdp_data[str(year)][scenario][
                    gdp_data["Model"][scenario] == model
                ]
                elec_co2 = electricity_emissions_data[str(year)][scenario][
                    electricity_emissions_data["Model"][scenario] == model
                ]
                elec = final_electricity_data[str(year)][scenario][
                    final_electricity_data["Model"][scenario] == model
                ]
                biomass = biomass_data[str(year)][scenario][
                    biomass_data["Model"][scenario] == model
                ]
            else:
                pop = population_data[str(year)][scenario][
                    population_data["Model"][scenario] == model
                ].to_numpy()
                gdp = gdp_data[str(year)][scenario][
                    gdp_data["Model"][scenario] == model
                ].to_numpy()
                elec_co2 = electricity_emissions_data[str(year)][scenario][
                    electricity_emissions_data["Model"][scenario] == model
                ].to_numpy()
                elec = final_electricity_data[str(year)][scenario][
                    final_electricity_data["Model"][scenario] == model
                ].to_numpy()
                biomass = biomass_data[str(year)][scenario][
                    biomass_data["Model"][scenario] == model
                ].to_numpy()

            if not any(
                isnan(data_array) or data_array.size == 0
                for data_array in [pop, gdp, elec_co2, elec, biomass]
            ):
                if year not in years:
                    years.append(year)
                ar6_data["population"][scenario].append(
                    float(pop) * var_units_name_convert["population"][-1]
                )
                ar6_data["gdp"][scenario].append(
                    float(gdp) * var_units_name_convert["gdp"][-1]
                )
                ar6_data["gdp_per_capita"][scenario].append(
                    float(gdp / pop) * var_units_name_convert["gdp_per_capita"][-1]
                )
                ar6_data["ELECTRICITY.CO2_index"][scenario].append(
                    max(0.0, float(elec_co2 / elec))  # no below zero electricity
                    # float(elec_co2 / elec)
                    * var_units_name_convert["ELECTRICITY.CO2_index"][-1]
                )
                ar6_data["ELECTRICITY.global_production"][scenario].append(
                    float(elec)
                    * var_units_name_convert["ELECTRICITY.global_production"][-1]
                )
                ar6_data["BIOMASS.global_production"][scenario].append(
                    float(biomass)
                    * var_units_name_convert["BIOMASS.global_production"][-1]
                )

    if plot_data:
        var_groups = {
            # "socioeconomic": ["population", "gdp", "gdp_per_capita"],
            # "energy": [
            #     "ELECTRICITY.CO2_index",
            #     "ELECTRICITY.global_production",
            #     "BIOMASS.global_production",
            # ],
            "all": [
                "population",
                "gdp_per_capita",
                "ELECTRICITY.CO2_index",
                "ELECTRICITY.global_production",
                "BIOMASS.global_production",
            ],
        }
        for variable_group in var_groups.values():
            fig, axes = subplots(5, 1, layout="constrained")
            fig.set_size_inches(5, 16)
            lines = lines_gen()
            last_idx = 0
            for i, name in enumerate(variable_group):
                for scenario in scenario_to_model:
                    name_split = scenario.split("SSP")
                    ssp_idx = int(name_split[1][0])
                    if ssp_idx != last_idx:
                        lines = lines_gen()
                    line = next(lines)
                    last_idx = ssp_idx
                    axes[i].plot(
                        years,
                        [
                            value / var_units_name_convert[name][-1]
                            for value in ar6_data[name][scenario]
                        ],
                        label=scenario,
                        color=get_scenario_color(scenario),
                        linestyle=line,
                        linewidth=2,
                    )
                    axes[i].set_title(var_units_name_convert[name][1])
                    axes[i].set_ylabel(var_units_name_convert[name][0])
                    axes[i].set_xlabel("Year")
            axes[0].legend(loc="lower left")
            axes[0].set_ylim(bottom=0)
            axes[1].set_ylim(bottom=0)
            axes[2].set_ylim(bottom=0)
            axes[3].set_ylim(bottom=0)
            axes[4].set_ylim(bottom=0)
            fig.savefig("ar6_data.pdf")
            # fig.show()
            # close(fig)

    del ar6_data["gdp"]
    return ar6_data, years


def get_ar6_output_data(start_year=2010, end_year=2080, plot_data=True):
    """Get relevant aviation output data from AR6 scenarios."""
    passenger_energy_data = read_csv(
        data_file(
            "noads.application", "ar6_scenarios_data", "aviation_passenger_energy.csv"
        ),
        index_col=1,
    )
    passenger_co2_data = read_csv(
        data_file(
            "noads.application", "ar6_scenarios_data", "aviation_passenger_co2.csv"
        ),
        index_col=1,
    )
    total_energy_data = read_csv(
        data_file(
            "noads.application", "ar6_scenarios_data", "aviation_total_energy.csv"
        ),
        index_col=1,
    )
    total_co2_data = read_csv(
        data_file("noads.application", "ar6_scenarios_data", "aviation_total_co2.csv"),
        index_col=1,
    )

    all_years = [
        int(year)
        for year in list(passenger_energy_data.keys())[5:]
        if start_year <= int(year) <= end_year
    ]
    years_array = array(all_years)
    # var_units_name_convert = {
    #     "passenger_energy": ("EJ/yr", "Passenger Aviation Final Energy", 1.0),
    #     "passenger_co2": ("Mt CO2/yr", "Passenger Aviation CO2 Emissions", 1.0),
    #     "total_energy": ("EJ/yr", "Total Aviation Final Energy", 1.0),
    #     "total_co2": ("Mt CO2/yr", "Total Aviation CO2 Emissions", 1.0),
    # }

    ar6_data = {
        "passenger_energy": {},
        "passenger_co2": {},
        "total_energy": {},
        "total_co2": {},
    }
    var_to_data = {
        "passenger_energy": passenger_energy_data,
        "passenger_co2": passenger_co2_data,
        "total_energy": total_energy_data,
        "total_co2": total_co2_data,
    }

    # years = []
    for variable, dataframe in var_to_data.items():
        scenarios = dataframe["Model"].keys()
        models = dataframe["Model"].tolist()
        for model in models:
            if "AR6 Scenario Explorer" not in model:
                ar6_data[variable][model] = {}
                for scenario in scenarios:
                    ar6_data[variable][model][scenario] = []
                    for year in all_years:
                        data_at_year = dataframe[str(year)]
                        value_year = data_at_year[scenario]
                        ar6_data[variable][model][scenario].append(value_year)

    if plot_data:
        fig, axes = subplots(2, 2, layout="constrained")
        fig.set_size_inches(9, 9)

        # min_value_models = {}
        # max_value_models = {}
        for variable, model_scenarios in ar6_data.items():
            line = 0 if "energy" in variable else 1
            col = 0 if "passenger" in variable else 1
            for model_name, scenarios_data in model_scenarios.items():
                for scenario_name, scenario_data in scenarios_data.items():
                    data_array = array(scenario_data)
                    valid_indices = ravel(argwhere(~isnan(scenario_data)))
                    axes[line, col].plot(
                        years_array[valid_indices],
                        data_array[valid_indices],
                        color=model_to_colors[model_name],
                        label=model_name
                        if scenario_name == next(iter(scenarios_data.keys()))
                        else "_",
                        alpha=0.2,
                    )

                # model_min = []
                # model_max = []
                # for i, year in enumerate(all_years):
                #     values_at_year_scenarios = [
                #         yearly_values[i]
                #         for scenario, yearly_values in scenarios_data.items()
                #     ]
                #     model_min.append(min(values_at_year_scenarios))
                #     model_max.append(max(values_at_year_scenarios))
                #
                # min_value_models[model_name] = array(model_min)
                # max_value_models[model_name] = array(model_max)
                #
                # valid_indices = ravel(argwhere(~isnan(model_min)))
                # # print(model_min)
                # # print(valid_indices)
                # axes[line, col].fill_between(
                #     years_array[valid_indices],
                #     min_value_models[model_name][valid_indices],
                #     max_value_models[model_name][valid_indices],
                #     alpha=0.2,
                #     label=model_name,
                # )
            axes[0, 0].set_ylabel("EJ / yr")
            axes[1, 0].set_ylabel("Mt CO2 / yr")
            axes[1, 0].set_xlabel("Year")
            axes[1, 1].set_xlabel("Year")

            axes[0, 0].set_title("Passenger Final Energy")
            axes[0, 1].set_title("Total Final Energy")

            axes[1, 0].set_title("Passenger Emissions")
            axes[1, 1].set_title("Total Emissions")

            axes[0, 0].legend(loc="upper left")
            axes[0, 1].legend(loc="upper left")
            axes[1, 0].legend(loc="upper left")
            axes[1, 1].legend(loc="upper left")

        fig.savefig("ar6_aviation.pdf")
    return ar6_data, years_array
