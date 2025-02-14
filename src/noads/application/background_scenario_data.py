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

from matplotlib.pyplot import subplots
from numpy import isnan
from numpy import ndarray
from pandas import read_csv

scenario_to_model = {
    # "SSP1-34": "MESSAGE-GLOBIOM 1.0",
    # "SSP1-19": "WITCH-GLOBIOM 3.1",
    "SSP1-45": "IMAGE 3.0.1",
    # "SSP1-26": "IMAGE 3.0.1",
    "SSP2-45": "MESSAGE-GLOBIOM 1.0",
    "SSP2-34": "MESSAGE-GLOBIOM 1.0",
    "SSP2-26": "MESSAGE-GLOBIOM 1.0",
    "SSP2-19": "MESSAGE-GLOBIOM 1.0",
    "SSP3-45": "MESSAGE-GLOBIOM 1.0",
    # "SSP3-34": "MESSAGE-GLOBIOM 1.0",
    "SSP4-45": "AIM/CGE 2.0",
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

co2_budget_2p0deg_66percent = 944e15
co2_budget_1p8deg_66percent = 571e15
co2_budget_1p5deg_66percent = 60e15


def lines_gen():
    """Linestyle generator."""
    yield "-"
    for i in range(1, 12):
        yield 0, (i, 1, 1, 1)


def get_scenario_color(scenario_name: str):
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


def get_ar6_data(start_year=2010, end_year=2080, plot_data=True):
    population_data = read_csv(
        "C:/Users/i.costa-alves/Documents/Kodigus/opt-avi-decarb-scenarios/noads/src/application/ar6_scenarios_data/population.csv",
        index_col=1,
    )
    gdp_data = read_csv(
        "C:/Users/i.costa-alves/Documents/Kodigus/opt-avi-decarb-scenarios/noads/src/application/ar6_scenarios_data/gdp.csv",
        index_col=1,
    )
    electricity_emissions_data = read_csv(
        "C:/Users/i.costa-alves/Documents/Kodigus/opt-avi-decarb-scenarios/noads/src/application/ar6_scenarios_data/electricity_emissions.csv",
        index_col=1,
    )
    final_electricity_data = read_csv(
        "C:/Users/i.costa-alves/Documents/Kodigus/opt-avi-decarb-scenarios/noads/src/application/ar6_scenarios_data/final_electricity.csv",
        index_col=1,
    )
    biomass_data = read_csv(
        "C:/Users/i.costa-alves/Documents/Kodigus/opt-avi-decarb-scenarios/noads/src/application/ar6_scenarios_data/biomass_total.csv",
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
            "socioeconomic": ["population", "gdp", "gdp_per_capita"],
            "energy": [
                "ELECTRICITY.CO2_index",
                "ELECTRICITY.global_production",
                "BIOMASS.global_production",
            ],
        }
        for variable_group in var_groups.values():
            fig, axes = subplots(1, 3, layout="constrained")
            fig.set_size_inches(12, 5)
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
            axes[0].legend(loc="upper right")
            fig.show()
            # close(fig)

    del ar6_data["gdp"]
    return ar6_data, years
