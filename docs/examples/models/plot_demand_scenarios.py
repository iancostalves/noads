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

"""Air traffic demand estimation from AR6 scenarios."""

from jax import vmap
from matplotlib.pyplot import show
from matplotlib.pyplot import subplots
from numpy import array as np_array
from numpy import linspace
from numpy import ones

from noads.application.background_scenario_data import get_ar6_data
from noads.application.background_scenario_data import get_scenario_color
from noads.application.background_scenario_data import lines_gen
from noads.core.models.interpolation import interpolate_data
from noads.core.models.traffic import AirTraffic

# %%
# # AR6 scenario data
# Let's start by getting the AR6 scenario data and taking a look at it.
scenario_data, years_data = get_ar6_data(start_year=2000, end_year=2100, plot_data=True)

# %%
# Not all data is required for air traffic estimation, only Population and GDP per
# capita. Now we interpolate, for each scenario, to get values yearly.
years = linspace(2000, 2100, 101)
population = {
    scenario: interpolate_data(
        years, np_array(years_data), np_array(scenario_data["population"][scenario])
    )
    for scenario in scenario_data["population"]
}
gdp_per_capita = {
    scenario: interpolate_data(
        years, np_array(years_data), np_array(scenario_data["gdp_per_capita"][scenario])
    )
    for scenario in scenario_data["gdp_per_capita"]
}

# %%
# Also, we consider COVID effects by delaying trend traffic of a fixed amount of years,
# here we must also include the assumption of end of COVID related traffic constraints.
year_covid_end = 2024
gdp_per_capita_end_covid = {
    scenario: interpolate_data(
        year_covid_end,
        np_array(years_data),
        np_array(scenario_data["gdp_per_capita"][scenario]),
    )
    for scenario in scenario_data["gdp_per_capita"]
}

# %%
# # Model vectorization
# When a model must be evaluated yearly, rather than once, we must vectorize its
# function to account for vectorized inputs. When running a TemporalScenario this is
# done automatically, but here we must to it manually.
model = AirTraffic()
model.discipline.jax_out_func = vmap(model.discipline.jax_out_func)

# %%
# # Results
# Now let's visualize the results for each scenario
outputs = {}
fig, axes = subplots(2, 2, figsize=(8, 6), layout="constrained")
vones = ones(years.shape)

lines = lines_gen()
last_ssp_index = 0
for scenario in population:
    model_input = {
        "year": years,
        "gdp_per_capita": gdp_per_capita[scenario],
        "population": population[scenario],
        "gdp_per_capita_covid_end": gdp_per_capita_end_covid[scenario] * vones,
        "load_factor_end_year": 95.0 * vones,
        "end_year": 2100.0 * vones,
    }
    model_output = model.discipline.execute(model_input)
    outputs[scenario] = model_output

    name_split = scenario.split("SSP")
    ssp_idx = int(name_split[1][0])
    if ssp_idx != last_ssp_index:
        lines = lines_gen()
    line = next(lines)
    last_ssp_index = ssp_idx

    axes[0, 0].plot(
        years,
        model_output["rpk_per_capita"],
        label=scenario,
        linestyle=line,
        color=get_scenario_color(scenario),
    )
    axes[0, 1].plot(
        years,
        1.0e-12 * model_output["rpk_trend"],
        linestyle=line,
        color=get_scenario_color(scenario),
    )
    axes[1, 0].plot(
        years,
        model_output["load_factor"],
        linestyle=line,
        color=get_scenario_color(scenario),
    )
    axes[1, 1].plot(
        years,
        1.0e-12 * model_output["ask_trend"],
        linestyle=line,
        color=get_scenario_color(scenario),
    )

axes[0, 0].set_xlabel("Year")
axes[0, 0].set_ylabel("pax-km per hab.")
axes[0, 0].set_title("RPK per capita")
axes[0, 1].set_xlabel("Year")
axes[0, 1].set_ylabel("trillion pax-km")
axes[0, 1].set_title("RPK")
axes[1, 0].set_xlabel("Year")
axes[1, 0].set_ylabel("pax/seat [%]")
axes[1, 0].set_title("Load factor")
axes[1, 1].set_xlabel("Year")
axes[1, 1].set_ylabel("trillion seat-km")
axes[1, 1].set_title("ASK")
show()
