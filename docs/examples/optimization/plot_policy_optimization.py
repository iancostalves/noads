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

"""Optimal decarbonization scenario based on a single objective."""

# ruff: noqa: FURB113

from noads.application.examples import single_policy_scenario_optimization
from noads.application.visualization import plot_tech_scenario_comparison

# %%
# # Scenario Optimization
# Now that the elementary models are understood, let's run a scenario optimization.
# We'll use the SSP2-2.6 as our background scenario, representing a global commitment to
# stay below +2°C temperature increase, but without major shifts from current trends.

background_scenario = "SSP2-26"

# %%
# We will compare 3 scenarios, all with introduction of new aircraft with alternative
# energy carriers (Breakthrough mitigation):
# - Baseline: minimize cumulative CO2 emissions, subject to trend traffic and 5% of
# global production of electricity and biomass.
#
# - Extra availability: minimize cumulative CO2 emissions, subject to trend traffic and
# 8.6% of global production of electricity and biomass.
#
# - Low demand: minimize the burden of demand avoidance, subject to 5% of global
# production of electricity and biomass, and constrain cumulative emissions to 3% of the
# 2°C carbon budget.
#
# Each of them further investigated with lower, mid, and upper aircraft technology (9
# scenarios in total). The optimization results will only be shown for the lower and
# upper technology , to compare the fleet composition and energy mix between the most
# optimistic and pessimistic assumptions. The optimization history will only be shown
# for the mid technology, to compare overall behaviour of the optimization problem as
# constraints and objectives change.

scenario_tech_outputs = {}

# %%
# ## Baseline
# First let's run the baseline scenario with lower aircraft technology.
#
# Interactive sankey diagrams may be visualized for the years
# [2030](./sankey_SSP2-26-low_2030.html), [2045](./sankey_SSP2-26-low_2045.html), and
# [2060](./sankey_SSP2-26-low_2060.html).

baseline_outputs = []
baseline_outputs.append(
    single_policy_scenario_optimization(
        global_scenario_name=background_scenario,
        carbon_budget_percent=3.0,
        technology_index=0,
        drop_in_only=False,
        low_demand_formulation=False,
        preferential_energy=False,
        plot_optimum=True,
        save_optimum=False,
    )
)

# %%
# Then with mid aircraft technology.

baseline_outputs.append(
    single_policy_scenario_optimization(
        global_scenario_name=background_scenario,
        carbon_budget_percent=3.0,
        technology_index=1,
        drop_in_only=False,
        low_demand_formulation=False,
        preferential_energy=False,
        plot_optimum=False,
        save_optimum=False,
        plot_history_view=True,
    )
)

# %%
# Finally with upper aircraft technology.
#
# Interactive sankey diagrams may be visualized for the years
# [2030](./sankey_SSP2-26-up_2030.html), [2045](./sankey_SSP2-26-up_2045.html), and
# [2060](./sankey_SSP2-26-up_2060.html).

baseline_outputs.append(
    single_policy_scenario_optimization(
        global_scenario_name=background_scenario,
        carbon_budget_percent=3.0,
        technology_index=2,
        drop_in_only=False,
        low_demand_formulation=False,
        preferential_energy=False,
        plot_optimum=True,
        save_optimum=False,
    )
)
scenario_tech_outputs["Baseline"] = baseline_outputs

# %%
# ## Extra Availability
# Now instead of 5%, we'll give aviation 8.6% of global energy production. This is
# equivalent to the sector 2019 share of oil consumption.
#
# With lower aircraft technology.
#
# Interactive sankey diagrams may be visualized for the years
# [2030](./sankey_SSP2-26-E-low_2030.html), [2045](./sankey_SSP2-26-E-low_2045.html),
# and [2060](./sankey_SSP2-26-E-low_2060.html).

availability_outputs = []
availability_outputs.append(
    single_policy_scenario_optimization(
        global_scenario_name=background_scenario,
        carbon_budget_percent=3.0,
        technology_index=0,
        drop_in_only=False,
        low_demand_formulation=False,
        preferential_energy=True,
        plot_optimum=True,
        save_optimum=False,
    )
)

# %%
# Then with mid technology.

availability_outputs.append(
    single_policy_scenario_optimization(
        global_scenario_name=background_scenario,
        carbon_budget_percent=3.0,
        technology_index=1,
        drop_in_only=False,
        low_demand_formulation=False,
        preferential_energy=True,
        plot_optimum=False,
        save_optimum=False,
        plot_history_view=True,
    )
)

# %%
# And upper technology.
#
# Interactive sankey diagrams may be visualized for the years
# [2030](./sankey_SSP2-26-E-uplow_2030.html), [2045](./sankey_SSP2-26-E-up_2045.html),
# and [2060](./sankey_SSP2-26-E-up_2060.html).

availability_outputs.append(
    single_policy_scenario_optimization(
        global_scenario_name=background_scenario,
        carbon_budget_percent=3.0,
        technology_index=2,
        drop_in_only=False,
        low_demand_formulation=False,
        preferential_energy=True,
        plot_optimum=True,
        save_optimum=False,
    )
)
scenario_tech_outputs["Availability"] = availability_outputs

# %%
# ## Low demand
# Also, let's run the low demand scenario, where trend traffic is not entirely met, and
# instead we minimize the burden of demand avoidance, such that cumulative emissions are
# constrained to 3% of the 2°C carbon budget.
#
# With lower aircraft technology.
#
# Interactive sankey diagrams may be visualized for the years
# [2030](./sankey_SSP2-26-LD-low_2030.html), [2045](./sankey_SSP2-26-LD-low_2045.html),
# and [2060](./sankey_SSP2-26-LD-low_2060.html).

low_demand_outputs = []
low_demand_outputs.append(
    single_policy_scenario_optimization(
        global_scenario_name=background_scenario,
        carbon_budget_percent=3.0,
        technology_index=0,
        drop_in_only=False,
        low_demand_formulation=True,
        preferential_energy=False,
        plot_optimum=True,
        save_optimum=False,
    )
)

# %%
# Mid technology.

low_demand_outputs.append(
    single_policy_scenario_optimization(
        global_scenario_name=background_scenario,
        carbon_budget_percent=3.0,
        technology_index=1,
        drop_in_only=False,
        low_demand_formulation=True,
        preferential_energy=False,
        plot_optimum=False,
        save_optimum=False,
        plot_history_view=True,
    )
)

# %%
# And upper technology.
#
# Interactive sankey diagrams may be visualized for the years
# [2030](./sankey_SSP2-26-LD-up_2030.html), [2045](./sankey_SSP2-26-LD-up_2045.html),
# and [2060](./sankey_SSP2-26-LD-up_2060.html).

low_demand_outputs.append(
    single_policy_scenario_optimization(
        global_scenario_name=background_scenario,
        carbon_budget_percent=3.0,
        technology_index=2,
        drop_in_only=False,
        low_demand_formulation=True,
        preferential_energy=False,
        plot_optimum=True,
        save_optimum=False,
    )
)
scenario_tech_outputs["Low-demand"] = low_demand_outputs

# %%
# # Scenario Comparison
# Finally, let's visualize the comparison between scenarios.

plot_tech_scenario_comparison(
    scenario_outputs=scenario_tech_outputs,
    colors=["royalblue", "deeppink", "forestgreen"],
)
