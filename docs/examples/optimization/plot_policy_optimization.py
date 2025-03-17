# Copyright 2025 ISAE-SUPAERO, https://www.isae-supaero.fr/en/
# Copyright 2025 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This work is licensed under a BSD 0-Clause License.
#
# Permission to use, copy, modify, and/or distribute this software
# for any purpose with or without fee is hereby granted.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL
# WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL
# THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT,
# OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING
# FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
# NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
# WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

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
scenario_tech_outputs = {}

# %%
# # # Baseline
# First let's run the baseline scenario: minimize cumulative CO2 emissions, subject to
# trend traffic and 5% of global production of electricity and biomass.
#
# At first with lower aircraft technology.

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
        save_optimum=True,
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
        low_demand_formulation=True,
        preferential_energy=False,
        plot_optimum=True,
        save_optimum=True,
    )
)

# %%
# Finally with upper aircraft technology.

baseline_outputs.append(
    single_policy_scenario_optimization(
        global_scenario_name=background_scenario,
        carbon_budget_percent=3.0,
        technology_index=2,
        drop_in_only=False,
        low_demand_formulation=True,
        preferential_energy=False,
        plot_optimum=True,
        save_optimum=True,
    )
)
scenario_tech_outputs["Baseline"] = baseline_outputs

# %%
# # # Extra Availability
# Now instead of 5%, we'll give aviation 8.6% of global energy production. This is
# equivalent to the sector 2019 share of oil consumption.
#
# With lower aircraft technology.

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
        save_optimum=True,
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
        plot_optimum=True,
        save_optimum=True,
    )
)

# %%
# And upper technology.

availability_outputs.append(
    single_policy_scenario_optimization(
        global_scenario_name=background_scenario,
        carbon_budget_percent=3.0,
        technology_index=2,
        drop_in_only=False,
        low_demand_formulation=False,
        preferential_energy=True,
        plot_optimum=True,
        save_optimum=True,
    )
)
scenario_tech_outputs["Availability"] = availability_outputs

# %%
# # # Low demand
# Also, let's run the low demand scenario: trend traffic is not entirely met and
# instead minimize the burden of demand avoidance, subject to 5% of global production of
# electricity and biomass, and constrain cumulative emissions to 3% of the 2°C carbon
# budget.
#
# With lower aircraft technology.

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
        save_optimum=True,
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
        plot_optimum=True,
        save_optimum=True,
    )
)

# %%
# And upper technology.

low_demand_outputs.append(
    single_policy_scenario_optimization(
        global_scenario_name=background_scenario,
        carbon_budget_percent=3.0,
        technology_index=2,
        drop_in_only=False,
        low_demand_formulation=True,
        preferential_energy=False,
        plot_optimum=True,
        save_optimum=True,
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
