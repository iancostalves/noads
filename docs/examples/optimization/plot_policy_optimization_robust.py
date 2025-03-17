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

"""Optimal decarbonization scenario robust to several background scenarios."""

from noads.application.examples import single_policy_robust_scenario_optimization

# %%
# # Scenario-Robust Optimization
# Now that scenario optimization is better understood, let's run a scenario optimization
# robust to a change in background scenario.
#
# As the single scenario was based on the SSP2-2.6, now we will include the scenarios
# SSP2-1.9 and SSP2-3.4 to the analysis, representing two adjacent scenarios to our
# previous target.

background_scenarios = ["SSP2-34", "SSP2-26", "SSP2-19"]

# %%
# Similar to the single scenario, we will compare 3 scenarios:
# - Baseline: minimize cumulative CO2 emissions, subject to trend traffic and 5% of
# global production of electricity and biomass.
# - Extra availability: minimize cumulative CO2 emissions, subject to trend traffic and
# 8.6% of global production of electricity and biomass.
# - Low demand: minimize the burden of demand avoidance, subject to 5% of global
# production of electricity and biomass, and constrain cumulative emissions to 3% of the
# 2°C carbon budget.
#
# For simplification, they will only be investigated with mid aircraft technology. The
# optimization history will only be shown for the baseline case.
#
# ## Baseline

single_policy_robust_scenario_optimization(
    scenario_name="robust-SSP2-26",
    global_scenario_names=background_scenarios,
    carbon_budget_percent=3.0,
    technology_index=1,
    drop_in_only=False,
    low_demand_formulation=False,
    preferential_energy=False,
    plot_optimum=True,
    save_optimum=False,
    plot_history_view=True,
)

# %%
# ## Extra Availability

single_policy_robust_scenario_optimization(
    scenario_name="robust-SSP2-26",
    global_scenario_names=background_scenarios,
    carbon_budget_percent=3.0,
    technology_index=1,
    drop_in_only=False,
    low_demand_formulation=False,
    preferential_energy=True,
    plot_optimum=True,
    save_optimum=False,
    plot_history_view=False,
)

# %%
# ## Low demand

single_policy_robust_scenario_optimization(
    scenario_name="robust-SSP2-26",
    global_scenario_names=background_scenarios,
    carbon_budget_percent=3.0,
    technology_index=1,
    drop_in_only=False,
    low_demand_formulation=True,
    preferential_energy=False,
    plot_optimum=True,
    save_optimum=False,
    plot_history_view=False,
)
