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

"""Breakthrough (H2GT, H2FC, Elec) aircraft + SAF."""

# ruff: noqa: FURB113

from noads.application.examples import single_policy_scenario_optimization
from noads.application.visualization import plot_tech_scenario_jet_fuel
from noads.application.visualization import plot_tech_scenarios_trends

# %%
# # Scenario Optimization
# Now that the elementary models are understood, let's run a scenario optimization.
# We'll use the SSP2-2.6 as our background scenario, representing a global commitment to
# stay below +2°C temperature increase, but without major shifts from current trends.


# Load saved optimization results instead of re-running
def load_dropin_results(
    global_scenario_name,
    technology_index,
    drop_in_only=True,
    fossil_kerosene_only=False,
    low_demand_formulation=False,
    preferential_energy=False,
    **kwargs,
):
    """Load previously saved optimization results without re-running the optimization."""
    return single_policy_scenario_optimization(
        global_scenario_name=global_scenario_name,
        technology_index=technology_index,
        drop_in_only=drop_in_only,
        fossil_kerosene_only=fossil_kerosene_only,
        low_demand_formulation=low_demand_formulation,
        preferential_energy=preferential_energy,
        load_optimum=True,  # Load existing results
        plot_optimum=False,  # Don't plot during loading
        save_optimum=False,  # Don't save again
        save_figs=False,  # Don't save figures during loading
        **kwargs,
    )


background_scenario = "SSP2-26"
scenario_tech_outputs = {}

# Load Baseline scenario results
baseline = [
    load_dropin_results(background_scenario, 0, fossil_kerosene_only=True),
    load_dropin_results(background_scenario, 1, fossil_kerosene_only=True),
    load_dropin_results(background_scenario, 2, fossil_kerosene_only=True),
]
scenario_tech_outputs["Baseline SSP2"] = baseline

# Load Drop-in trend scenario results
dropin_trend = [
    load_dropin_results(background_scenario, 0),
    load_dropin_results(background_scenario, 1),
    load_dropin_results(background_scenario, 2),
]
scenario_tech_outputs["Drop-in trend"] = dropin_trend

# Load Drop-in availability scenario results (8.6% energy share)
dropin_availability = [
    load_dropin_results(background_scenario, 0, preferential_energy=True),
    load_dropin_results(background_scenario, 1, preferential_energy=True),
    load_dropin_results(background_scenario, 2, preferential_energy=True),
]
scenario_tech_outputs["Drop-in availability"] = dropin_availability

# Load Drop-in low-demand scenario results
dropin_lowdemand = [
    load_dropin_results(background_scenario, 0, low_demand_formulation=True),
    load_dropin_results(background_scenario, 1, low_demand_formulation=True),
    load_dropin_results(background_scenario, 2, low_demand_formulation=True),
]
scenario_tech_outputs["Drop-in low-demand"] = dropin_lowdemand

# Generate comparison plots with loaded data
plot_tech_scenarios_trends(
    scenario_outputs=scenario_tech_outputs,
    colors=["#7F7F7F", "#D62728", "#9467BD", "#BCBD22"],
    save_fig=True,
    directory_filename="./single_policy/dropin",
)
plot_tech_scenario_jet_fuel(
    scenario_outputs=scenario_tech_outputs,
    colors=["#7F7F7F", "#D62728", "#9467BD", "#BCBD22"],
    save_fig=True,
    directory_filename="./single_policy/dropin_jetfuel",
)
