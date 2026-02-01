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
from noads.application.visualization import plot_tech_scenario_fleet_carriers
from noads.application.visualization import plot_tech_scenario_jet_fuel
from noads.application.visualization import plot_tech_scenarios_trends


# Load saved optimization results instead of re-running
def load_breakthrough_results(
    global_scenario_name,
    technology_index,
    drop_in_only=False,
    low_demand_formulation=False,
    preferential_energy=False,
    **kwargs,
):
    """Load previously saved optimization results without re-running the optimization."""
    return single_policy_scenario_optimization(
        global_scenario_name=global_scenario_name,
        technology_index=technology_index,
        drop_in_only=drop_in_only,
        low_demand_formulation=low_demand_formulation,
        preferential_energy=preferential_energy,
        load_optimum=True,  # Load existing results
        plot_optimum=False,  # Don't plot during loading
        save_optimum=False,  # Don't save again
        save_figs=False,  # Don't save figures during loading
        **kwargs,
    )


scenario_tech_outputs = {}

background_scenario = "SSP2-26"

# Load Baseline scenario results
baseline = [
    load_breakthrough_results(
        background_scenario, 0, drop_in_only=True, fossil_kerosene_only=True
    ),
    load_breakthrough_results(
        background_scenario, 1, drop_in_only=True, fossil_kerosene_only=True
    ),
    load_breakthrough_results(
        background_scenario, 2, drop_in_only=True, fossil_kerosene_only=True
    ),
]
scenario_tech_outputs["Baseline SSP2"] = baseline

# Load Breakthrough trend scenario results
break_trend = [
    load_breakthrough_results(background_scenario, 0),
    load_breakthrough_results(background_scenario, 1),
    load_breakthrough_results(background_scenario, 2),
]
scenario_tech_outputs["Breakthrough trend"] = break_trend

# Load Breakthrough availability scenario results (8.6% energy share)
break_availability = [
    load_breakthrough_results(background_scenario, 0, preferential_energy=True),
    load_breakthrough_results(background_scenario, 1, preferential_energy=True),
    load_breakthrough_results(background_scenario, 2, preferential_energy=True),
]
scenario_tech_outputs["Breakthrough availability"] = break_availability

# Load Breakthrough low-demand scenario results
break_lowdemand = [
    load_breakthrough_results(background_scenario, 0, low_demand_formulation=True),
    load_breakthrough_results(background_scenario, 1, low_demand_formulation=True),
    load_breakthrough_results(background_scenario, 2, low_demand_formulation=True),
]
scenario_tech_outputs["Breakthrough low-demand"] = break_lowdemand

# Generate comparison plots with loaded data
plot_tech_scenarios_trends(
    scenario_outputs=scenario_tech_outputs,
    colors=["#7F7F7F", "#FF7F0E", "#1F77B4", "#2CA02C"],
    save_fig=True,
    directory_filename="./single_policy/breakthrough",
)
plot_tech_scenario_jet_fuel(
    scenario_outputs=scenario_tech_outputs,
    colors=["#7F7F7F", "#FF7F0E", "#1F77B4", "#2CA02C"],
    save_fig=True,
    directory_filename="./single_policy/breakthrough_jetfuel",
    zoom_efuel=True,
)
plot_tech_scenario_fleet_carriers(
    scenario_outputs=scenario_tech_outputs,
    colors=["#7F7F7F", "#FF7F0E", "#1F77B4", "#2CA02C"],
    save_fig=True,
    directory_filename="./single_policy/breakthrough_carriers",
    zoom_battery=True,
)
