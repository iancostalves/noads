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

"""Scenario-Robust Baseline Policy."""

from noads.application.examples import single_policy_robust_scenario_optimization
from noads.application.visualization import plot_multiple_multi_scenario_result


# Load saved optimization results instead of re-running
def load_robust_results(
    scenario_name,
    global_scenario_names,
    drop_in_only=False,
    low_demand_formulation=False,
    preferential_energy=False,
    **kwargs,
):
    """Load previously saved robust optimization results without re-running the optimization."""
    return single_policy_robust_scenario_optimization(
        scenario_name=scenario_name,
        global_scenario_names=global_scenario_names,
        carbon_budget_percent=3.0,
        technology_index=1,
        drop_in_only=drop_in_only,
        low_demand_formulation=low_demand_formulation,
        preferential_energy=preferential_energy,
        load_optimum=True,
        plot_optimum=False,
        save_optimum=False,
        save_history_view=False,
        save_figs=False,
        **kwargs,
    )


background_scenarios = ["SSP2-34", "SSP2-26", "SSP2-19"]

# Dictionary to store all multi-scenario results
multi_scenario_results = {}

# Load Robust trend results (conventional availability)
robust_trend = load_robust_results(
    scenario_name="robust-SSP2",
    global_scenario_names=background_scenarios,
    drop_in_only=False,
    low_demand_formulation=False,
    preferential_energy=False,
)
multi_scenario_results["Robust trend"] = robust_trend

# Load Robust availability results (8.6% energy share)
# robust_availability = load_robust_results(
#     scenario_name="robust-SSP2-availability",
#     global_scenario_names=background_scenarios,
#     drop_in_only=False,
#     low_demand_formulation=False,
#     preferential_energy=True,
# )
# multi_scenario_results["Robust availability"] = robust_availability

# Load Robust low-demand results
robust_lowdemand = load_robust_results(
    scenario_name="robust-SSP2-lowdemand",
    global_scenario_names=background_scenarios,
    drop_in_only=False,
    low_demand_formulation=True,
    preferential_energy=False,
)
multi_scenario_results["Robust low-demand"] = robust_lowdemand

# Generate comparison plots across all three robust optimizations
plot_multiple_multi_scenario_result(
    global_scenario_names=background_scenarios,
    multi_scenario_results=multi_scenario_results,
    colors=["#D95F02", "#1B9E77"],  # "#E7298A"
    save_figs=True,
    directory_path="./robust_policy",
)
