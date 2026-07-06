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

"""
Robust policy: comparison of trend vs low-demand variants
=========================================================
"""

from noads.application.examples import single_policy_robust_scenario_optimization
from noads.application.visualization import plot_multiple_multi_scenario_result

# %%
# Robust policy comparison
# ------------------------
# Gather results from the two robust variants (trend, low-demand) and plot
# multi-scenario comparisons showing how the single policy performs across all three
# SSP2 climate futures.

background_scenarios = ["SSP2-34", "SSP2-26", "SSP2-19"]


def load_robust_results(
    scenario_name,
    low_demand_formulation=False,
    preferential_energy=False,
):
    """Load saved robust optimization results."""
    return single_policy_robust_scenario_optimization(
        scenario_name=scenario_name,
        global_scenario_names=background_scenarios,
        carbon_budget_percent=3.0,
        technology_index=1,
        drop_in_only=False,
        low_demand_formulation=low_demand_formulation,
        preferential_energy=preferential_energy,
        load_optimum=True,
        plot_optimum=False,
        save_optimum=False,
        save_history_view=False,
        save_figs=False,
    )


# %%
# Load all variants
# ^^^^^^^^^^^^^^^^^
multi_scenario_results = {
    "Robust trend": load_robust_results("robust-SSP2"),
    "Robust low-demand": load_robust_results(
        "robust-SSP2-lowdemand",
        low_demand_formulation=True,
    ),
}

# %%
# Plot multi-scenario comparison
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
plot_multiple_multi_scenario_result(
    global_scenario_names=background_scenarios,
    multi_scenario_results=multi_scenario_results,
    colors=["#D95F02", "#1B9E77"],
    save_figs=False,
)
