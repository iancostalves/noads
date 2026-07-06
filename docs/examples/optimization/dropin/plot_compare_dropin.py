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
Drop-in SAF: comparison of all variants
=======================================
"""

from noads.application.examples import single_policy_scenario_optimization
from noads.application.visualization import plot_tech_scenario_jet_fuel
from noads.application.visualization import plot_tech_scenarios_trends

# %%
# Drop-in scenario comparison
# ---------------------------
# Gather results from all four drop-in variants (baseline, trend, availability,
# low-demand) and plot side-by-side comparisons of technology trends and jet fuel
# composition.

BACKGROUND = "SSP2-26"


def load_results(
    fossil_kerosene_only=False,
    low_demand_formulation=False,
    preferential_energy=False,
):
    """Load saved results for all three technology levels."""
    return [
        single_policy_scenario_optimization(
            global_scenario_name=BACKGROUND,
            technology_index=tech_idx,
            drop_in_only=True,
            fossil_kerosene_only=fossil_kerosene_only,
            low_demand_formulation=low_demand_formulation,
            preferential_energy=preferential_energy,
            load_optimum=True,
            plot_optimum=False,
            save_optimum=False,
            save_figs=False,
        )
        for tech_idx in range(3)
    ]


# %%
# Load all variants
# ^^^^^^^^^^^^^^^^^
scenario_tech_outputs = {
    "Baseline SSP2": load_results(fossil_kerosene_only=True),
    "Drop-in trend": load_results(),
    "Drop-in availability": load_results(preferential_energy=True),
    "Drop-in low-demand": load_results(low_demand_formulation=True),
}

# %%
# Technology scenario trends
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
plot_tech_scenarios_trends(
    scenario_outputs=scenario_tech_outputs,
    colors=["#7F7F7F", "#D62728", "#9467BD", "#BCBD22"],
    save_fig=False,
)

# %%
# Jet fuel composition over time
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
plot_tech_scenario_jet_fuel(
    scenario_outputs=scenario_tech_outputs,
    colors=["#7F7F7F", "#D62728", "#9467BD", "#BCBD22"],
    save_fig=False,
)
