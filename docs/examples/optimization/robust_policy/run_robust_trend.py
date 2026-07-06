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
Robust policy: trend resource allocation across SSP2 variants
=============================================================
"""

from noads.application.examples import single_policy_robust_scenario_optimization

# %%
# Robust optimization with trend resources
# ----------------------------------------
# Find a single policy that is feasible across SSP2-1.9, SSP2-2.6, and SSP2-3.4
# simultaneously. All propulsion architectures are available, with trend resource
# allocation.

background_scenarios = ["SSP2-34", "SSP2-26", "SSP2-19"]

result = single_policy_robust_scenario_optimization(
    scenario_name="robust-SSP2",
    global_scenario_names=background_scenarios,
    carbon_budget_percent=3.0,
    technology_index=1,
    drop_in_only=False,
    low_demand_formulation=False,
    preferential_energy=False,
    load_optimum=True,
    plot_optimum=True,
    save_optimum=False,
    save_history_view=False,
    save_figs=False,
)
