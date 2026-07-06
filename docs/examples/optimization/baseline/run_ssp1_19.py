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
Baseline: SSP1-1.9 (fossil kerosene only)
=========================================
"""

from noads.application.examples import single_policy_scenario_optimization

# %%
# Baseline optimization under SSP1-1.9
# ------------------------------------
# Fossil kerosene only, no alternative fuels. The SSP1-1.9 pathway represents a
# strong sustainability scenario with a ~1.5°C temperature target — the tightest
# carbon budget.

results_per_tech = []
for technology_index in range(3):
    result = single_policy_scenario_optimization(
        global_scenario_name="SSP1-19",
        carbon_budget_percent=3.0,
        technology_index=technology_index,
        drop_in_only=True,
        fossil_kerosene_only=True,
        low_demand_formulation=False,
        preferential_energy=False,
        load_optimum=True,
        plot_optimum=technology_index == 1,  # plot mid-tech only
        save_optimum=False,
        save_figs=False,
        save_history_view=False,
    )
    results_per_tech.append(result)
