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

"""Decarbonization scenario pareto front based on two objectives."""

# %%
# # Multi-Objective Policy Trade-Off
# Now that scenario optimization is better understood, let's run a trade-off between a
# policy optimization with two different objectives.
#
# Similarly to the single scenario optimization, the background scenario is based on the
# SSP2-2.6.
from noads.application.examples import multi_objective_policy_scenario_optimization
from noads.application.visualization import plot_traffic_emissions_pareto_front

background_scenario = "SSP2-26"

# %%
# Similar to the single scenario, we will compare 2 scenarios:
# - Baseline: 5% of global production of electricity and biomass.
# - Extra availability: 8.6% of global production of electricity and biomass.
#
# For simplification, they will only be investigated with mid aircraft technology. The
# optimization history will only be shown for the baseline case.
#
# ## Baseline

baseline_outputs = multi_objective_policy_scenario_optimization(
    global_scenario_name=background_scenario,
    n_sub_optim=7,
    technology_index=1,
    drop_in_only=False,
    preferential_energy=False,
    plot_optimum=True,
    save_optimum=False,
)

# %%
# ## Extra Availability

availability_outputs = multi_objective_policy_scenario_optimization(
    global_scenario_name=background_scenario,
    n_sub_optim=7,
    technology_index=1,
    drop_in_only=False,
    preferential_energy=True,
    plot_optimum=True,
    save_optimum=False,
)

# %%
# # Traffic vs. Emissions
# Now let's visualize the Pareto front of Traffic and Emissions.

plot_traffic_emissions_pareto_front(
    scenario_outputs={
        "Baseline": baseline_outputs,
        "Availability": availability_outputs,
    },
    colors_markers=[("royalblue", "d"), ("deeppink", "s")],
    n_sub_opt=7,
)
