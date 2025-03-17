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

"""Decarbonization scenario pareto front based on two objectives."""

# %%
# # Multi-Objective Policy Trade-Off
# Now that scenario optimization is better understood, let's run a trade-off between a
# policy optimization with two different objectives.
#
# Similarly to the single scenario optimization, the background scenario is based on the
# SSP2-2.6.
from noads.application.examples import multi_policy_scenario_optimization
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

baseline_outputs = multi_policy_scenario_optimization(
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

availability_outputs = multi_policy_scenario_optimization(
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
