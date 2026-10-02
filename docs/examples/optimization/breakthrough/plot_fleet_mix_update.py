# Copyright 2025 ISAE-SUPAERO, https://www.isae-supaero.fr/en/
# Copyright 2025 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION
# OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

"""
Breakthrough aircraft, update model: fleet mix
==============================================
"""

from noads.application.examples import single_policy_scenario_optimization
from noads.application.scenario_setup import single_scenario_setup
from noads.application.visualization import plot_fleet_mix_variants

# %%
# Fleet mix of the breakthrough variants
# --------------------------------------
# One figure per technology scenario (Lower, Mid, Upper), with the three variants of
# the breakthrough case (trend, availability and low-demand), optimized with the
# update aircraft model (``aircraft_model="update"``), with the timeline of the paper
# (results in the ``update`` subfolder of the results directory) and with the extended
# timeline (``update-2100``).

BACKGROUND = "SSP2-26"
VARIANTS = {
    "Breakthrough trend": {},
    "Breakthrough availability": {"preferential_energy": True},
    "Breakthrough low-demand": {"low_demand_formulation": True},
}
TECHNOLOGIES = ("Lower", "Mid", "Upper")
TIMELINES = {
    "paper": ("", "2025-2075"),
    "extended": ("_2100", "2025-2100, new aircraft until 2080"),
}

for timeline, (suffix, period) in TIMELINES.items():
    for technology_index, technology in enumerate(TECHNOLOGIES):
        # the fleet differs by technology: infeasible designs are left out
        _, _, _, _, fleet = single_scenario_setup(
            "fleet-mix",
            BACKGROUND,
            technology_index=technology_index,
            plot_scenario_data=False,
            integrate_constraints=False,
            aircraft_model="update",
            end_year=2100.0 if timeline == "extended" else None,
            last_entry_into_service=2080.0 if timeline == "extended" else None,
            compile_jit=False,
        )
        variant_outputs = {
            variant: single_policy_scenario_optimization(
                global_scenario_name=BACKGROUND,
                technology_index=technology_index,
                load_optimum=True,
                plot_optimum=False,
                save_optimum=False,
                save_figs=False,
                aircraft_model="update",
                timeline=timeline,
                **options,
            )
            for variant, options in VARIANTS.items()
        }
        plot_fleet_mix_variants(
            variant_outputs,
            fleet,
            title=f"Fleet mix, update model, {technology} technology ({period})",
            save_fig=True,
            directory_filename=f"./fleet_mix_update_{technology.lower()}{suffix}.png",
        )
