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

"""Pre-computation of the optima of the update aircraft model.

Same scenarios, timeline (2025-2075, new aircraft until 2060) and solver settings as
the main paper results, with ``aircraft_model="update"``. Results go to the
``update`` subfolder of this directory.

Usage: ``python precompute_update.py [--extended] list`` prints the scenario names
(breakthrough variants first), and ``python precompute_update.py [--extended] <name>``
computes one scenario (skipped if it already exists), so that scenarios can be run in
parallel processes. With ``--extended``, the timeline is extended (until 2100, new
aircraft until 2080) and the results go to the ``update-2100`` subfolder.
"""

import logging
import os
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent
os.environ.setdefault("NOADS_RESULTS_DIR", str(RESULTS_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

from noads.application.examples import (  # noqa: E402
    single_policy_robust_scenario_optimization,
)
from noads.application.examples import single_policy_scenario_optimization  # noqa: E402

LOGGER = logging.getLogger("precompute_update")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

TECH_SUFFIXES = {0: "lowTech", 1: "midTech", 2: "upTech"}

# suffix: (drop_in_only, fossil_kerosene_only, low_demand, preferential_energy)
VARIANTS = {
    "": (False, False, False, False),
    "-Availability": (False, False, False, True),
    "-LowDemand": (False, False, True, False),
    "-DropIn": (True, False, False, False),
    "-DropIn-Availability": (True, False, False, True),
    "-DropIn-LowDemand": (True, False, True, False),
    "-Fossil": (True, True, False, False),
}

SINGLE = {}
for variant, flags in VARIANTS.items():
    for index, tech in TECH_SUFFIXES.items():
        SINGLE[f"SSP2-26{variant}-{tech}"] = ("SSP2-26", index, *flags)
for background in ("SSP1-19", "SSP5-45"):
    for index, tech in TECH_SUFFIXES.items():
        SINGLE[f"{background}-Fossil-{tech}"] = (
            background,
            index,
            *VARIANTS["-Fossil"],
        )

BACKGROUNDS = ["SSP2-34", "SSP2-26", "SSP2-19"]
ROBUST = {
    "robust-SSP2-midTech": ("robust-SSP2", False),
    "robust-SSP2-lowdemand-LowDemand-midTech": ("robust-SSP2-lowdemand", True),
}


def run(name, timeline="paper"):
    """Compute one scenario of the update model, unless already computed."""
    folder = "update-2100" if timeline == "extended" else "update"
    if (RESULTS_DIR / folder / name / "opt_result.json").is_file():
        LOGGER.info("SKIP %s (already computed)", name)
        return
    LOGGER.info("RUN  %s", name)
    if name in ROBUST:
        base_name, low_demand = ROBUST[name]
        single_policy_robust_scenario_optimization(
            scenario_name=base_name,
            global_scenario_names=BACKGROUNDS,
            carbon_budget_percent=3.0,
            technology_index=1,
            drop_in_only=False,
            low_demand_formulation=low_demand,
            preferential_energy=False,
            load_optimum=False,
            plot_optimum=False,
            save_optimum=True,
            save_figs=False,
            save_history_view=False,
            aircraft_model="update",
            timeline=timeline,
        )
    else:
        background, index, drop_in, fossil, low_demand, preferential = SINGLE[name]
        single_policy_scenario_optimization(
            global_scenario_name=background,
            carbon_budget_percent=3.0,
            technology_index=index,
            drop_in_only=drop_in,
            fossil_kerosene_only=fossil,
            low_demand_formulation=low_demand,
            preferential_energy=preferential,
            load_optimum=False,
            plot_optimum=False,
            save_optimum=True,
            save_figs=False,
            save_history_view=False,
            aircraft_model="update",
            timeline=timeline,
        )
    LOGGER.info("DONE %s", name)


BREAKTHROUGH = [
    f"SSP2-26{variant}-{tech}"
    for variant in ("", "-Availability", "-LowDemand")
    for tech in TECH_SUFFIXES.values()
]


if __name__ == "__main__":
    arguments = sys.argv[1:]
    timeline = "extended" if "--extended" in arguments else "paper"
    arguments = [argument for argument in arguments if argument != "--extended"]
    names = [
        *BREAKTHROUGH,
        *(name for name in [*SINGLE, *ROBUST] if name not in BREAKTHROUGH),
    ]
    if arguments == ["list"]:
        sys.stdout.write("\n".join(names) + "\n")
    else:
        for argument in arguments:
            run(argument, timeline)
