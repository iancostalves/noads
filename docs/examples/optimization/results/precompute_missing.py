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

"""One-time pre-computation of the optima missing from the shipped results.

Run from this directory (or with ``NOADS_RESULTS_DIR`` pointing at it) to
produce the ``opt_result.json`` files needed by the example gallery:
the fossil-only baselines under SSP1-1.9 and SSP5-4.5, and the two robust
multi-scenario policies. Skips any scenario whose result already exists.
"""

import logging
import os
from pathlib import Path

RESULTS_DIR = Path(__file__).parent
os.environ.setdefault("NOADS_RESULTS_DIR", str(RESULTS_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

from noads.application.examples import (  # noqa: E402
    single_policy_robust_scenario_optimization,
)
from noads.application.examples import single_policy_scenario_optimization  # noqa: E402

LOGGER = logging.getLogger("precompute")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

TECH_SUFFIXES = {0: "lowTech", 1: "midTech", 2: "upTech"}


def _exists(scenario_name: str) -> bool:
    return (RESULTS_DIR / scenario_name / "opt_result.json").is_file()


# Fossil-only baselines (SSP1-1.9 and SSP5-4.5, three technology levels each).
for background in ["SSP1-19", "SSP5-45"]:
    for tech_index, tech_suffix in TECH_SUFFIXES.items():
        name = f"{background}-Fossil-{tech_suffix}"
        if _exists(name):
            LOGGER.info("SKIP %s (already computed)", name)
            continue
        LOGGER.info("RUN  %s", name)
        single_policy_scenario_optimization(
            global_scenario_name=background,
            carbon_budget_percent=3.0,
            technology_index=tech_index,
            drop_in_only=True,
            fossil_kerosene_only=True,
            low_demand_formulation=False,
            preferential_energy=False,
            load_optimum=False,
            plot_optimum=False,
            save_optimum=True,
            save_figs=False,
            save_history_view=False,
        )
        LOGGER.info("DONE %s", name)

# Robust multi-scenario policies (trend and low-demand, mid technology).
BACKGROUNDS = ["SSP2-34", "SSP2-26", "SSP2-19"]
for base_name, low_demand, full_name in [
    ("robust-SSP2", False, "robust-SSP2-midTech"),
    ("robust-SSP2-lowdemand", True, "robust-SSP2-lowdemand-LowDemand-midTech"),
]:
    if _exists(full_name):
        LOGGER.info("SKIP %s (already computed)", full_name)
        continue
    LOGGER.info("RUN  %s", full_name)
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
    )
    LOGGER.info("DONE %s", full_name)

LOGGER.info("All missing optima computed.")
