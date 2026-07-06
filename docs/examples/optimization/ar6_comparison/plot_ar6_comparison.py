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
Validation of NOADS results against the AR6 scenario ensemble
=============================================================
"""

from json import load
from os import environ
from os import walk
from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.pyplot import subplots
from numpy import argwhere
from numpy import array
from numpy import isnan
from numpy import ravel

from noads.application.background_scenario_data import get_ar6_output_data

# %%
# AR6 ensemble comparison
# -----------------------
# Load all NOADS optimization results and overlay them on the IPCC AR6 integrated
# assessment model ensemble. This validates that NOADS trajectories fall within the
# expected range for final energy (EJ/yr) and CO₂ emissions (MtCO₂/yr).

# The directory holding the pre-computed optimization results
# (shared across the optimization examples via the NOADS_RESULTS_DIR variable).
directory = Path(environ.get("NOADS_RESULTS_DIR", "results"))

ar6_data, years_array = get_ar6_output_data(
    start_year=2025, end_year=2080, plot_data=False
)

fig, axes = subplots(1, 2, layout="constrained")
fig.set_size_inches(8, 3)

# %%
# Plot AR6 ensemble (grey background)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
for variable, model_scenarios in ar6_data.items():
    line = 0 if "energy" in variable else 1
    for _model_name, scenarios_data in model_scenarios.items():
        for _scenario_name, scenario_data in scenarios_data.items():
            data_array = array(scenario_data)
            valid_indices = ravel(argwhere(~isnan(scenario_data)))
            axes[line].plot(
                years_array[valid_indices],
                data_array[valid_indices],
                color="k",
                alpha=0.05,
            )

# %%
# Overlay NOADS results (blue)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Single-scenario results expose flat ``year``/``ask``/``CO2`` keys; the
# scenario-robust results expose a shared ``fixed.year`` and per-scenario prefixed
# keys (e.g. ``SSP2-26.CO2``). Both kinds are overlaid.


def iter_trajectories(scenario_data):
    """Yield (years, ask, energy_per_ask, CO2) for every trajectory in a result."""
    if "year" in scenario_data:
        yield (
            scenario_data["year"],
            array(scenario_data["ask"]),
            array(scenario_data["mean_energy_per_ask"]),
            array(scenario_data["CO2"]),
        )
        return
    # Scenario-robust result: one trajectory per background scenario prefix.
    years = scenario_data.get("fixed.year")
    if years is None:
        return
    prefixes = sorted({
        key.rsplit(".CO2", 1)[0] for key in scenario_data if key.endswith(".CO2")
    })
    for prefix in prefixes:
        if f"{prefix}.mean_energy_per_ask" not in scenario_data:
            continue
        yield (
            years,
            array(scenario_data[f"{prefix}.ask"]),
            array(scenario_data[f"{prefix}.mean_energy_per_ask"]),
            array(scenario_data[f"{prefix}.CO2"]),
        )


for _path, folders, _files in walk(directory):
    for folder_name in folders:
        result_path = directory / folder_name / "opt_result.json"
        if not result_path.exists():
            continue
        with result_path.open() as f:
            result = load(f)

        scenario_data = {**result["inputs"], **result["outputs"]}
        for years, ask, energy_per_ask, emissions in iter_trajectories(scenario_data):
            axes[0].plot(
                years,
                energy_per_ask * ask * 1.0e-12,
                ".-",
                color="blue",
                alpha=0.2,
            )
            axes[1].plot(
                years,
                emissions * 1.0e-12,
                ".-",
                color="blue",
                alpha=0.2,
            )

# %%
# Format axes
# ^^^^^^^^^^^
axes[0].set_xlim(2025, 2075)
axes[0].set_ylim(0, 50)
axes[0].set_title("Final Energy")
axes[0].set_ylabel("EJ / yr")
axes[0].set_xlabel("Year")

axes[1].set_xlim(2025, 2075)
axes[1].set_ylim(0, 10000)
axes[1].set_title("Emissions")
axes[1].set_ylabel("Mt CO2 / yr")
axes[1].set_xlabel("Year")

custom_lines = [
    Line2D([0], [0], color="k", alpha=0.2, lw=2),
    Line2D([0], [0], marker=".", color="blue", alpha=0.4, lw=2),
]

axes[1].legend(custom_lines, ["AR6 scenario ensemble", "This work"], loc="upper left")
