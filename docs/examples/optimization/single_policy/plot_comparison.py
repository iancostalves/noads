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

"""Scenario comparisson with AR6."""

# ruff: noqa: FURB113
from json import load
from os import walk
from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.pyplot import subplots
from numpy import argwhere
from numpy import array
from numpy import isnan
from numpy import ravel

from noads.application.background_scenario_data import get_ar6_output_data

directory = "./single_policy"

ar6_data, years_array = get_ar6_output_data(
    start_year=2025, end_year=2080, plot_data=False
)

fig, axes = subplots(1, 2, layout="constrained")
fig.set_size_inches(8, 3)

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
                # label="AR6 scenario ensemble" if axes[line].lines else "_",
                alpha=0.05,
            )

for _path, folders, _files in walk(directory):
    for folder_name in folders:
        with Path(f"{directory}/{folder_name}/opt_result.json").open() as f:
            result = load(f)

            scenario_data = {**result["inputs"], **result["outputs"]}
            years = scenario_data["year"]

            energy_per_ask = array(scenario_data["mean_energy_per_ask"])
            ask = array(scenario_data["ask"])
            final_energy = energy_per_ask * ask

            axes[0].plot(
                years,
                final_energy * 1.0e-12,
                ".-",
                color="blue",
                alpha=0.2,
            )

            emissions = array(scenario_data["CO2"])

            axes[1].plot(
                years,
                emissions * 1.0e-12,
                ".-",
                color="blue",
                # label="This work" if folder_name == folders[0] else "_",
                alpha=0.2,
            )

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

fig.savefig("ar6_comparison.pdf")
