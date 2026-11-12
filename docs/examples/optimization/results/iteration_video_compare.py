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

"""Video comparing the iterations of several update optima.

Usage: ``python iteration_video_compare.py [--extended] <output name> <name> <name>...``
or, without names, the breakthrough variants of the Mid technology scenario (trend,
availability and low demand). The iterations of the optima, saved by
``precompute_update.py``, are played in sync (the optima that converge first stay on
their last iteration). Each variant has a row of fleet mixes per market and a row of
energy mixes, all on the same scales; the video is saved next to the results.
"""

import logging
import sys

import imageio_ffmpeg
import matplotlib as mpl
from iteration_video import ENERGY_PANELS
from iteration_video import FPS
from iteration_video import HOLD_SECONDS
from iteration_video import RESULTS_DIR
from iteration_video import evaluate_iterations
from iteration_video import frame_counts
from iteration_video import setup
from matplotlib.animation import FFMpegWriter
from matplotlib.pyplot import figure

from noads.application.background_scenario_data import co2_budget_2p0deg_66percent
from noads.application.visualization import propulsion_colors

LOGGER = logging.getLogger("iteration_video_compare")

MID_TECH_VARIANTS = {
    "SSP2-26-midTech": "Trend",
    "SSP2-26-Availability-midTech": "Availability",
    "SSP2-26-LowDemand-midTech": "Low demand",
}


def prepare(folder, name, timeline):
    """Fleet and energy mix series of every iteration of an optimum."""
    scenario, _, _, fleet, low_demand = setup(name, timeline)
    outputs = evaluate_iterations(folder / name, scenario, fleet)
    markets = {}
    for fleet_i in fleet.fleets:
        series = [
            (a.name, outputs[f"{a.name}.ask"] * 1e-12)
            for a in fleet_i.operating_aircraft
            if f"{a.name}.ask" in outputs
        ]
        avoided = f"{fleet_i.name}.ask_avoided"
        if low_demand and avoided in outputs:
            series.append(("Avoided demand", outputs[avoided] * 1e-12))
        markets[fleet_i.name] = series
    energy = {
        panel: [s for s in panel_series if s[0] in outputs]
        for panel, panel_series in ENERGY_PANELS.items()
    }
    budget = co2_budget_2p0deg_66percent * 3.0e-2
    co2 = outputs["cumulative.CO2"].reshape(len(outputs["iteration"]), -1)[:, -1]
    return {
        "outputs": outputs,
        "markets": markets,
        "energy": energy,
        "co2": 100.0 * co2 / budget,
        "years": outputs["year"],
        "n": len(outputs["iteration"]),
    }


def color(aircraft_name):
    """Color of an aircraft in the fleet mix."""
    if aircraft_name == "Avoided demand":
        return "tan"
    return next(
        (c for prop, c in propulsion_colors.items() if prop in aircraft_name), "black"
    )


def render(path, labels, runs, title):
    """Play the iterations of all runs in sync, one frame per iteration."""
    mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    market_names = list(runs[0]["markets"])
    n_cols = max(len(market_names), len(ENERGY_PANELS) + 1)
    n_rows = len(runs)
    fig = figure(figsize=(3.6 * n_cols, 2.6 * 2 * n_rows + 0.8), layout="constrained")
    grid = fig.add_gridspec(2 * n_rows, n_cols)
    fleet_axes = [
        [fig.add_subplot(grid[r, c]) for c in range(len(market_names))]
        for r in range(n_rows)
    ]
    energy_axes = [
        [fig.add_subplot(grid[n_rows + r, c]) for c in range(len(ENERGY_PANELS))]
        for r in range(n_rows)
    ]
    progress_axes = [
        fig.add_subplot(grid[n_rows + r, len(ENERGY_PANELS)]) for r in range(n_rows)
    ]
    # the scales are shared by all variants, to compare them
    fleet_top = {
        market: max(sum(s for _, s in run["markets"][market]).max() for run in runs)
        * 1.05
        for market in market_names
    }
    energy_top = (
        max(
            sum(run["outputs"][key] * 1e-12 for key, _, _ in series).max()
            for run in runs
            for series in run["energy"].values()
            if series
        )
        * 1.05
    )
    co2_top = max(run["co2"].max() for run in runs) * 1.05
    max_iterations = max(run["n"] for run in runs)

    def draw(k):
        for r, (label, run) in enumerate(zip(labels, runs, strict=True)):
            kr = min(k, run["n"] - 1)
            years = run["years"]
            for c, market in enumerate(market_names):
                ax = fleet_axes[r][c]
                ax.clear()
                series = run["markets"][market]
                ax.stackplot(
                    years,
                    [s[kr] for _, s in series],
                    colors=[color(n) for n, _ in series],
                    labels=[
                        n.replace(f"_{market}", "").replace("_", " ") for n, _ in series
                    ],
                )
                ax.set_xlim(years[0], years[-1])
                ax.set_ylim(0.0, fleet_top[market])
                if r == 0:
                    ax.set_title(market.replace("_", " "))
                if c == 0:
                    ax.set_ylabel(f"{label}\nASK [trillion seat km]")
                if r == 0 and c in (0, len(market_names) - 1):
                    ax.legend(loc="upper left", fontsize="xx-small", framealpha=0.5)
            for c, (panel, series) in enumerate(run["energy"].items()):
                ax = energy_axes[r][c]
                ax.clear()
                ax.set_xlim(years[0], years[-1])
                ax.set_ylim(0.0, energy_top)
                if r == 0:
                    ax.set_title(panel, fontsize="medium")
                if c == 0:
                    ax.set_ylabel(f"{label}\nEnergy [EJ / year]")
                if not series:
                    continue
                ax.stackplot(
                    years,
                    [run["outputs"][key][kr] * 1e-12 for key, _, _ in series],
                    colors=[col for _, _, col in series],
                    labels=[lab for _, lab, _ in series],
                )
                if r == 0:
                    ax.legend(loc="upper left", fontsize="xx-small", framealpha=0.5)
            ax = progress_axes[r]
            ax.clear()
            ax.plot(run["co2"], color="lightgray")
            ax.plot(run["co2"][: kr + 1], color="black")
            ax.plot(kr, run["co2"][kr], "o", color="crimson")
            ax.axhline(100.0, color="gray", linestyle=":")
            ax.set_xlim(0, max_iterations)
            ax.set_ylim(0.0, co2_top)
            if r == 0:
                ax.set_title("Cumulative CO2 [% of budget]", fontsize="medium")
            ax.set_xlabel(f"Iteration ({kr + 1} / {run['n']})", fontsize="small")
        fig.suptitle(
            f"{title}: iteration {k + 1} / {max_iterations}", fontsize="x-large"
        )

    writer = FFMpegWriter(fps=FPS, bitrate=6000)
    with writer.saving(fig, path, dpi=80):
        for k, count in enumerate(frame_counts(max_iterations)):
            draw(k)
            for _ in range(count):
                writer.grab_frame()
        for _ in range(FPS * HOLD_SECONDS):
            writer.grab_frame()
    LOGGER.info("saved %s", path)


if __name__ == "__main__":
    arguments = sys.argv[1:]
    timeline = "extended" if "--extended" in arguments else "paper"
    arguments = [argument for argument in arguments if argument != "--extended"]
    if arguments:
        output_name, *names = arguments
        names = {name: name for name in names}
    else:
        output_name = "compare-midTech-breakthrough"
        names = MID_TECH_VARIANTS
    folder = RESULTS_DIR / ("update-2100" if timeline == "extended" else "update")
    runs = [prepare(folder, name, timeline) for name in names]
    render(
        folder / f"{output_name}.mp4",
        list(names.values()),
        runs,
        "Update model, SSP2-26, Mid technology" if not arguments else output_name,
    )
