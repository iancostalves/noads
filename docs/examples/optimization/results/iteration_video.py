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

"""Video of the fleet and energy mixes along the iterations of an update optimum.

Usage: ``python iteration_video.py [--extended] <name>``, with ``<name>`` a
single-scenario optimum of ``precompute_update.py`` (``list`` prints them).

The design of every optimizer iteration, saved in ``iterations.npz`` next to the
optimum, is evaluated with the scenario model. The fleet and energy mixes of every
iteration are cached in ``iteration_outputs.npz``, then rendered as one video frame
each in ``iterations.mp4`` (requires ``imageio-ffmpeg``).
"""

import logging
import os
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent
os.environ.setdefault("NOADS_RESULTS_DIR", str(RESULTS_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import imageio_ffmpeg  # noqa: E402
import matplotlib as mpl  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.pyplot import subplots  # noqa: E402
from numpy import array  # noqa: E402
from numpy import cumsum  # noqa: E402
from numpy import load  # noqa: E402
from numpy import savez_compressed  # noqa: E402
from numpy import split  # noqa: E402
from precompute_update import SINGLE  # noqa: E402

from noads.application.background_scenario_data import (  # noqa: E402
    co2_budget_2p0deg_66percent,
)
from noads.application.examples import EXTENDED_TIMELINE  # noqa: E402
from noads.application.examples import HISTORY_FILE  # noqa: E402
from noads.application.scenario_setup import END_YEAR  # noqa: E402
from noads.application.scenario_setup import LAST_ENTRY_INTO_SERVICE  # noqa: E402
from noads.application.scenario_setup import single_scenario_setup  # noqa: E402
from noads.application.visualization import propulsion_colors  # noqa: E402

LOGGER = logging.getLogger("iteration_video")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

OUTPUTS_FILE = "iteration_outputs.npz"
VIDEO_FILE = "iterations.mp4"
FPS = 12
HOLD_SECONDS = 3

# Energy mix panels: produced energies stacked by pathway, then the final carriers
# of the alternative aircraft
ENERGY_PANELS = {
    "Jet-A": ("Fossil", "Biofuel", "Electrofuel"),
    "Biofuel": ("HEFA", "ATJ", "FT"),
    "Gaseous hydrogen": ("Electrolysis", "Gas_reforming"),
    "LH2 and battery": ("H2_liquefaction", "Charging"),
}
PATHWAY_COLORS = {
    "Fossil": "dimgray",
    "Biofuel": "olivedrab",
    "Electrofuel": "darkorange",
    "HEFA": "olivedrab",
    "ATJ": "yellowgreen",
    "FT": "darkgreen",
    "Electrolysis": "deepskyblue",
    "Gas_reforming": "slategray",
    "H2_liquefaction": "royalblue",
    "Charging": "limegreen",
}
PATHWAY_LABELS = {
    "Gas_reforming": "Gas reforming",
    "H2_liquefaction": "Liquid H2",
    "Charging": "Battery",
}


def setup(name, timeline):
    """The scenario of an update optimum, as optimized by ``precompute_update.py``."""
    background, index, drop_in, fossil, low_demand, preferential = SINGLE[name]
    end_year, last_eis = (
        EXTENDED_TIMELINE
        if timeline == "extended"
        else (END_YEAR["update"], LAST_ENTRY_INTO_SERVICE["update"])
    )
    scenario, design_space, _, energy_mix, fleet = single_scenario_setup(
        name=name,
        background_scenario_name=background,
        start_year=2025.0,
        end_year=end_year,
        technology_index=index,
        integrate_constraints=False,
        demand_aversion=low_demand,
        drop_in_only=drop_in,
        fossil_kerosene_only=fossil,
        preferential_energy=preferential,
        aircraft_model="update",
        last_entry_into_service=last_eis,
    )
    return scenario, design_space, energy_mix, fleet, low_demand


def evaluate_iterations(folder, scenario, fleet):
    """Fleet and energy mixes of every iteration, cached next to the optimum."""
    cache = folder / OUTPUTS_FILE
    history = load(folder / HISTORY_FILE)
    if cache.is_file():
        outputs = dict(load(cache))
        if len(outputs["iteration"]) == len(history["iteration"]):
            return outputs
    names = [str(name) for name in history["names"]]
    pathways = [p for panel in ENERGY_PANELS.values() for p in panel]
    keys = [f"{a.name}.ask" for a in fleet.operating_aircraft]
    keys += [f"{f.name}.ask_avoided" for f in fleet.fleets]
    keys += [f"{p}.production" for p in pathways]
    keys += ["cumulative.CO2"]
    rows = {key: [] for key in keys}
    for i, x_vect in enumerate(history["x"]):
        x_dict = dict(
            zip(names, split(x_vect, cumsum(history["sizes"])[:-1]), strict=True)
        )
        data = scenario.discipline.execute(x_dict)
        for key in keys:
            if key in data:
                rows[key].append(array(data[key]))
        if i % 50 == 0:
            LOGGER.info("evaluated iteration %d / %d", i + 1, len(history["x"]))
    outputs = {key: array(values) for key, values in rows.items() if values}
    outputs["year"] = array(data["year"])
    outputs["iteration"] = history["iteration"]
    savez_compressed(cache, **outputs)
    return outputs


def render(folder, name, outputs, fleet, low_demand):
    """One frame per iteration: fleet mix per market, energy mix, objective."""
    mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    years = outputs["year"]
    iterations = outputs["iteration"]
    budget = co2_budget_2p0deg_66percent * 3.0e-2
    co2 = outputs["cumulative.CO2"].reshape(len(iterations), -1)[:, -1] / budget
    n_markets = len(fleet.fleets)
    fig, axes = subplots(
        2, n_markets, figsize=(4.0 * n_markets, 7.5), layout="constrained"
    )

    def market_series(fleet_i):
        aircraft = [a for a in fleet_i.operating_aircraft if f"{a.name}.ask" in outputs]
        series = [(a.name, outputs[f"{a.name}.ask"] * 1e-12) for a in aircraft]
        avoided = f"{fleet_i.name}.ask_avoided"
        if low_demand and avoided in outputs:
            series.append(("Avoided demand", outputs[avoided] * 1e-12))
        return series

    markets = {f.name: market_series(f) for f in fleet.fleets}
    fleet_top = {
        market: max(sum(s for _, s in series).max(), 1e-9)
        for market, series in markets.items()
    }
    # pathways absent from the scenario (e.g. hydrogen in drop-in runs) are skipped
    energy_panels = {
        panel: [p for p in pathways if f"{p}.production" in outputs]
        for panel, pathways in ENERGY_PANELS.items()
    }
    energy_top = {
        panel: max(
            sum(outputs[f"{p}.production"] * 1e-12 for p in pathways).max()
            if pathways
            else 0.0,
            1e-9,
        )
        for panel, pathways in energy_panels.items()
    }

    def color(aircraft_name):
        if aircraft_name == "Avoided demand":
            return "tan"
        return next(
            (c for prop, c in propulsion_colors.items() if prop in aircraft_name),
            "black",
        )

    def draw(k):
        for ax in axes.flat:
            ax.clear()
        for col, (market, series) in enumerate(markets.items()):
            ax = axes[0, col]
            ax.stackplot(
                years,
                [s[k] for _, s in series],
                colors=[color(n) for n, _ in series],
                labels=[
                    n.replace(f"_{market}", "").replace("_fleet", " fleet")
                    for n, _ in series
                ],
            )
            ax.set_xlim(years[0], years[-1])
            ax.set_ylim(0.0, 1.05 * fleet_top[market])
            ax.set_title(market.replace("_", " "))
            ax.legend(loc="upper left", fontsize="x-small", framealpha=0.5)
        axes[0, 0].set_ylabel("ASK [trillion seat km]")
        for col, (panel, pathways) in enumerate(energy_panels.items()):
            ax = axes[1, col]
            ax.set_title(panel)
            if not pathways:
                continue
            ax.stackplot(
                years,
                [outputs[f"{p}.production"][k] * 1e-12 for p in pathways],
                colors=[PATHWAY_COLORS[p] for p in pathways],
                labels=[PATHWAY_LABELS.get(p, p) for p in pathways],
            )
            ax.set_xlim(years[0], years[-1])
            ax.set_ylim(0.0, 1.05 * energy_top[panel])
            ax.set_title(panel)
            ax.legend(loc="upper left", fontsize="x-small", framealpha=0.5)
        axes[1, 0].set_ylabel("Production [EJ / year]")
        ax = axes[1, -1]
        ax.plot(iterations, 100.0 * co2, color="lightgray")
        ax.plot(iterations[: k + 1], 100.0 * co2[: k + 1], color="black")
        ax.plot(iterations[k], 100.0 * co2[k], "o", color="crimson")
        ax.axhline(100.0, color="gray", linestyle=":")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cumulative CO2 [% of budget]")
        ax.set_title("Optimization progress")
        fig.suptitle(
            f"{name}, update model: iteration {iterations[k]} / {iterations[-1]}",
            fontsize="x-large",
        )

    writer = FFMpegWriter(fps=FPS, bitrate=4000)
    with writer.saving(fig, folder / VIDEO_FILE, dpi=100):
        for k in range(len(iterations)):
            draw(k)
            writer.grab_frame()
        for _ in range(FPS * HOLD_SECONDS):
            writer.grab_frame()
    LOGGER.info("saved %s", folder / VIDEO_FILE)


if __name__ == "__main__":
    arguments = sys.argv[1:]
    timeline = "extended" if "--extended" in arguments else "paper"
    arguments = [argument for argument in arguments if argument != "--extended"]
    if arguments == ["list"]:
        print("\n".join(SINGLE))  # noqa: T201
        sys.exit()
    for name in arguments:
        folder = RESULTS_DIR / ("update-2100" if timeline == "extended" else "update")
        folder /= name
        scenario, _, _, fleet, low_demand = setup(name, timeline)
        outputs = evaluate_iterations(folder, scenario, fleet)
        render(folder, name, outputs, fleet, low_demand)
