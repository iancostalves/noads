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

"""Figures comparing the results of the main paper and of the update."""

import json
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import comparison_data as cd  # noqa: E402
import design_sweep  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from noads.application.base_objects import initialize_base_objects  # noqa: E402
from noads.application.visualization import CO2_2019_MT  # noqa: E402
from noads.application.visualization import plot_fleet_mix_variants  # noqa: E402
from noads.application.visualization import propulsion_colors  # noqa: E402

FIGURES = Path(__file__).with_name("figures")
COLORS = {"paper": "dimgray", "update": "tab:blue"}
LABELS = {"paper": "Main paper (GAM V2.0)", "update": "Update (GAM V3.0)"}
DPI = 150


def _band(
    ax, model, base, years_key="year", series=lambda d: d["CO2"] * 1e-12 / CO2_2019_MT
):
    """Lower-Upper band and Mid line of a quantity, for one model."""
    data = {t: cd.load(model, base, t) for t in cd.TECHS}
    years = data["Mid"][years_key]
    values = {t: series(d) for t, d in data.items()}
    ax.fill_between(
        years,
        np.minimum(values["Lower"], values["Upper"]),
        np.maximum(values["Lower"], values["Upper"]),
        color=COLORS[model],
        alpha=0.2,
        linewidth=0,
    )
    ax.plot(
        years,
        values["Mid"],
        color=COLORS[model],
        linestyle="--" if model == "paper" else "-",
        label=LABELS[model],
    )


def design_figure():
    """Energy per ASK of the new aircraft versus EIS, per market (Mid technology)."""
    sweep = design_sweep.load()
    archs = {
        "JetA-GasTurbine-v1": "Jet-A v1",
        "lH2-GasTurbine": "LH2 gas turbine",
        "lH2-FuelCell": "LH2 fuel cell",
        "Battery-Electric": "Battery-electric",
    }
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.8), sharex=True, layout="constrained")
    for ax, market in zip(axes, cd.MARKETS, strict=True):
        for model, linestyle in (("paper", "--"), ("update", "-")):
            block = sweep[model]["Mid"]
            for key, label in archs.items():
                rows = block["designs"].get(f"{key}_{market}")
                if not rows:
                    continue
                eis = sorted(int(e) for e in rows)
                values = []
                for e in eis:
                    row = rows[str(e)]
                    feasible = (
                        np.isfinite(row["energy_per_ask"])
                        and row.get("closure_gap", 0.0) <= 1e-4
                        and (
                            ("Electric" not in key and "FuelCell" not in key)
                            or row.get("unit_power_ratio", 0.0) <= 1.0
                        )
                    )
                    values.append(row["energy_per_ask"] if feasible else np.nan)
                ax.plot(
                    eis,
                    values,
                    linestyle=linestyle,
                    marker="o",
                    markersize=3,
                    color=propulsion_colors.get(
                        key if key != "JetA-GasTurbine-v1" else "JetA-GasTurbine-v1"
                    ),
                    label=f"{label} ({'paper' if model == 'paper' else 'update'})",
                )
        ax.axhline(
            sweep["update"]["Mid"]["fleet_energy"][market],
            color="gray",
            linestyle=":",
        )
        ax.set_yscale("log")
        ax.set_title(market.replace("_", " "))
        ax.set_xlabel("Entry into service")
    axes[0].set_ylabel("Energy per ASK [MJ / seat-km]")
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes[1:]:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="outside lower center", ncols=4, fontsize="small")
    fig.suptitle(
        "New aircraft, Mid technology: dashed = main paper, solid = update; "
        "dotted grey = 2019 fleet"
    )
    fig.savefig(FIGURES / "design_energy_per_ask.png", dpi=DPI)
    plt.close(fig)


def baseline_figure():
    """Annual emissions of the baseline scenarios, paper versus update."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True, layout="constrained")
    for ax, (family, base) in zip(
        axes,
        [
            ("SSP1-1.9", "SSP1-19-Fossil"),
            ("SSP2-2.6", "SSP2-26-Fossil"),
            ("SSP5-4.5", "SSP5-45-Fossil"),
        ],
        strict=True,
    ):
        for model in ("paper", "update"):
            _band(ax, model, base)
        ax.axhline(1.0, color="k", linewidth=0.6, linestyle=":")
        ax.set_title(f"Baseline {family}")
        ax.set_xlabel("Year")
    axes[0].set_ylabel("CO2 emissions / 2019")
    axes[0].legend()
    fig.suptitle("Baseline scenarios (fossil Jet-A): line = Mid, band = Lower to Upper")
    fig.savefig(FIGURES / "baseline_trends.png", dpi=DPI)
    plt.close(fig)


def mitigation_figure():
    """Annual emissions of the mitigation scenarios, paper versus update."""
    rows = [
        (
            "Drop-in",
            [
                "SSP2-26-DropIn",
                "SSP2-26-DropIn-Availability",
                "SSP2-26-DropIn-LowDemand",
            ],
        ),
        ("Breakthrough", ["SSP2-26", "SSP2-26-Availability", "SSP2-26-LowDemand"]),
    ]
    fig, axes = plt.subplots(
        2, 3, figsize=(14, 7), sharex=True, sharey=True, layout="constrained"
    )
    for r, (family, bases) in enumerate(rows):
        for c, (base, name) in enumerate(
            zip(bases, ["trend", "availability", "low-demand"], strict=True)
        ):
            ax = axes[r, c]
            for model in ("paper", "update"):
                _band(ax, model, base)
            ax.axhline(1.0, color="k", linewidth=0.6, linestyle=":")
            ax.set_title(f"{family}, {name}")
            if r == 1:
                ax.set_xlabel("Year")
    axes[0, 0].set_ylabel("CO2 emissions / 2019")
    axes[1, 0].set_ylabel("CO2 emissions / 2019")
    axes[0, 0].legend()
    fig.suptitle("Mitigation scenarios, SSP2-2.6: line = Mid, band = Lower to Upper")
    fig.savefig(FIGURES / "mitigation_trends.png", dpi=DPI)
    plt.close(fig)


def demand_figure():
    """Traffic of the low-demand variants relative to the trend."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True, layout="constrained")
    for ax, (family, base) in zip(
        axes,
        [
            ("Drop-in low-demand", "SSP2-26-DropIn-LowDemand"),
            ("Breakthrough low-demand", "SSP2-26-LowDemand"),
        ],
        strict=True,
    ):
        for model in ("paper", "update"):
            _band(
                ax, model, base, series=lambda d: d["rpk"] / (cd.RPK_2019_TRIL * 1e12)
            )
        trend = cd.load("paper", base, "Mid")
        ax.plot(
            trend["year"],
            trend["rpk_trend"] / (cd.RPK_2019_TRIL * 1e12),
            color="k",
            linestyle=":",
            label="Trend demand",
        )
        ax.set_title(family)
        ax.set_xlabel("Year")
    axes[0].set_ylabel("RPK / 2019")
    axes[0].legend()
    fig.savefig(FIGURES / "low_demand_traffic.png", dpi=DPI)
    plt.close(fig)


def shares_figure():
    """Fleet mix of 2075 per architecture, paper versus update."""
    table = cd.table()
    labels = list(cd.ARCHITECTURES)
    colors = {
        "Current fleet": "dimgray",
        "Jet-A v1": "maroon",
        "Jet-A v2": "y",
        "LH2 gas turbine": "orangered",
        "LH2 fuel cell": "royalblue",
        "Battery-electric": "limegreen",
    }
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True, layout="constrained")
    for ax, variant in zip(axes, ["trend", "availability", "low-demand"], strict=True):
        ticks, tick_labels = [], []
        for t, tech in enumerate(cd.TECHS):
            for m, model in enumerate(("paper", "update")):
                x = t * 3 + m
                bottom = 0.0
                m_ = table[model, "Breakthrough", variant, tech]
                for label in labels:
                    height = m_[f"share_{label}"]
                    ax.bar(
                        x,
                        height,
                        bottom=bottom,
                        color=colors[label],
                        label=label if (t, m) == (0, 0) else None,
                        edgecolor="white",
                        linewidth=0.4,
                    )
                    bottom += height
                ticks.append(x)
                tick_labels.append(
                    f"{tech}\n{'paper' if model == 'paper' else 'update'}"
                )
        ax.set_xticks(ticks, tick_labels, fontsize=8)
        ax.set_title(f"Breakthrough {variant}")
        ax.set_ylim(0.0, 100.0)
    axes[0].set_ylabel("Share of the supplied ASK in 2075 [%]")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside lower center", ncols=6, fontsize="small")
    fig.savefig(FIGURES / "fleet_shares_2075.png", dpi=DPI)
    plt.close(fig)


def fleet_mix_figures():
    """Fleet mix per market, Mid technology, for both models."""
    for model in ("paper", "update"):
        _, fleet = initialize_base_objects(technology_index=1, aircraft_model=model)
        variants = {
            "Breakthrough trend": cd.load(model, "SSP2-26", "Mid"),
            "Breakthrough availability": cd.load(model, "SSP2-26-Availability", "Mid"),
            "Breakthrough low-demand": cd.load(model, "SSP2-26-LowDemand", "Mid"),
        }
        fig = plot_fleet_mix_variants(
            variants,
            fleet,
            title=f"Fleet mix, Mid technology: {LABELS[model]}",
            save_fig=True,
            directory_filename=str(FIGURES / f"fleet_mix_mid_{model}.png"),
        )
        plt.close(fig)


def energy_figure():
    """Supply per final energy carrier, Mid technology, Breakthrough variants."""
    fig, axes = plt.subplots(
        3, 2, figsize=(12, 9), sharex=True, sharey=True, layout="constrained"
    )
    series = [
        ("Fossil.production", "Fossil Jet-A", "dimgray"),
        ("Biofuel.production", "Biofuel Jet-A", "olivedrab"),
        ("Electrofuel.production", "E-fuel Jet-A", "darkorange"),
        ("LIQUID-H2.production", "Liquid hydrogen", "royalblue"),
        ("BATTERY.production", "Battery electricity", "limegreen"),
    ]
    for r, (base, name) in enumerate([
        ("SSP2-26", "trend"),
        ("SSP2-26-Availability", "availability"),
        ("SSP2-26-LowDemand", "low-demand"),
    ]):
        for c, model in enumerate(("paper", "update")):
            ax = axes[r, c]
            d = cd.load(model, base, "Mid")
            ax.stackplot(
                d["year"],
                [d[k] * 1e-12 for k, _, _ in series],
                colors=[col for _, _, col in series],
                labels=[lab for _, lab, _ in series],
            )
            if r == 0:
                ax.set_title(LABELS[model])
            if c == 0:
                ax.set_ylabel(f"Breakthrough {name}\nEnergy [EJ / year]")
            if r == 2:
                ax.set_xlabel("Year")
    axes[0, 0].legend(loc="upper right", fontsize="small")
    fig.savefig(FIGURES / "energy_supply_mid.png", dpi=DPI)
    plt.close(fig)


def robust_figure():
    """Emissions of the scenario-robust policies, paper versus update."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9), sharey=True, layout="constrained")
    for ax, (name, title) in zip(
        axes,
        [
            ("robust-SSP2-midTech", "Robust trend"),
            ("robust-SSP2-lowdemand-LowDemand-midTech", "Robust low-demand"),
        ],
        strict=True,
    ):
        for model in ("paper", "update"):
            folder = cd.RESULTS / ("update" if model == "update" else "") / name
            with (folder / "opt_result.json").open() as file:
                out = {k: np.array(v) for k, v in json.load(file)["outputs"].items()}
            for scenario, style in (
                ("SSP2-19", ":"),
                ("SSP2-26", "-"),
                ("SSP2-34", "--"),
            ):
                ax.plot(
                    np.linspace(2025.0, 2075.0, out[f"{scenario}.CO2"].size),
                    out[f"{scenario}.CO2"] * 1e-12 / CO2_2019_MT,
                    color=COLORS[model],
                    linestyle=style,
                    label=f"{LABELS[model].split(' (')[0]} {scenario}",
                )
        ax.axhline(1.0, color="k", linewidth=0.6, linestyle=":")
        ax.set_title(title)
        ax.set_xlabel("Year")
    axes[0].set_ylabel("CO2 emissions / 2019")
    axes[1].legend(fontsize="x-small", ncols=2)
    fig.savefig(FIGURES / "robust_trends.png", dpi=DPI)
    plt.close(fig)


def all_figures():
    """Generate all the figures of the comparison."""
    FIGURES.mkdir(exist_ok=True)
    for function in (
        design_figure,
        baseline_figure,
        mitigation_figure,
        demand_figure,
        shares_figure,
        fleet_mix_figures,
        energy_figure,
        robust_figure,
    ):
        function()
        print("done", function.__name__, flush=True)  # noqa: T201


if __name__ == "__main__":
    all_figures()
