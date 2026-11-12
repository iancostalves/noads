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

"""Figures of the hydrogen aircraft study (``h2_design_study.py``)."""

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import h2_design_study as h2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from noads.application.base_objects import category_conso  # noqa: E402
from noads.application.visualization import propulsion_colors  # noqa: E402

DPI = 150
COLORS = {prop: propulsion_colors[prop] for prop in h2.ARCHITECTURES}
FIRST_EIS = h2.FIRST_ALTERNATIVE_EIS
ROWS = {
    "energy_per_ask": "Energy per seat-km\n[MJ / seat-km]",
    "max_power": "Power per propulsor\n[MW]",
    "mtow": "MTOW [t]",
}


def _feasible_mask(study, tech, prop, market):
    """Where a design satisfies the constraints of the optimization."""
    key = f"{tech}|{prop}|{market}"
    years, gap = h2.series(study["update"], key, "closure_gap")
    ok = gap <= 1.0e-4
    if prop == "lH2-FuelCell":
        _, ratio = h2.series(study["update"], key, "unit_power_ratio")
        _, energy = h2.series(study["update"], key, "energy_per_ask")
        fleet = category_conso[market][h2.TECHS[tech]]
        ok &= (ratio <= 1.0) & (energy <= fleet)
    return years, ok


def designs_figure(study):
    """Energy per seat-km, power per propulsor and MTOW per market, FC vs GT."""
    fig, axes = plt.subplots(3, 5, figsize=(17, 9.5), sharex=True, layout="constrained")
    for col, market in enumerate(h2.MARKETS):
        for row, output in enumerate(ROWS):
            ax = axes[row, col]
            visible = []  # Mid and paper values from 2040, to scale the axis
            ax.axvspan(2035, h2.FIRST_ALTERNATIVE_EIS, color="0.92", zorder=0)
            for prop, label in h2.ARCHITECTURES.items():
                color = COLORS[prop]
                lows, highs = [], []
                for tech in ("Lower", "Upper"):
                    years, values = h2.series(
                        study["update"], f"{tech}|{prop}|{market}", output
                    )
                    (lows if tech == "Lower" else highs).append(values)
                ax.fill_between(
                    years,
                    np.minimum(lows[0], highs[0]),
                    np.maximum(lows[0], highs[0]),
                    color=color,
                    alpha=0.13,
                    linewidth=0,
                )
                years, values = h2.series(
                    study["update"], f"Mid|{prop}|{market}", output
                )
                _, ok = _feasible_mask(study, "Mid", prop, market)
                ax.plot(years, values, color=color, linestyle=":", linewidth=1.2)
                visible.extend(values[years >= FIRST_EIS])
                ax.plot(
                    years,
                    np.where(ok, values, np.nan),
                    color=color,
                    linewidth=2,
                    label=f"{label}, update (Mid; band Lower-Upper)",
                )
                years_p, values_p = h2.series(
                    study["paper"], f"Mid|{prop}|{market}", output
                )
                ax.plot(
                    years_p,
                    values_p,
                    color=color,
                    linestyle="--",
                    marker="o",
                    markersize=3,
                    alpha=0.6,
                    label=f"{label}, main paper (Mid)",
                )
                visible.extend(values_p[years_p >= FIRST_EIS])
                if output == "energy_per_ask":
                    eis = h2.optimum_eis("update", prop, market)
                    if eis is not None:
                        y = np.interp(eis, years, values)
                        ax.plot(
                            eis,
                            y,
                            marker="*",
                            markersize=13,
                            color=color,
                            markeredgecolor="black",
                            zorder=5,
                        )
            if output == "energy_per_ask":
                ax.axhline(
                    category_conso[market][1],
                    color="gray",
                    linestyle="-.",
                    linewidth=1,
                    label="2019 fleet (Mid)",
                )
                visible.append(category_conso[market][1])
            if output == "max_power":
                limits = {
                    tech: h2.series(
                        study["update"],
                        f"{tech}|lH2-FuelCell|{market}",
                        "max_unit_power",
                    )
                    for tech in h2.TECHS
                }
                ax.fill_between(
                    limits["Mid"][0],
                    limits["Lower"][1],
                    limits["Upper"][1],
                    color="black",
                    alpha=0.07,
                    linewidth=0,
                )
                ax.plot(
                    *limits["Mid"],
                    color="black",
                    linewidth=1.2,
                    label="Max. power per electric propulsor (Mid; band L-U)",
                )
                visible.extend(limits["Mid"][1])
            visible = np.array(visible, dtype=float)
            visible = visible[np.isfinite(visible)]
            ax.set_yscale("log")
            ax.set_ylim(0.8 * visible.min(), 1.25 * visible.max())
            if row == 0:
                ax.set_title(market.replace("_", " "), fontsize="large")
            if col == 0:
                ax.set_ylabel(ROWS[output])
            if row == 2:
                ax.set_xlabel("Entry into service")
            ax.set_xlim(2035, 2060)
    handles, labels = [], []
    for ax in axes[:, 0]:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="outside lower center", ncols=3, fontsize="small")
    fig.suptitle(
        "LH2 fuel cell vs LH2 gas turbine per market. Solid: design satisfies the "
        "optimization constraints (Mid); dotted: it does not; star: EIS of the "
        "update optimum (SSP2-2.6, Mid); grey area: before the earliest EIS (2040)",
        fontsize="medium",
    )
    fig.savefig(h2.FIGURES / "h2_designs_per_market.png", dpi=DPI)
    plt.close(fig)


MASS_PARTS = {
    "Airframe and systems": (
        "owe",
        "-propulsion_mass",
        "-fuel_cell_system_mass",
        "-energy_storage_mass",
    ),
    "Engines or motors": ("propulsion_mass",),
    "Fuel cell system and TMS": ("fuel_cell_system_mass",),
    "LH2 tanks and fuel system": ("energy_storage_mass",),
    "LH2 fuel": ("total_fuel",),
    "Payload": ("payload",),
}
MASS_COLORS = ["0.55", "maroon", "royalblue", "lightskyblue", "deepskyblue", "khaki"]


def mass_figure(study, years=(2045, 2055)):
    """Mass breakdown of the fuel cell and gas turbine aircraft (Mid)."""
    fig, axes = plt.subplots(1, 5, figsize=(17, 4.2), layout="constrained")
    for ax, market in zip(axes, h2.MARKETS, strict=True):
        labels, bottoms = [], None
        bars = []
        for year in years:
            for prop in h2.ARCHITECTURES:
                row = study["update"][f"Mid|{prop}|{market}"][str(year)]
                bars.append([
                    sum(
                        -(row[k[1:]] or 0.0) if k.startswith("-") else (row[k] or 0.0)
                        for k in keys
                    )
                    for keys in MASS_PARTS.values()
                ])
                labels.append(f"{'FC' if 'Fuel' in prop else 'GT'}\n{year}")
        bars = np.array(bars)
        bottoms = np.zeros(len(bars))
        for j, (part, color) in enumerate(zip(MASS_PARTS, MASS_COLORS, strict=True)):
            ax.bar(
                range(len(bars)),
                bars[:, j],
                bottom=bottoms,
                color=color,
                label=part,
                edgecolor="white",
                linewidth=0.5,
            )
            bottoms += bars[:, j]
        ax.set_xticks(range(len(bars)), labels, fontsize="small")
        ax.set_title(market.replace("_", " "))
    axes[0].set_ylabel("Mass [t] (stacked to MTOW)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=6, fontsize="small")
    fig.suptitle(
        "Mass breakdown of the LH2 aircraft, update model, Mid technology "
        "(FC: fuel cell, GT: gas turbine)",
        fontsize="medium",
    )
    fig.savefig(h2.FIGURES / "h2_mass_breakdown.png", dpi=DPI)
    plt.close(fig)


ZEROE_STYLES = ["royalblue", "navy", "mediumpurple", "teal", "crimson"]


def zeroe_figure(study, tech="Mid"):
    """Fuel cell aircraft on NOADS missions and on a ZEROe-like mission."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), layout="constrained")
    for name, color in zip(h2.ZEROE_CASES, ZEROE_STYLES, strict=True):
        key = f"{tech}|{name}"
        years, power = h2.series(study["zeroe"], key, "max_power")
        _, total = h2.series(study["zeroe"], key, "total_power")
        _, energy = h2.series(study["zeroe"], key, "energy_per_ask")
        style = {
            "color": color,
            "linewidth": 2.2 if "ZEROe" in name else 1.5,
            "linestyle": "-" if "NOADS" in name or "ZEROe" in name else "--",
        }
        axes[0].plot(years, power, label=name, **style)
        axes[1].plot(years, total, label=name, **style)
        axes[2].plot(years, energy, label=name, **style)
    years, limit = h2.series(
        study["zeroe"], f"{tech}|{next(iter(h2.ZEROE_CASES))}", "max_unit_power"
    )
    axes[0].plot(
        years,
        limit,
        color="black",
        linewidth=1.2,
        label=f"Max. power per propulsor ({tech})",
    )
    axes[0].axhspan(
        2.0,
        2.5,
        color="crimson",
        alpha=0.12,
        label="ZEROe announced: 2-2.5 MW per propulsor",
    )
    axes[1].axhspan(
        8.0, 10.0, color="crimson", alpha=0.12, label="ZEROe announced: 4 x 2-2.5 MW"
    )
    for market, style in (("commuter", ":"), ("regional", "-.")):
        axes[2].axhline(
            category_conso[market][h2.TECHS[tech]],
            color="gray",
            linestyle=style,
            label=f"2019 {market} fleet",
        )
    for ax, title in zip(
        axes,
        (
            "Power per propulsor [MW]",
            "Total power [MW]",
            "Energy per seat-km [MJ / seat-km]",
        ),
        strict=True,
    ):
        ax.set_title(title)
        ax.set_xlabel("Entry into service")
        ax.axvspan(2035, h2.FIRST_ALTERNATIVE_EIS, color="0.92", zorder=0)
        ax.set_xlim(2035, 2060)
        ax.set_yscale("log")
    handles, labels = [], []
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="outside lower center", ncols=4, fontsize="small")
    fig.suptitle(
        f"LH2 fuel cell aircraft, update model, {tech} technology: NOADS "
        "markets versus a ZEROe-like mission",
        fontsize="medium",
    )
    fig.savefig(h2.FIGURES / f"h2_zeroe_check_{tech.lower()}.png", dpi=DPI)
    plt.close(fig)


STEP_COLORS = {"JetA-GasTurbine": "maroon", **COLORS}


def steps_figure(study):
    """Energy per seat-km from the paper model to the update model, step by step."""
    fig, axes = plt.subplots(1, 5, figsize=(17, 5.6), sharex=True, layout="constrained")
    steps = list(next(iter(study["steps"].values())))
    for ax, market in zip(axes, h2.MARKETS, strict=True):
        for prop, label in h2.STEP_ARCHITECTURES.items():
            rows = study["steps"][f"{prop}|{market}"]
            values = [rows[s]["energy_per_ask"] for s in steps]
            ax.plot(
                range(len(steps)),
                values,
                marker="o",
                color=STEP_COLORS[prop],
                label=label,
            )
        ax.axhline(
            category_conso[market][1],
            color="gray",
            linestyle="-.",
            linewidth=1,
            label="2019 fleet (Mid)",
        )
        ax.set_xticks(range(len(steps)), steps, rotation=90, fontsize="x-small")
        ax.set_title(market.replace("_", " "))
        ax.set_yscale("log")
    axes[0].set_ylabel("Energy per seat-km [MJ / seat-km]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=4, fontsize="small")
    fig.suptitle(
        f"From the paper aircraft model to the update, step by step (EIS "
        f"{h2.STEP_EIS}, Mid technology; each step adds one group of update "
        "assumptions to the previous one)",
        fontsize="medium",
    )
    fig.savefig(h2.FIGURES / "gam_update_steps.png", dpi=DPI)
    plt.close(fig)


def all_figures():
    """Generate the figures of the study."""
    study = h2.load()
    designs_figure(study)
    mass_figure(study)
    for tech in h2.TECHS:
        zeroe_figure(study, tech)
    steps_figure(study)


if __name__ == "__main__":
    all_figures()
