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

# ruff: noqa: RUF001, RUF002, RUF003, E501
"""Comparison document (PDF) of the update of the aircraft model with the main paper.

Usage, from this directory: ``python build_pdf.py``. It regenerates the figures
(``comparison_figures.py``), the metrics (``results_summary.csv``) and the document
``update_vs_paper_comparison.pdf`` from the pre-computed optima of
``docs/examples/optimization/results`` (main paper) and ``.../results/update``.
"""

import csv
import json
from pathlib import Path

import comparison_data as cd
import comparison_figures as cf
import design_sweep
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image
from reportlab.platypus import KeepTogether
from reportlab.platypus import PageBreak
from reportlab.platypus import Paragraph
from reportlab.platypus import SimpleDocTemplate
from reportlab.platypus import Spacer
from reportlab.platypus import Table
from reportlab.platypus import TableStyle

from noads.application.base_objects import aircraft_tech_params

HERE = Path(__file__).parent
PDF = HERE / "update_vs_paper_comparison.pdf"
FIGURES = HERE / "figures"
BLUE = colors.HexColor("#1f5fa8")
LIGHT = colors.HexColor("#e8eef6")

styles = getSampleStyleSheet()
BODY = ParagraphStyle(
    "body",
    parent=styles["BodyText"],
    fontName="Helvetica",
    fontSize=9.3,
    leading=12.4,
    alignment=TA_JUSTIFY,
    spaceAfter=5,
)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8, leading=10, alignment=0)
CELL = ParagraphStyle(
    "cell", parent=BODY, fontSize=7.6, leading=9.2, alignment=0, spaceAfter=0
)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="Helvetica-Bold")
CAPTION = ParagraphStyle(
    "caption",
    parent=BODY,
    fontSize=8,
    leading=10,
    textColor=colors.HexColor("#333333"),
    alignment=0,
    spaceAfter=9,
)
H1 = ParagraphStyle(
    "h1",
    parent=styles["Heading1"],
    fontName="Helvetica-Bold",
    fontSize=16,
    leading=19,
    textColor=BLUE,
    spaceBefore=6,
    spaceAfter=8,
)
H2 = ParagraphStyle(
    "h2",
    parent=styles["Heading2"],
    fontName="Helvetica-Bold",
    fontSize=12,
    leading=15,
    textColor=BLUE,
    spaceBefore=9,
    spaceAfter=5,
)
H3 = ParagraphStyle(
    "h3",
    parent=styles["Heading3"],
    fontName="Helvetica-Bold",
    fontSize=10,
    leading=12.5,
    spaceBefore=6,
    spaceAfter=3,
)
TITLE = ParagraphStyle(
    "title",
    parent=styles["Title"],
    fontName="Helvetica-Bold",
    fontSize=22,
    leading=27,
    textColor=BLUE,
    alignment=0,
)
BULLET = ParagraphStyle(
    "bullet", parent=BODY, leftIndent=12, bulletIndent=2, spaceAfter=2.5
)
CODE = ParagraphStyle(
    "code",
    parent=SMALL,
    fontName="Courier",
    fontSize=7.4,
    leading=9,
    backColor=colors.HexColor("#f4f4f4"),
    leftIndent=4,
    rightIndent=4,
    spaceAfter=6,
)

WIDTH = A4[0] - 4 * cm


def p(text, style=BODY):
    return Paragraph(text, style)


def bullets(items):
    return [Paragraph(item, BULLET, bulletText="•") for item in items]


def figure(name, caption, width=WIDTH):
    image = Image(str(FIGURES / name))
    ratio = image.imageHeight / image.imageWidth
    image.drawWidth, image.drawHeight = width, width * ratio
    return KeepTogether([image, Spacer(1, 2), p(caption, CAPTION)])


def table(rows, widths, header=1, font=7.6, align_right_from=1, zebra=True):
    """A table of strings, with wrapped cells."""
    style_cell = ParagraphStyle("c", parent=CELL, fontSize=font, leading=font + 1.6)
    style_head = ParagraphStyle(
        "ch", parent=style_cell, fontName="Helvetica-Bold", textColor=colors.white
    )
    data = []
    for i, row in enumerate(rows):
        data.append([
            Paragraph(str(c), style_head if i < header else style_cell) for c in row
        ])
    t = Table(data, colWidths=widths, repeatRows=header)
    commands = [
        ("BACKGROUND", (0, 0), (-1, header - 1), BLUE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b8c4d6")),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]
    if zebra:
        for i in range(header, len(rows)):
            if (i - header) % 2:
                commands.append(("BACKGROUND", (0, i), (-1, i), LIGHT))
    t.setStyle(TableStyle(commands))
    return t


def fmt(a, b, nd=0, suffix=""):
    """'paper -> update' cell text."""

    def one(x):
        return (
            "-"
            if x is None or (isinstance(x, float) and not np.isfinite(x))
            else f"{x:.{nd}f}{suffix}"
        )

    return f"{one(a)} → {one(b)}"


def page_decor(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(
        2 * cm,
        1.2 * cm,
        "NOADS – Update of the aircraft model: comparison with the main paper",
    )
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"{doc.page}")
    canvas.restoreState()


def write_csv(table_):
    keys = sorted(next(iter(table_.values())))
    with (HERE / "results_summary.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "family", "variant", "technology", *keys])
        for (model, family, variant, tech), m in table_.items():
            writer.writerow([
                model,
                family,
                variant,
                tech,
                *[f"{m[k]:.5g}" for k in keys],
            ])


def tech_rows():
    """Technology parameters of the Mid scenario, paper versus update."""
    paper = {x.name: x for x in aircraft_tech_params(1, "paper")}
    update = {x.name: x for x in aircraft_tech_params(1, "update")}
    names = {
        "battery_specific_energy": ("Battery specific energy", "Wh/kg", 0),
        "emotor_specific_power": ("E-motor specific power (at 1 MW)", "kW/kg", 1),
        "electronics_specific_power": ("Power electronics specific power", "kW/kg", 1),
        "fuelcell_specific_power": (
            "Fuel cell specific power (update: without TMS)",
            "kW/kg",
            2,
        ),
        "fuelcell_efficiency": ("Fuel cell system efficiency at cruise", "%", 0),
        "struct_weight_factor": ("Structure mass factor (relative to today)", "%", 0),
    }
    rows = [
        [
            "Parameter (Mid scenario)",
            "Unit",
            "2040 paper → update",
            "2060 paper → update",
        ]
    ]
    for key, (label, unit, nd) in names.items():
        row = [label, unit]
        for eis in (2040, 2060):
            a = float(paper[key].value_at_entry_into_service(eis))
            b = float(update[key].value_at_entry_into_service(eis))
            row.append(fmt(a, b, nd))
        rows.append(row)
    extra = {
        "emotor_efficiency": ("E-motor efficiency (new)", "%", 1),
        "fuelcell_tms_heat_rejection": (
            "Fuel cell TMS specific heat rejection (new)",
            "kW/kg",
            1,
        ),
        "fuelcell_tms_power_loss": (
            "Fuel cell TMS parasitic power per heat (new)",
            "-",
            2,
        ),
        "max_unit_power": ("Maximum power per propulsor (new)", "MW", 1),
        "engine_efficiency_factor": ("Thermal engine efficiency factor (new)", "%", 0),
        "lh2tank_mass_factor": ("LH2 tank mass factor k (new)", "-", 2),
        "lh2_fuel_system_ratio": ("LH2 fuel system mass ratio s (new)", "-", 3),
    }
    for key, (label, unit, nd) in extra.items():
        row = [label, unit]
        for eis in (2040, 2060):
            row.append(
                f"→ {float(update[key].value_at_entry_into_service(eis)):.{nd}f}"
            )
        rows.append(row)
    return rows


def design_rows(sweep):
    """Energy per ASK of the new aircraft at 2040 and 2060 (Mid), paper versus update."""
    archs = [
        ("JetA-GasTurbine-v1", "Jet-A turbofan"),
        ("lH2-GasTurbine", "LH2 gas turbine"),
        ("lH2-FuelCell", "LH2 fuel cell"),
        ("Battery-Electric", "Battery-electric"),
    ]
    rows = [
        [
            "Market / architecture (Mid)",
            "2019 fleet",
            "2040 paper",
            "2040 update",
            "2060 paper",
            "2060 update",
        ]
    ]
    for market in cd.MARKETS:
        fleet = sweep["update"]["Mid"]["fleet_energy"][market]
        for key, label in archs:
            cells = []
            any_ = False
            for eis in ("2040", "2060"):
                for model in ("paper", "update"):
                    d = sweep[model]["Mid"]["designs"].get(f"{key}_{market}")
                    if not d:
                        cells.append("n/a")
                        continue
                    r = d[eis]
                    limited = (
                        "Electric" in key or "FuelCell" in key
                    )  # the power limit constrains these only
                    ok = (
                        np.isfinite(r["energy_per_ask"])
                        and r["energy_per_ask"] > 0
                        and r.get("closure_gap", 0.0) <= 1e-4
                        and (not limited or r.get("unit_power_ratio", 0.0) <= 1.0)
                    )
                    cells.append(f"{r['energy_per_ask']:.2f}" if ok else "infeasible")
                    any_ = any_ or ok
            if any(c != "n/a" for c in cells):
                rows.append([
                    f"{market.replace('_', ' ')} – {label}",
                    f"{fleet:.2f}",
                    *cells,
                ])
    return rows


def eis_rows():
    """Range of the EIS chosen by the optimizer per architecture (Mid)."""
    rows = [
        [
            "Mid technology, EIS range over the markets",
            "Trend: paper",
            "Trend: update",
            "Availability: paper",
            "Availability: update",
            "Low-demand: paper",
            "Low-demand: update",
        ]
    ]
    data = {
        (v, m): cd.load(m, base, "Mid")
        for v, base in (
            ("t", "SSP2-26"),
            ("a", "SSP2-26-Availability"),
            ("l", "SSP2-26-LowDemand"),
        )
        for m in ("paper", "update")
    }
    for label, key in list(cd.ARCHITECTURES.items())[1:]:
        row = [label]
        for v in "tal":
            for m in ("paper", "update"):
                d = data[v, m]
                values = [
                    float(d[k].ravel()[0])
                    for k in d
                    if k.startswith(f"constant.{key}_")
                    and k.endswith("entry_into_service")
                ]
                row.append(
                    f"{min(values):.0f}–{max(values):.0f} ({len(values)})"
                    if values
                    else "n/a"
                )
        rows.append(row)
    return rows


def headline_rows(t):
    rows = [
        [
            "Scenario (Mid technology)",
            "Cumulative CO2 [% budget]",
            "Peak emissions [x 2019]",
            "Peak year",
            "Emissions 2075 [x 2019]",
            "Avoided demand 2075 [%]",
            "Fossil Jet-A 2075 [EJ]",
        ]
    ]
    for (family, variant), _ in cd.SCENARIOS.items():
        pm = t["paper", family, variant, "Mid"]
        um = t["update", family, variant, "Mid"]
        rows.append([
            f"{family} {variant}".strip(),
            fmt(pm["cum_co2_pct"], um["cum_co2_pct"]),
            fmt(pm["peak_co2"], um["peak_co2"], 2),
            fmt(pm["peak_year"], um["peak_year"]),
            fmt(pm["co2_2075"], um["co2_2075"], 2),
            fmt(pm["rpk_avoided_2075"], um["rpk_avoided_2075"]),
            fmt(pm["fossil_2075"], um["fossil_2075"], 1),
        ])
    return rows


def full_rows(t):
    rows = [
        [
            "Scenario",
            "Tech",
            "Cum. CO2 [% budget]",
            "Peak [x 2019]",
            "Peak year",
            "2075 [x 2019]",
            "Avoided demand 2075 [%]",
            "Fossil 2075 [EJ]",
            "Biomass peak [EJ/yr]",
            "Electricity peak [EJ/yr]",
        ]
    ]
    for (family, variant), _ in cd.SCENARIOS.items():
        for tech in cd.TECHS:
            pm = t["paper", family, variant, tech]
            um = t["update", family, variant, tech]
            rows.append([
                f"{family} {variant}".strip(),
                tech,
                fmt(pm["cum_co2_pct"], um["cum_co2_pct"]),
                fmt(pm["peak_co2"], um["peak_co2"], 2),
                fmt(pm["peak_year"], um["peak_year"]),
                fmt(pm["co2_2075"], um["co2_2075"], 2),
                fmt(pm["rpk_avoided_2075"], um["rpk_avoided_2075"]),
                fmt(pm["fossil_2075"], um["fossil_2075"], 1),
                fmt(pm["biomass_peak"], um["biomass_peak"], 1),
                fmt(pm["electricity_peak"], um["electricity_peak"], 1),
            ])
    return rows


def share_rows(t):
    rows = [
        [
            "Share of supplied ASK in 2075 [%]",
            *[f"{v}: paper → update" for v in ("trend", "availability", "low-demand")],
        ]
    ]
    for tech in cd.TECHS:
        for label in cd.ARCHITECTURES:
            row = [f"{tech} – {label}"]
            for variant in ("trend", "availability", "low-demand"):
                a = t["paper", "Breakthrough", variant, tech][f"share_{label}"]
                b = t["update", "Breakthrough", variant, tech][f"share_{label}"]
                row.append(fmt(a, b, 0))
            rows.append(row)
    return rows


def robust_rows():
    rows = [
        [
            "Robust policy (Mid technology)",
            "Scenario",
            "Cumulative CO2 [% budget]",
            "Peak [x 2019]",
            "Emissions 2075 [x 2019]",
            "Avoided demand 2075 [%]",
        ]
    ]
    for name, title in (
        ("robust-SSP2-midTech", "trend"),
        ("robust-SSP2-lowdemand-LowDemand-midTech", "low-demand"),
    ):
        out = {}
        for model in ("paper", "update"):
            folder = cd.RESULTS / ("update" if model == "update" else "") / name
            with (folder / "opt_result.json").open() as file:
                out[model] = {
                    k: np.array(v) for k, v in json.load(file)["outputs"].items()
                }
        for scenario in ("SSP2-19", "SSP2-26", "SSP2-34"):
            vals = {}
            for model in ("paper", "update"):
                o = out[model]
                co2 = o[f"{scenario}.CO2"] * 1e-12 / cd.CO2_2019_MT
                vals[model] = (
                    100.0
                    * float(o[f"{scenario}.cumulative.CO2"].ravel()[-1])
                    / cd.BUDGET,
                    float(co2.max()),
                    float(co2[-1]),
                    100.0
                    * (
                        1.0
                        - float(
                            o[f"{scenario}.rpk"][-1] / o[f"{scenario}.rpk_trend"][-1]
                        )
                    ),
                )
            rows.append([
                f"Robust {title}",
                scenario,
                fmt(vals["paper"][0], vals["update"][0]),
                fmt(vals["paper"][1], vals["update"][1], 2),
                fmt(vals["paper"][2], vals["update"][2], 2),
                fmt(vals["paper"][3], vals["update"][3]),
            ])
    return rows


def story(t, sweep):
    """The document."""
    s = []
    # ---- numbers used in the text

    def delta(family, variant, tech="Mid"):
        a = t["paper", family, variant, tech]["cum_co2_pct"]
        b = t["update", family, variant, tech]["cum_co2_pct"]
        return a, b

    bt = delta("Breakthrough", "trend")
    ba = delta("Breakthrough", "availability")
    delta("Breakthrough", "low-demand")
    increases = [
        t["update", f, v, k]["cum_co2_pct"] - t["paper", f, v, k]["cum_co2_pct"]
        for (f, v) in cd.SCENARIOS
        for k in cd.TECHS
        if v != "low-demand"
    ]
    wide = f"{min(increases):.0f} to {max(increases):.0f}"

    s += [
        p("Update of the aircraft model versus the main paper", TITLE),
        p(
            "Comparison of the pre-computed optimization results, organised like the paper "
            "(Methods, then Results)",
            H3,
        ),
        p(
            "Cumulative CO<sub>2</sub> is expressed as a percentage of the 3 % share of the 2.0 °C "
            "(66 %) carbon budget allocated to aviation, as in the paper; 'paper → update' always reads "
            "main paper result → update result. All numbers are read from the committed optima, "
            "none was recomputed by hand.",
            SMALL,
        ),
        Spacer(1, 6),
        p("Summary", H1),
    ]
    s += [
        p(
            "The update model (<font face='Courier'>aircraft_model=\"update\"</font>) replaces GAM V2.0 by "
            "GAM V3.0 and adds powertrain scale laws with a maximum power per propulsor, a revised fuel cell "
            "system with thermal management, a size-dependent LH2 tank gravimetric index, an engine efficiency "
            "trend and four-point technology scenarios (2020, 2040, 2060, 2080). Energy pathways, background "
            "scenarios, demand model, controls and optimizer are those of the main paper, and the timeline is the "
            "paper's (2025–2075, entry into service up to 2060). Entry into service is bounded below by 2035 "
            "(Jet-A) and 2040 (alternative architectures) in the update, and the LH2 gas turbine and fuel cell "
            "start at the same date."
        )
    ]
    s += bullets([
        (
            f"<b>Emissions are higher in every scenario.</b> For the Mid technology breakthrough scenarios, the "
            f"cumulative emissions go from {bt[0]:.0f} to {bt[1]:.0f} % of the budget (trend), "
            f"{ba[0]:.0f} to {ba[1]:.0f} % (availability); the increase ranges from {wide} points of budget over "
            "all scenarios without demand reduction and technology levels."
        ),
        (
            f"<b>The budget is respected only with demand reduction.</b> The low-demand variants stay feasible in both "
            f"models (100 % of the budget) but need more avoided demand in the update "
            f"({t['paper', 'Breakthrough', 'low-demand', 'Mid']['rpk_avoided_2075']:.0f} → "
            f"{t['update', 'Breakthrough', 'low-demand', 'Mid']['rpk_avoided_2075']:.0f} % of the trend traffic in 2075)."
        ),
        (
            "<b>The ranking of the scenarios is unchanged</b>: baseline &gt; drop-in &gt; breakthrough, availability "
            "and low demand reduce emissions, higher technology reduces emissions."
        ),
        (
            "<b>The architectures change.</b> Fuel cell aircraft lose share (Mid technology: 57 → 29 % of the 2075 supplied ASK in "
            "the breakthrough trend scenario, 85 → 44 % with availability; 30–57 % in the paper's Lower scenario, none in the update). "
            "The LH2 gas turbine, the Jet-A new generation (v2) and the battery-electric aircraft gain share (see Part II, Fleet mix)."
        ),
        (
            "<b>Why:</b> the small-turbofan penalty of GAM V3.0 and its refitted regressions make the general, commuter "
            "and regional aircraft much less efficient than in the paper; the revised structure masses and fuel cell "
            "technology, the maximum power per propulsor and the later entry-into-service bounds further delay or "
            "remove the alternative architectures. No ablation study separates these contributions: the attribution "
            "is a reading of the design results (Part I), not a measured decomposition."
        ),
    ])

    # ---- Part I
    s += [
        PageBreak(),
        p("Part I – Models", H1),
        p(
            "The sections follow the paper's Methods (background scenarios, demand, fleet, energy, controls, optimization). "
            "For each, the table lists what is identical and what changes."
        ),
    ]
    s += [
        p("1. Background scenarios and demand", H2),
        p(
            "<b>Unchanged.</b> The AR6 SSP scenarios, the 2.0 °C carbon budget, the traffic demand (generalized "
            "logistic of GDP per capita) and the low-demand (demand aversion) formulation are those of the paper. "
            "Demand and its elasticity do not depend on the aircraft model, so the trend traffic is identical in "
            "both models; only the part of the demand that is supplied differs (Figure 8)."
        ),
    ]
    s += [
        p("2. Fleet: aircraft design and technology", H2),
        p(
            "<b>Changed – this is the update.</b> The fleet logic (sigmoid adoption, market shares, retirement of the "
            "current fleet) is unchanged. The aircraft design model changes as follows:"
        ),
    ]
    s += bullets([
        (
            "<b>GAM V3.0</b> replaces V2.0: refitted reference power and empty mass regressions, lift-to-drag "
            "regression distinguishing fan and propeller aircraft, refitted turboshaft consumption, and a scale factor for "
            "the efficiency loss of small turbofans."
        ),
        (
            "<b>Powertrain scale laws</b> (electric motor, power electronics, fuel cell) and a <b>maximum power per "
            "propulsor</b> available at entry into service. Battery-electric and fuel cell designs that cannot respect it "
            "are removed from a market (per-market feasibility filter), as the paper already does for electric aircraft."
        ),
        (
            "<b>Fuel cell</b>: system specific power without thermal management, plus a thermal management system "
            "(heat rejection per kilogram, parasitic power per heat), mission-averaged efficiency."
        ),
        "<b>LH2 tanks</b>: the gravimetric index depends on the tank size and on the technology scenario.",
        "<b>Thermal engine</b>: efficiency trend with entry into service.",
        (
            "<b>Technology scenarios</b>: four points (2020/2040/2060/2080) interpolated with PCHIP "
            "(<font face='Courier'>ScenarioTechParameter</font>) instead of three (2020/2040/2060)."
        ),
        (
            "<b>Entry into service</b> lower bounds: Jet-A 2035, alternatives 2040; the initial entry into service of the LH2 "
            "gas turbine and the LH2 fuel cell is the same (2040). Turboprops are not part of the optimized fleets."
        ),
        (
            "<b>Constraints</b> added to the optimization for electric and fuel cell aircraft: unit power ratio ≤ 1, design "
            "closure gap ≤ 1e-4, relative efficiency gain ≥ 1."
        ),
    ])
    s += [
        Spacer(1, 4),
        p("Technology parameters of the Mid scenario (paper → update)", H3),
        table(tech_rows(), [WIDTH * 0.46, WIDTH * 0.1, WIDTH * 0.22, WIDTH * 0.22]),
        Spacer(1, 6),
    ]
    s += [
        p("Energy per available seat-km of the new aircraft (Mid technology)", H3),
        p(
            "Energy per ASK of the new designs by entry into service (MJ per seat-km), compared with the 2019 fleet. "
            "'infeasible' means the design does not close or violates the power limit; 'n/a' means the architecture "
            "is not available in the market in that model.",
            SMALL,
        ),
        table(
            design_rows(sweep),
            [
                WIDTH * 0.34,
                WIDTH * 0.13,
                WIDTH * 0.13,
                WIDTH * 0.13,
                WIDTH * 0.13,
                WIDTH * 0.14,
            ],
        ),
        Spacer(1, 6),
        figure(
            "design_energy_per_ask.png",
            "Figure 1 – Energy per ASK of the new aircraft versus entry into service, per "
            "market, Mid technology. Dashed: main paper; solid: update; dotted grey: 2019 fleet. Discontinuities "
            "show designs that do not close or violate the power limit. In the paper, some battery-electric "
            "designs have unphysical (negative or very large) values at early dates and are masked by the paper's own filter.",
        ),
    ]
    s += [
        p(
            "Reading: the update Jet-A aircraft are 2 to 4 times more energy consuming than the paper's in the general and "
            "commuter markets (small turbofans), 40–60 % in the regional market, and 2–8 % in the short-medium and long-range "
            "markets, where it is even better than the paper at long range. Fuel cell aircraft lose their advantage "
            "(system efficiency, thermal management, power limit) and are unavailable for short-medium and long-range. "
            "The LH2 gas turbine is hardly affected in the large markets and is penalised in the small ones."
        )
    ]
    s += [
        p("3. Energy: carriers, pathways and primary energy", H2),
        p(
            "<b>Unchanged.</b> The energy carriers and production pathways (fossil, HEFA, ATJ, FT, electrolysis, gas reforming, "
            "power-to-liquid, hydrogen liquefaction, battery charging), their efficiencies and cost/resource impacts are "
            "those of the paper, in every technology scenario. The technology scenarios vary the aircraft technology only. "
            "<b>Added</b>: primary energy factors per carrier and per aircraft (documentation and figures only, no change in "
            "the optimization)."
        ),
    ]
    s += [
        p("4. Controls and design space", H2),
        p(
            "<b>Unchanged</b> controls (aircraft market penetration, energy mix per pathway and period, supply cap for low demand). "
            "<b>Changed</b> bounds of the entry-into-service controls (paper: 2035–2060, 2030 for the first Jet-A generation; update: see Fleet) and the corresponding "
            "initial values, and the removal of the market/architecture combinations that are infeasible."
        ),
    ]
    s += [
        p("5. Optimization problem", H2),
        p(
            "<b>Unchanged</b> formulations (trend: minimise the cumulative CO2; low demand: minimise the time-discounted burden of demand aversion under the cumulative emissions budget), constraints on the biomass and electricity fair shares, solver "
            "(SLSQP through GEMSEO, automatic differentiation with JAX), tolerances and the 2000 (single) and 5000 (robust) "
            "iteration caps. <b>Added</b>: the three design constraints above, the differentiable design closure gap, the "
            "implicit adjoint through the Newton sizing solver, and checkpoint/warm start for the robust runs. "
            "All 29 optima of the update (27 single, 2 robust) end feasible."
        ),
    ]

    # ---- Part II
    s += [
        p("Part II – Results", H1),
        p(
            "The structure follows the paper's Results: baseline, mitigation (drop-in and breakthrough), robust policies, "
            "and the comparison with the literature."
        ),
    ]
    s += [
        p("6. Baseline scenarios (no mitigation measure)", H2),
        p(
            "Fossil-only scenarios. Without any alternative fuel or aircraft, only the fleet renewal with conventional aircraft "
            "reduces energy consumption; emissions in the update are higher by 25–30 % because the new Jet-A aircraft are "
            "less efficient, and the peak is later (2060 instead of 2041 in SSP1)."
        ),
    ]
    ba_rows = [
        [
            "Baseline (Mid)",
            "Cumulative CO2 [% budget]",
            "Peak [x 2019]",
            "Peak year",
            "2075 [x 2019]",
        ]
    ]
    for fam in ("Baseline SSP1", "Baseline SSP2", "Baseline SSP5"):
        a, b = t["paper", fam, "", "Mid"], t["update", fam, "", "Mid"]
        ba_rows.append([
            fam,
            fmt(a["cum_co2_pct"], b["cum_co2_pct"]),
            fmt(a["peak_co2"], b["peak_co2"], 2),
            fmt(a["peak_year"], b["peak_year"]),
            fmt(a["co2_2075"], b["co2_2075"], 2),
        ])
    s += [
        table(
            ba_rows,
            [WIDTH * 0.24, WIDTH * 0.26, WIDTH * 0.17, WIDTH * 0.15, WIDTH * 0.18],
        ),
        Spacer(1, 6),
        figure(
            "baseline_trends.png",
            "Figure 2 – Annual CO2 emissions of the baseline scenarios, Lower to Upper technology "
            "(bands), Mid line. Grey dashed: main paper; blue: update.",
        ),
    ]
    s += [
        p("7. Mitigation scenarios", H2),
        p(
            "Headline metrics of the Mid technology optima (all variants). The update emits more in every case, and the "
            "ordering is preserved. Full results for the three technology levels are in Appendix C."
        ),
    ]
    s += [
        table(
            headline_rows(t),
            [
                WIDTH * 0.2,
                WIDTH * 0.15,
                WIDTH * 0.13,
                WIDTH * 0.11,
                WIDTH * 0.13,
                WIDTH * 0.14,
                WIDTH * 0.14,
            ],
        ),
        Spacer(1, 6),
        figure(
            "mitigation_trends.png",
            "Figure 3 – Annual CO2 emissions of the drop-in and breakthrough scenarios (trend, "
            "availability, low demand) for the paper and the update.",
        ),
        p(
            "<b>Drop-in</b> (only drop-in fuels, no alternative aircraft): the drop-in results are less affected by the aircraft "
            "design but still higher because the Jet-A new-generation aircraft are less efficient, so more biofuel and e-fuel "
            "are needed. <b>Breakthrough</b> (alternative architectures allowed): the update loses part of the benefit of "
            "hydrogen and electric aircraft, so the breakthrough advantage over drop-in is smaller "
            f"({t['update', 'Drop-in', 'trend', 'Mid']['cum_co2_pct']:.0f} vs {t['update', 'Breakthrough', 'trend', 'Mid']['cum_co2_pct']:.0f} % "
            f"in the update, {t['paper', 'Drop-in', 'trend', 'Mid']['cum_co2_pct']:.0f} vs {t['paper', 'Breakthrough', 'trend', 'Mid']['cum_co2_pct']:.0f} % in the paper)."
        ),
        figure(
            "low_demand_traffic.png",
            "Figure 4 – Traffic of the low-demand variants relative to the trend (demand aversion).",
        ),
    ]
    s += [
        p("Fleet mix", H3),
        table(share_rows(t), [WIDTH * 0.28, WIDTH * 0.24, WIDTH * 0.24, WIDTH * 0.24]),
        Spacer(1, 4),
        figure(
            "fleet_shares_2075.png",
            "Figure 5 – Share of the supplied ASK in 2075 per architecture, Breakthrough scenarios.",
        ),
        figure(
            "fleet_mix_mid_paper.png",
            "Figure 6 – Fleet mix per market, Mid technology, Breakthrough trend, main paper.",
        ),
        figure(
            "fleet_mix_mid_update.png",
            "Figure 7 – Same, update. Jet-A v2, LH2 gas turbine and battery-electric aircraft partly replace the fuel cell aircraft; "
            "the 2019 fleet is renewed in every market.",
        ),
        p(
            "Entry into service chosen by the optimizer, per architecture (range over the markets, number of markets with the "
            "architecture):",
            SMALL,
        ),
        table(eis_rows(), [WIDTH * 0.19, *[WIDTH * 0.135] * 6], font=7),
        Spacer(1, 6),
    ]
    s += [
        p("Energy supply", H3),
        p(
            "Energy carriers produced and resources consumed (Mid, breakthrough). The update needs more fossil Jet-A "
            "(Part II table) because of the lower efficiency of the new fleet, while the biomass and electricity peaks are about the same "
            "because they are capped at their fair share of the global production."
        ),
    ]
    s += [
        figure(
            "energy_supply_mid.png",
            "Figure 8 – Supply per final energy carrier, Mid technology, Breakthrough variants. "
            "Dashed: main paper; solid: update.",
        )
    ]
    s += [
        p("8. Robust policies", H2),
        p(
            "A single aircraft policy (the energy mix may adapt to each scenario) optimised on the mean objective over three SSP2 background scenarios (SSP2-19, SSP2-26, SSP2-34). In the update the trend policy no longer meets the budget in any "
            "of the three scenarios (it did not in the paper either), while the low-demand policy remains feasible."
        ),
    ]
    s += [
        table(
            robust_rows(),
            [
                WIDTH * 0.16,
                WIDTH * 0.12,
                WIDTH * 0.2,
                WIDTH * 0.17,
                WIDTH * 0.17,
                WIDTH * 0.18,
            ],
        ),
        Spacer(1, 6),
        figure(
            "robust_trends.png",
            "Figure 9 – Emissions of the robust policies in the three SSP2 variants.",
        ),
    ]
    s += [
        p("9. Comparison with the literature", H2),
        p(
            "The paper compares its results with the literature (docs/paper/results/literature_comparison.md). That comparison is not recomputed here: "
            "the update emits more than the paper in every scenario, so the paper's position relative to the literature is expected to "
            "shift toward higher emissions, which would need to be checked with the same comparison script."
        ),
    ]

    # ---- Discussion
    s += [PageBreak(), p("Discussion and limits", H1)]
    s += bullets([
        (
            "<b>Attribution.</b> The increase of emissions comes from the combination of the model changes. The design table shows "
            "where the new Jet-A aircraft lose efficiency (small markets, GAM V3 turbofan scale factor) and where the alternative "
            "architectures are lost (power limit, fuel cell system). The share of each change in the emissions was not measured by "
            "a switch-by-switch study; running one would require optimizations with each update enabled alone."
        ),
        (
            "<b>Paper unchanged.</b> The paper model is the default and is guarded by <font face='Courier'>tests/test_gam_paper_baseline.py</font>; "
            "the paper results quoted here are the committed paper optima."
        ),
        (
            "<b>Optimization.</b> Optima are local (SLSQP); all update optima are feasible, but they were obtained from the same "
            "initial controls modified for the new bounds, so the paper-versus-update differences include optimizer effects "
            "that are small compared with the differences reported."
        ),
        (
            "<b>Not in this document.</b> The extended timeline (entry into service to 2080, end year 2100) is on hold, "
            "and geological hydrogen is planned work."
        ),
    ])

    # ---- Appendices
    s += [
        p("Appendix A – Where to find the results", H1),
        p(
            "All paths are relative to the repository root (<font face='Courier'>iancostalves/noads</font>, branch "
            "<font face='Courier'>ccr-906e3641-noua7t</font>)."
        ),
    ]
    where = [
        ["What", "Where"],
        [
            "This document, its figures, scripts and tables",
            "docs/update/comparison/ (update_vs_paper_comparison.pdf, figures/, results_summary.csv, build_pdf.py, comparison_data.py, comparison_figures.py, design_sweep.py, design_sweep.json, PR_DESCRIPTION.md)",
        ],
        [
            "Main paper optima (29)",
            "docs/examples/optimization/results/&lt;name&gt;/opt_result.json",
        ],
        [
            "Update optima (29)",
            "docs/examples/optimization/results/update/&lt;name&gt;/opt_result.json",
        ],
        [
            "Names",
            "&lt;scenario&gt;-&lt;tech&gt; with tech = lowTech, midTech, upTech; scenarios SSP1-19-Fossil, SSP2-26-Fossil, SSP5-45-Fossil, SSP2-26-DropIn[-Availability|-LowDemand], SSP2-26[-Availability|-LowDemand]; robust-SSP2-midTech, robust-SSP2-lowdemand-LowDemand-midTech",
        ],
        [
            "Content of opt_result.json",
            "'inputs' (optimal controls and constants such as entry into service) and 'outputs' (time series 2025–2075: CO2, cumulative.CO2, fleet ask per aircraft, energy production/consumption per carrier and pathway, rpk, rpk_trend, ...)",
        ],
        [
            "Update fleet-mix figures",
            "docs/update/figures/fleet_mix_update_{lower,mid,upper}.png",
        ],
        [
            "Update model documentation",
            "docs/update/index.md, aircraft_design.md, energy_pathways.md",
        ],
        ["Paper results documentation", "docs/paper/results/ and docs/paper/models/"],
        [
            "Code of the update",
            "src/noads/gam_jax/models/gam_v3.py, src/noads/application/base_objects.py (parameters, feasibility filter), scenario_setup.py (entry-into-service bounds, constraints), examples.py (drivers, checkpoint)",
        ],
        [
            "Local only, not in Git",
            "update/&lt;name&gt;/iterations.npz (every optimizer iteration), update/*.mp4 (iteration videos)",
        ],
    ]
    s += [
        table(where, [WIDTH * 0.28, WIDTH * 0.72], zebra=True),
        Spacer(1, 6),
        p(
            "<b>How the metrics are computed</b> (<font face='Courier'>comparison_data.py</font>): cumulative CO2 is the last value of "
            "<font face='Courier'>cumulative.CO2</font> divided by 3 % of the 2.0 °C (66 %) budget; emissions are expressed relative to "
            "the 2019 value of 844.43 MtCO2; avoided demand is 1 − rpk/rpk_trend in 2075; architecture shares are the "
            "ASK of 2075 of the aircraft of each architecture over the total ASK of the market fleets."
        ),
    ]
    s += [
        p("Appendix B – Reproduction", H1),
        p(
            'pip install -e ".[test]" reportlab matplotlib<br/>'
            "cd docs/examples/optimization/results<br/>"
            "python precompute_update.py &lt;name&gt;&nbsp;&nbsp;# recompute an update optimum (see the script for the list)<br/>"
            "cd docs/update/comparison<br/>"
            "rm design_sweep.json&nbsp;&nbsp;# optional: recompute the design sweep<br/>"
            "python build_pdf.py&nbsp;&nbsp;# figures, results_summary.csv and this PDF",
            CODE,
        ),
    ]
    s += [
        PageBreak(),
        p("Appendix C – Full results, all technology levels", H1),
        p("Paper → update, for every single-scenario optimum.", SMALL),
        table(
            full_rows(t),
            [
                WIDTH * w
                for w in (0.15, 0.06, 0.12, 0.1, 0.09, 0.1, 0.1, 0.09, 0.09, 0.1)
            ],
            font=6.6,
        ),
    ]
    return s


def main():
    cf.all_figures()
    t = cd.table()
    write_csv(t)
    sweep = design_sweep.load()
    doc = SimpleDocTemplate(
        str(PDF),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=2 * cm,
        title="NOADS: update of the aircraft model versus the main paper",
        author="NOADS",
    )
    doc.build(story(t, sweep), onFirstPage=page_decor, onLaterPages=page_decor)
    print("saved", PDF)  # noqa: T201


if __name__ == "__main__":
    main()
