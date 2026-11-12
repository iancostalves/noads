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

# ruff: noqa: RUF001, E501
"""Sections of the comparison document on the hydrogen aircraft and the GAM changes.

The numbers of the text are read from ``h2_design_study.json`` when the document is
built, so that the text follows the model.
"""

import h2_design_study as h2

from noads.application.base_objects import categories_mission
from noads.application.base_objects import category_conso
from noads.application.base_objects import fuelcell_engine_count
from noads.application.base_objects import propulsion_architectures
from noads.application.base_objects import propulsion_mission


def _year(value):
    if value is None:
        return "after 2060"
    if value <= h2.YEARS[0]:
        return f"≤ {h2.YEARS[0]}"
    return f"{value:.1f}"


def _eis(value):
    return "–" if value is None else f"{value:.1f}"


def _at(study, block, key, year, output):
    return study[block][key][str(year)][output]


def mission_rows():
    """Missions of the NOADS markets and of the ZEROe fuel cell concept."""
    rows = [
        [
            "Mission",
            "Seats",
            "Design range [km]",
            "Cruise: fuel cell / gas turbine",
            "Propulsors: fuel cell / gas turbine",
        ]
    ]
    fc, gt = propulsion_mission["lH2-FuelCell"], propulsion_mission["lH2-GasTurbine"]
    cruise = (
        f"{fc['speed'] * 3.6:.0f} km/h at {fc['altitude'] / 0.3048:,.0f} ft / "
        f"{gt['speed'] * 3.6:.0f} km/h at {gt['altitude'] / 0.3048:,.0f} ft"
    )
    for market, mission in categories_mission.items():
        rows.append([
            f"NOADS {market.replace('_', ' ')}",
            f"{mission['npax']}",
            f"{mission['range'] * 1e-3:,.0f}",
            cruise,
            (
                f"{fuelcell_engine_count[market]} / "
                f"{propulsion_architectures['lH2-GasTurbine']['engine_count']}"
            ),
        ])
    rows.append([
        "Airbus ZEROe fuel cell concept",
        "about 100",
        "about 1,850 (1,000 nmi)",
        "not used here",
        "4 electric propulsors of 2–2.5 MW",
    ])
    return rows


def feasibility_rows(study):
    """First EIS of the fuel cell aircraft satisfying each design constraint."""
    rows = [
        [
            "Fuel cell aircraft",
            "Power per propulsor ≤ maximum",
            "Energy per seat-km ≤ 2019 fleet",
            "Both (feasible)",
            "EIS in the update optimum",
            "EIS in the paper optimum",
        ]
    ]
    for market in h2.MARKETS:
        for tech in h2.TECHS:
            power, efficiency, both = h2.feasibility_years(study, tech, market)
            rows.append([
                f"{market.replace('_', ' ')} – {tech}",
                _year(power),
                _year(efficiency),
                _year(both),
                _eis(h2.optimum_eis("update", "lH2-FuelCell", market, tech)),
                _eis(h2.optimum_eis("paper", "lH2-FuelCell", market, tech)),
            ])
    return rows


def zeroe_rows(study, tech="Mid"):
    """Fuel cell designs on NOADS missions and on a ZEROe-like mission."""
    rows = [
        [
            f"Fuel cell aircraft ({tech})",
            "Power per propulsor 2040 [MW]",
            "Power per propulsor 2045 [MW]",
            "Total power 2045 [MW]",
            "MTOW 2045 [t]",
            "Energy per seat-km 2045 [MJ]",
            "First EIS within the power limit",
        ]
    ]
    for name in h2.ZEROE_CASES:
        key = f"{tech}|{name}"
        years, ratio = h2.series(study["zeroe"], key, "unit_power_ratio")
        rows.append([
            name,
            f"{_at(study, 'zeroe', key, 2040, 'max_power'):.1f}",
            f"{_at(study, 'zeroe', key, 2045, 'max_power'):.1f}",
            f"{_at(study, 'zeroe', key, 2045, 'total_power'):.1f}",
            f"{_at(study, 'zeroe', key, 2045, 'mtow'):.0f}",
            f"{_at(study, 'zeroe', key, 2045, 'energy_per_ask'):.2f}",
            _year(h2.crossing(years, ratio, 1.0)),
        ])
    return rows


STEP_COLUMNS = [
    ("JetA-GasTurbine", "commuter"),
    ("lH2-GasTurbine", "commuter"),
    ("lH2-FuelCell", "commuter"),
    ("JetA-GasTurbine", "regional"),
    ("lH2-GasTurbine", "regional"),
    ("lH2-FuelCell", "regional"),
]


def steps_rows(study):
    """Energy per seat-km along the steps from the paper model to the update."""
    short = {
        "JetA-GasTurbine": "Jet-A",
        "lH2-GasTurbine": "LH2 GT",
        "lH2-FuelCell": "LH2 FC",
    }
    rows = [
        [
            "Step (EIS 2045, Mid)",
            *[f"{m.capitalize()} {short[p]}" for p, m in STEP_COLUMNS],
        ]
    ]
    steps = list(study["steps"][f"{STEP_COLUMNS[0][0]}|{STEP_COLUMNS[0][1]}"])
    for step in steps:
        rows.append([
            step,
            *[
                f"{study['steps'][f'{p}|{m}'][step]['energy_per_ask']:.2f}"
                for p, m in STEP_COLUMNS
            ],
        ])
    rows.append([
        "2019 fleet of the market",
        *[f"{category_conso[m][1]:.2f}" for _, m in STEP_COLUMNS],
    ])
    return rows


def numbers(study):
    """Numbers quoted in the text."""
    n = {}
    for market in h2.MARKETS:
        for tech in h2.TECHS:
            n[f"years|{market}|{tech}"] = h2.feasibility_years(study, tech, market)
    for market in ("commuter", "regional"):
        for year in (2040, 2045):
            row = study["update"][f"Mid|lH2-FuelCell|{market}"][str(year)]
            n[f"fc|{market}|{year}"] = row
            n[f"gt|{market}|{year}"] = study["update"][f"Mid|lH2-GasTurbine|{market}"][
                str(year)
            ]
    for name in h2.ZEROE_CASES:
        key = f"Mid|{name}"
        years, ratio = h2.series(study["zeroe"], key, "unit_power_ratio")
        n[f"zeroe|{name}|first"] = h2.crossing(years, ratio, 1.0)
        _, energy = h2.series(study["zeroe"], key, "energy_per_ask")
        host = h2.ZEROE_CASES[name][0]
        n[f"zeroe|{name}|efficiency"] = h2.crossing(
            years, energy / category_conso[host][1], 1.0
        )
        for year in (2040, 2045):
            n[f"zeroe|{name}|{year}"] = study["zeroe"][key][str(year)]
    steps = study["steps"]
    for prop in h2.STEP_ARCHITECTURES:
        for market in h2.MARKETS:
            n[f"steps|{prop}|{market}"] = {
                k: v["energy_per_ask"] for k, v in steps[f"{prop}|{market}"].items()
            }
    n["limit|Mid|2040"] = study["update"]["Mid|lH2-FuelCell|regional"]["2040"][
        "max_unit_power"
    ]
    n["limit|Mid|2045"] = study["update"]["Mid|lH2-FuelCell|regional"]["2045"][
        "max_unit_power"
    ]
    return n


def hydrogen_section(study, p, bullets, table, figure, width, styles):
    """Part I subsection: hydrogen aircraft per market, fuel cell vs gas turbine."""
    n = numbers(study)
    zeroe = "ZEROe-like (100 seats, 1850 km, 4 prop.)"
    four = "Regional with 4 prop. (80 seats, 4500 km)"
    short = "80 seats, 1850 km, 2 prop."
    fc_c40, _fc_c45 = n["fc|commuter|2040"], n["fc|commuter|2045"]
    fc_r40, fc_r45 = n["fc|regional|2040"], n["fc|regional|2045"]
    gt_r40 = n["gt|regional|2040"]
    ze40, ze45 = n[f"zeroe|{zeroe}|2040"], n[f"zeroe|{zeroe}|2045"]
    sh45 = n[f"zeroe|{short}|2045"]
    fr45 = n[f"zeroe|{four}|2045"]
    _c_power, _c_eff, c_both = n["years|commuter|Mid"]
    r_power, r_eff, r_both = n["years|regional|Mid"]
    ru_both = n["years|regional|Upper"][2]
    lim40, lim45 = n["limit|Mid|2040"], n["limit|Mid|2045"]
    out = [
        p(
            "2b. Hydrogen aircraft: fuel cell versus gas turbine, per market",
            styles["H2"],
        ),
        p(
            "The update adds three design constraints to the fuel cell (and battery-electric) aircraft, "
            "applied at the EIS chosen by the optimizer: the design must close, its power per propulsor must not "
            "exceed the maximum power of an electric propulsor available at that EIS (Mid: "
            f"{lim40:.1f} MW in 2040, {lim45:.1f} MW in 2045, 4 MW in 2060), and its energy per seat-km must not "
            "exceed that of the 2019 fleet of its market. The LH2 gas turbine has no such constraint: turbofans of "
            "several MW are current technology, and it only has to close. The EIS of alternative aircraft is bounded "
            "below by 2040."
        ),
        p("Missions of the markets", styles["H3"]),
        p(
            "Each market is designed for one mission: the number of seats and the design range of its fleet, "
            "and a cruise speed and altitude that depend on the propulsion. The regional market of NOADS is "
            "80 seats over 4500 km; a 100-seat aircraft over about 1850 km, like the Airbus ZEROe fuel cell "
            "concept, has the seats of the regional and short-medium markets but the range of the commuter market.",
            styles["SMALL"],
        ),
        table(
            mission_rows(),
            [width * 0.25, width * 0.09, width * 0.17, width * 0.27, width * 0.22],
        ),
        p("Why the fuel cell enters late in the regional market", styles["H3"]),
    ]
    out += bullets([
        (
            f"<b>The power per propulsor binds first.</b> In 2040 (Mid), the regional fuel cell aircraft needs "
            f"{fc_r40['total_power']:.1f} MW, i.e. {fc_r40['max_power']:.1f} MW per propulsor with 2 propulsors, "
            f"{fc_r40['unit_power_ratio']:.1f} times the {lim40:.1f} MW available, for an MTOW of {fc_r40['mtow']:.0f} t. "
            f"The maximum power grows and the design gets lighter with time, and the power limit is met from "
            f"{_year(r_power)} only; the optimizer chooses {_eis(h2.optimum_eis('update', 'lH2-FuelCell', 'regional'))}, "
            "right at that limit."
        ),
        (
            f"<b>The design range makes the aircraft heavy.</b> Over 4500 km, the fuel cell system with its thermal "
            f"management weighs {fc_r45['fuel_cell_system_mass']:.0f} t of an MTOW of {fc_r45['mtow']:.0f} t in 2045 "
            "(Figure 1c), and every extra kilogram needs more power, hence more fuel cells (mass snowball). "
            f"The same 80 seats over 1850 km need {sh45['total_power']:.1f} MW instead of {fc_r45['total_power']:.1f} MW "
            f"in 2045. The efficiency constraint is met from {_year(r_eff)}: before that, the regional fuel cell "
            "aircraft consumes more than the 2019 regional fleet."
        ),
        (
            f"<b>Four propulsors are not enough.</b> With 4 propulsors, the regional aircraft of 2045 is within the power "
            f"limit ({fr45['max_power']:.1f} MW per propulsor) but still consumes {fr45['energy_per_ask']:.2f} MJ/seat-km, "
            f"more than the {category_conso['regional'][1]:.2f} MJ/seat-km of the 2019 regional fleet, so it would enter "
            f"from {_year(n[f'zeroe|{four}|efficiency'])} instead of {_year(r_both)}."
        ),
        (
            f"<b>A ZEROe-like aircraft is consistent with the model.</b> Designed by the update model on the ZEROe "
            f"mission (100 seats, 1850 km, 4 propulsors), it needs {ze45['total_power']:.1f} MW in 2045, i.e. "
            f"{ze45['max_power']:.1f} MW per propulsor, close to the 4 × 2–2.5 MW announced by Airbus, and fits within "
            f"the power limit from {_year(n[f'zeroe|{zeroe}|first'])} (Mid; {ze40['max_power']:.1f} MW per propulsor in 2040). "
            f"It consumes {ze45['energy_per_ask']:.2f} MJ/seat-km in 2045, better than the 2019 commuter fleet "
            f"({category_conso['commuter'][1]:.2f}) and close to the regional one ({category_conso['regional'][1]:.2f}). "
            "The model is therefore not pessimistic on fuel cells as such: the late regional entry comes from the "
            "definition of the regional market (80 seats over 4500 km), and the ZEROe concept is closer to what NOADS "
            "calls commuter in range."
        ),
        (
            f"<b>Commuter market.</b> In 2040 the commuter fuel cell aircraft needs {fc_c40['max_power']:.2f} MW per "
            f"propulsor ({fc_c40['unit_power_ratio']:.2f} times the limit) and {fc_c40['energy_per_ask']:.3f} MJ/seat-km "
            f"(2019 fleet: {category_conso['commuter'][1]:.3f}), so it is infeasible in 2040 and feasible from "
            f"{_year(c_both)}: the optimizer chooses {_eis(h2.optimum_eis('update', 'lH2-FuelCell', 'commuter'))}, with both constraints active. "
            "The previous version of this document sampled the designs at 2035, 2040, 2050 and 2060 only, which explains "
            "why the commuter fuel cell looked infeasible in 2040 while it is in the results from early 2041."
        ),
        (
            f"<b>The gas turbine is not limited.</b> The regional LH2 gas turbine of 2040 consumes "
            f"{gt_r40['energy_per_ask']:.2f} MJ/seat-km, more than the 2019 fleet, but it has no efficiency constraint and "
            "emits no CO2 in flight, so the optimizer brings it in from 2040 in the commuter and regional markets. "
            f"With Upper technology, the regional fuel cell aircraft is feasible from {_year(ru_both)}."
        ),
    ])
    no_tms = h2.design_update(
        "lH2-FuelCell", "regional", 1, 2045, fuelcell_tms_power_loss=0.0
    )
    out += [
        p(
            "<b>Sensitivity to the thermal management.</b> The parasitic power of the thermal management is the largest "
            "fuel cell penalty of the update (Appendix D). Without it at all, the regional fuel cell aircraft of 2045 (Mid) "
            f"would consume {no_tms['energy_per_ask']:.2f} MJ/seat-km, better than the 2019 fleet, but would still need "
            f"{no_tms['max_power']:.2f} MW per propulsor, above the {lim45:.2f} MW available: the late regional entry is due "
            "to the mission and the power limit, whatever this assumption."
        ),
        p(
            "First EIS of the fuel cell aircraft satisfying each constraint",
            styles["H3"],
        ),
        p(
            "Interpolated from designs every year (update model). 'after 2060' means never within the paper timeline. "
            "The EIS of the optimum is that of the breakthrough trend scenario (SSP2-2.6) of the same technology; '–' means "
            "the aircraft is not in that fleet (removed by the feasibility filter at the last EIS, 2060).",
            styles["SMALL"],
        ),
        table(
            feasibility_rows(study),
            [
                width * 0.2,
                width * 0.16,
                width * 0.17,
                width * 0.13,
                width * 0.17,
                width * 0.17,
            ],
            font=7,
        ),
        figure(
            "h2_designs_per_market.png",
            "Figure 1b – LH2 fuel cell (blue) and LH2 gas turbine (red) aircraft per market and EIS. Solid: the Mid design "
            "satisfies the constraints of the optimization; dotted: it does not; band: Lower to Upper; dashed with dots: "
            "main paper (GAM V2.0, Mid); star: EIS in the update optimum (SSP2-2.6 trend, Mid). Middle row: the black line is "
            "the maximum power of an electric propulsor (Mid, band Lower–Upper), which constrains the fuel cell only.",
        ),
        figure(
            "h2_mass_breakdown.png",
            "Figure 1c – Mass breakdown of the LH2 aircraft in 2045 and 2055 (update, Mid). The fuel cell system and its "
            "thermal management dominate the fuel cell aircraft of the regional and larger markets.",
        ),
        table(
            zeroe_rows(study),
            [
                width * 0.29,
                width * 0.11,
                width * 0.11,
                width * 0.11,
                width * 0.09,
                width * 0.12,
                width * 0.17,
            ],
            font=7,
        ),
        figure(
            "h2_zeroe_check_mid.png",
            "Figure 1d – Fuel cell aircraft designed by the update model (Mid) for the NOADS commuter and regional "
            "missions and for missions between them, among which one close to ZEROe. Shaded: power announced for ZEROe.",
        ),
    ]
    return out


def steps_appendix(study, p, bullets, table, figure, width, styles):
    """Appendix: the changes of the aircraft model and their effect on the designs."""
    n = numbers(study)

    def step(prop, market, name):
        return n[f"steps|{prop}|{market}"][name]

    paper, v3 = "Paper (GAM V2.0)", "GAM V3.0 (paper technology)"
    out = [
        p("Appendix D – Changes of the aircraft model, step by step", styles["H1"]),
        p(
            "To explain the results, the aircraft of the paper is turned into the aircraft of the update one group of "
            f"assumptions at a time, at EIS {h2.STEP_EIS}, Mid technology. Each step keeps the previous ones, so the effect of "
            "a step depends on the order; the order follows the model: the GAM version, the airframe, the thermal engines, "
            "the fuel cell, the LH2 tanks, the electric powertrain and the mission options. The maximum power per propulsor "
            "is a constraint, not a design input, and does not appear here."
        ),
    ]
    out += bullets([
        (
            f"<b>GAM V3.0 penalizes small thermal aircraft.</b> With the paper technology, GAM V3.0 more than doubles the "
            f"energy per seat-km of the commuter Jet-A aircraft ({step('JetA-GasTurbine', 'commuter', paper):.2f} → "
            f"{step('JetA-GasTurbine', 'commuter', v3):.2f} MJ/seat-km) and of the commuter LH2 gas turbine "
            f"({step('lH2-GasTurbine', 'commuter', paper):.2f} → {step('lH2-GasTurbine', 'commuter', v3):.2f}), and raises "
            f"the regional ones by 60–70 %: its refitted regressions and the efficiency loss of small turbofans (scale factor "
            "on the power per engine) make the new aircraft of the small markets close to, or worse than, the 2019 fleet, "
            "whereas the paper's were 1.6 to 5 times better than the 2019 fleet."
        ),
        (
            f"<b>GAM V3.0 alone favours the fuel cell.</b> With the paper technology, the fuel cell aircraft get more efficient "
            f"(regional {step('lH2-FuelCell', 'regional', paper):.2f} → {step('lH2-FuelCell', 'regional', v3):.2f}): the "
            "fuel cell penalty is not from GAM V3.0."
        ),
        (
            f"<b>The revised fuel cell technology is the main fuel cell penalty.</b> The lower system efficiency, the specific "
            "power without heat rejection plus a sized thermal management system, and above all the parasitic power of the "
            f"thermal management bring the regional fuel cell aircraft from {step('lH2-FuelCell', 'regional', '+ thermal engine efficiency'):.2f} "
            f"to {step('lH2-FuelCell', 'regional', '+ TMS parasitic power'):.2f} MJ/seat-km. These values come from the 2026 "
            "literature review of the update (cruise system efficiency of about 40 % today, thermal management parasitic "
            "power of 0.2–0.27 per unit of heat today and 0.10–0.13 for optimized designs)."
        ),
        (
            "<b>Other steps are smaller:</b> the structure mass factor (the paper assumed 22–30 % lighter structures in "
            "2040–2060, the update 10–14 %) adds 5–10 %, the LH2 tank size law 5–11 % on hydrogen aircraft, the thermal "
            "engine efficiency trend saves 15–20 % on thermal engines, and the electric powertrain scale laws save about 10 % on "
            "fuel cell aircraft."
        ),
    ])
    out += [
        table(steps_rows(study), [width * 0.31, *[width * 0.115] * 6], font=7),
        figure(
            "gam_update_steps.png",
            f"Figure D1 – Energy per seat-km of the new aircraft, from the paper model (left) to the update (right), "
            f"one group of assumptions at a time, EIS {h2.STEP_EIS}, Mid technology.",
        ),
        p(
            "<b>Effect on the results.</b> The higher energy per seat-km of the new Jet-A and LH2 gas turbine aircraft in the "
            "small markets (GAM V3.0) and of the fuel cell aircraft in every market (fuel cell technology), together with the "
            "power limit that delays the fuel cell aircraft, explain the higher emissions of the update in every scenario "
            "(Part II). This decomposition is made on the designs; the share of each step in the cumulative emissions would "
            "need one optimization per step."
        ),
    ]
    return out
