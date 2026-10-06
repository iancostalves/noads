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
Prospective aircraft design with the update aircraft model
==========================================================

Same analysis as the prospective aircraft of the paper, with the update aircraft
model (``aircraft_model="update"``): GAM V3.0, powertrain scale effects with a
maximum power per propulsor, and LH2 tanks whose gravimetric index depends on
their size and on the entry-into-service. Turboprops are added to the comparison,
since GAM V3.0 changed how their efficiency varies with size. The technology
parameters and data of the main paper are shown in grey, those of the update in
blue.
"""

from pathlib import Path

from jax import jit
from jax import vmap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.pyplot import subplots
from numpy import asarray
from numpy import fmax
from numpy import fmin
from numpy import geomspace
from numpy import isfinite
from numpy import linspace
from numpy import nan
from numpy import vstack
from numpy import where
from pandas import concat
from pandas import read_csv
from pandas import to_numeric

from noads._data import data_file
from noads.application import lh2_tank_calibration
from noads.application.base_objects import aircraft_tech_params
from noads.application.base_objects import categories_mission
from noads.application.base_objects import category_conso
from noads.application.base_objects import propulsion_mission
from noads.application.base_objects import tech_params_lower_mid_upper_2020_2040_2060
from noads.application.base_objects import update_comparison_mission
from noads.application.base_objects import update_power_system
from noads.application.primary_energy import primary_energy_factors
from noads.core.models.fleet.aircraft_design import UPDATE_GAM_OPTIONS
from noads.core.models.fleet.aircraft_tech_parameter import AircraftTechParameter
from noads.gam_jax.models import gam_v3
from noads.gam_jax.models.gam_v3 import LH2_TANK_SIZE_LAW
from noads.gam_jax.models.generic_airplane_model import GAM as GAMV2
from noads.gam_jax.utils import unit

SCENARIOS = ("Lower", "Mid", "Upper")
years = linspace(2020, 2080, 61)
paper_years = linspace(2020, 2060, 41)
OLD = "dimgray"
NEW = "tab:blue"
RATED_EFFICIENCY_RATIO = 0.85  # fuel cell efficiency at rated power / cruise
REFERENCE_TANK_MASS = 1000.0  # kg of LH2 per tank for the GI panel


def tech_data_file(*parts):
    """Path to a bundled aircraft technology data file."""
    return data_file("noads.application", "aircraft_tech_data", *parts)


# %%
# Aircraft technology evolution
# -----------------------------
# The technology parameters of the update model are given for 2020, 2040, 2060 and
# 2080. The Mid scenario is interpolated with a monotone cubic spline (PCHIP), and
# the Lower and Upper scenarios add a monotone gap to it: all scenarios start from
# the same value in 2020, and the Upper-to-Lower band only widens. They are shown
# in blue, with the data points of the literature review of the update, and
# compared with the technology parameters of the main paper (quadratic spline
# through 2020, 2040 and 2060) and its literature data, in grey.
#
# The powertrain parameters of the update are given at a reference power of 1 MW
# per propulsor, the fuel cell specific power excludes its thermal management system
# (TMS), which is sized separately from the rated heat load, and the maximum power
# per propulsor bounds the electric and fuel cell designs. To compare with the main
# paper, the fuel cell specific power including the TMS is also shown, as well as
# the gravimetric index of an LH2 tank holding 1 t of LH2.

paper_data = read_csv(tech_data_file("paper_literature_data.csv"))
new_data = [read_csv(tech_data_file("powertrain", "powertrain_tech_data.csv"))]
literature_file = Path(tech_data_file("literature_data.csv"))
if literature_file.exists():
    new_data.append(read_csv(literature_file))
new_data = concat(new_data, ignore_index=True)
for column in ("value", "year"):
    new_data[column] = to_numeric(new_data[column], errors="coerce")
gi_data = lh2_tank_calibration.load_dataset()

tech_params = [
    {param.name: param for param in aircraft_tech_params(index, "update")}
    for index in range(3)
]


def update_curves(name):
    """Lower, Mid and Upper values of an update parameter versus EIS."""
    return [asarray(p[name].value_at_entry_into_service(years)) for p in tech_params]


def paper_curves(name):
    """Lower, Mid and Upper values of a main paper parameter versus EIS."""
    return [
        asarray(
            AircraftTechParameter(
                name, tuple(tech_params_lower_mid_upper_2020_2040_2060[name][i])
            ).value_at_entry_into_service(paper_years)
        )
        for i in range(3)
    ]


def fuelcell_system_curves():
    """Fuel cell specific power including the TMS sized at rated power (kW/kg)."""
    curves = []
    lower_mid_upper = zip(
        update_curves("fuelcell_specific_power"),
        update_curves("fuelcell_tms_heat_rejection"),
        update_curves("fuelcell_efficiency"),
    )
    for core, heat_rejection, efficiency in lower_mid_upper:
        heat_to_power = 1.0 / (1e-2 * efficiency * RATED_EFFICIENCY_RATIO) - 1.0
        curves.append(1.0 / (1.0 / core + heat_to_power / heat_rejection))
    return curves


def lh2_tank_gi_curves():
    """Gravimetric index (%) of an LH2 tank holding the reference LH2 mass."""
    a, b, c = LH2_TANK_SIZE_LAW
    ratio = a + b * REFERENCE_TANK_MASS ** (-1.0 / 3.0) + c / REFERENCE_TANK_MASS
    return [100.0 / (1.0 + k * ratio) for k in update_curves("lh2tank_mass_factor")]


def paper_constant(value):
    """A parameter constant in the main paper model."""
    return [asarray(paper_years * 0.0 + value)] * 3


# (title, unit, update curves, paper curves, data filters, paper data names)
# A data filter is (component, metric, factor, filled scopes, hollow scopes), the
# scopes being the mass scopes shown with filled or hollow markers (None: all).
tech_panels = [
    (
        "Battery Specific Energy\n(pack; hollow: cell)",
        "Wh/kg",
        lambda: update_curves("battery_specific_energy"),
        lambda: paper_curves("battery_specific_energy"),
        [("battery", "specific_energy", 1.0, ("pack",), ("cell",))],
        ["battery_specific_energy"],
    ),
    (
        "E-motor Specific Power\n(at 1 MW)",
        "kW/kg",
        lambda: update_curves("emotor_specific_power"),
        lambda: paper_curves("emotor_specific_power"),
        [("emotor", "specific_power", 1.0, None, ())],
        ["emotor_specific_power"],
    ),
    (
        "Electric chain Efficiency\n(at 1 MW; hollow: motor or inverter)",
        "%",
        lambda: update_curves("emotor_efficiency"),
        lambda: paper_constant(90.0),
        [
            ("electric_chain", "efficiency", 100.0, None, ()),
            ("emotor", "efficiency", 100.0, (), None),
            ("inverter", "efficiency", 100.0, (), None),
        ],
        [],
    ),
    (
        "Power electronics Specific Power",
        "kW/kg",
        lambda: update_curves("electronics_specific_power"),
        lambda: paper_curves("electronics_specific_power"),
        [
            ("inverter", "specific_power", 1.0, None, ()),
            ("dcdc", "specific_power", 1.0, None, ()),
        ],
        ["electronics_specific_power"],
    ),
    (
        "Fuel cell Specific Power\n(stack + BoP, no TMS; hollow: stack)",
        "kW/kg",
        lambda: update_curves("fuelcell_specific_power"),
        None,
        [
            ("fuelcell", "system_specific_power", 1.0, ("system_excl_tms",), ()),
            ("fuelcell", "stack_specific_power", 1.0, (), None),
        ],
        [],
    ),
    (
        "Fuel cell system Specific Power\n(including TMS)",
        "kW/kg",
        fuelcell_system_curves,
        lambda: paper_curves("fuelcell_specific_power"),
        [
            ("fuelcell", "system_specific_power_incl_tms", 1.0, None, ()),
            (
                "fuelcell",
                "system_specific_power",
                1.0,
                ("system", "system_incl_tms", "system_unclear"),
                (),
            ),
        ],
        ["fuelcell_specific_power"],
    ),
    (
        "Fuel cell system Efficiency\n(hollow: stack peak)",
        "%",
        lambda: update_curves("fuelcell_efficiency"),
        lambda: paper_curves("fuelcell_efficiency"),
        [
            ("fuelcell", "system_efficiency", 100.0, None, ()),
            ("fuelcell", "stack_peak_efficiency", 100.0, (), None),
        ],
        ["fuelcell_efficiency"],
    ),
    (
        "Fuel cell TMS\nSpecific Heat Rejection",
        "kW heat/kg",
        lambda: update_curves("fuelcell_tms_heat_rejection"),
        None,
        [
            ("tms", "specific_heat_rejection", 1.0, None, ()),
            ("tms", "radiator_specific_heat_rejection", 1.0, (), None),
        ],
        [],
    ),
    (
        "Fuel cell TMS\nParasitic Power",
        "kW / kW heat",
        lambda: update_curves("fuelcell_tms_power_loss"),
        None,
        [
            ("tms", "radiator_power_loss_ratio", 1.0, None, ()),
            ("tms", "parasitic_power_ratio", 1.0, None, ()),
        ],
        [],
    ),
    (
        "Maximum Power per Propulsor",
        "MW",
        lambda: update_curves("max_unit_power"),
        None,
        [
            (component, metric, 1e-3, None, ())
            for component in ("emotor", "fuelcell")
            for metric in (
                "max_unit_power_demonstrated",
                "max_unit_power_certified",
                "max_unit_power_projected",
            )
        ],
        [],
    ),
    (
        "Structural weight\n(relative to today; hollow: OEW)",
        "%",
        lambda: update_curves("struct_weight_factor"),
        lambda: paper_curves("struct_weight_factor"),
        [
            ("airframe", "structural_weight_factor", 1.0, None, ()),
            ("airframe", "empty_weight_factor", 1.0, (), None),
        ],
        ["struct_weight_factor"],
    ),
    (
        "Thermal engine efficiency\n(relative to today; paper: none)",
        "%",
        lambda: update_curves("engine_efficiency_factor"),
        lambda: paper_constant(100.0),
        [],
        [],
    ),
    (
        "LH2 tank Gravimetric Index\n(0.3 to 3 t of LH2 per tank)",
        "%",
        lh2_tank_gi_curves,
        lambda: paper_curves("lh2tank_gravimetric_index"),
        [],
        ["lh2tank_gravimetric_index"],
    ),
    (
        "LH2 Fuel System Mass\nper kg of LH2",
        "kg/kg",
        lambda: update_curves("lh2_fuel_system_ratio"),
        None,
        [("lh2_fuel_system", "mass_ratio", 1.0, None, ())],
        [],
    ),
]
status_markers = {
    "certified": "*",
    "product": "s",
    "demonstrator": "o",
    "state_of_art": "D",
    "claim": "P",
    "target": "X",
    "projection": "^",
    "concept": "v",
}
paper_markers = {
    "NASA": "o",
    "IATA": "s",
    "ATI": "d",
    "ICCT": "X",
    "EASA": "*",
    "Adler et al. (2025)": "P",
}
gi_markers = {
    "hardware_measured": "*",
    "hardware_claim": "P",
    "design_study": "o",
    "projection": "^",
    "target": "X",
    "requirement": "D",
}


def plot_band(ax, x, curves, color, band_alpha=0.2, lw=2):
    """Upper-to-Lower band, Lower (solid) and Mid (dotted) curves."""
    ax.fill_between(x, curves[0], curves[2], alpha=band_alpha, color=color, lw=0)
    ax.plot(x, curves[0], color=color, ls="-", lw=lw)
    ax.plot(x, curves[1], color=color, ls=":", lw=lw)


def plot_new_data(ax, filters):
    """Data points of the update literature review, by status (blue)."""
    for component, metric, factor, filled, hollow in filters:
        rows = new_data[new_data.component.eq(component) & new_data.metric.eq(metric)]
        for scopes, fill in ((filled, "full"), (hollow, "none")):
            if scopes is None:
                others = filled if fill == "none" else hollow
                selected = rows[~rows.mass_scope.isin(others or ())]
            else:
                selected = rows[rows.mass_scope.isin(scopes)]
            for status, marker in status_markers.items():
                points = selected[selected.status.eq(status)]
                ax.plot(
                    points.year,
                    points.value * factor,
                    marker,
                    color=NEW,
                    markersize=6,
                    alpha=0.75,
                    fillstyle=fill,
                )


def plot_paper_data(ax, names):
    """Data points of the literature review of the main paper, by source (grey)."""
    for source, marker in paper_markers.items():
        points = paper_data[
            paper_data.parameter.isin(names) & paper_data.source.eq(source)
        ]
        ax.plot(points.year, points.value, marker, color=OLD, markersize=6, alpha=0.8)


def plot_gi_data(ax):
    """Tank-only GI of tanks of 0.3 to 3 t of LH2, by data type (blue)."""
    tanks = gi_data[gi_data.gi_scope.eq("tank") & gi_data.m.between(300.0, 3000.0)]
    for data_type, marker in gi_markers.items():
        points = tanks[tanks.data_type.eq(data_type)]
        ax.plot(
            points.tech_year,
            100.0 * points.gi,
            marker,
            color=NEW,
            markersize=6,
            alpha=0.75,
        )


fig1, axes1 = subplots(4, 4, figsize=(15, 15), layout="constrained")
for ax, panel in zip(axes1.flat, tech_panels):
    title, unit_label, new_curves, old_curves, filters, paper_names = panel
    if old_curves is not None:
        plot_band(ax, paper_years, old_curves(), OLD, band_alpha=0.25)
    plot_band(ax, years, new_curves(), NEW)
    plot_paper_data(ax, paper_names)
    plot_new_data(ax, filters)
    if "Gravimetric" in title:
        plot_gi_data(ax)
    if "Maximum Power" in title:
        ax.set_yscale("log")
    ax.set_title(title, fontsize="medium")
    ax.set_ylabel(unit_label)
    ax.set_xlabel("Entry-Into-Service")
for ax in axes1.flat[len(tech_panels) :]:
    ax.set_axis_off()
legend_handles = [
    Patch(alpha=0.25, facecolor=OLD, label="Main paper: Upper-to-Lower"),
    Line2D([0], [0], color=OLD, ls="-", lw=2, label="Main paper: Lower"),
    Line2D([0], [0], color=OLD, ls=":", lw=2, label="Main paper: Mid"),
]
legend_handles.extend(
    Line2D([0], [0], ls="None", marker=marker, color=OLD, label=f"{source} (paper)")
    for source, marker in paper_markers.items()
)
legend_handles.extend([
    Patch(alpha=0.2, facecolor=NEW, label="Update: Upper-to-Lower"),
    Line2D([0], [0], color=NEW, ls="-", lw=2, label="Update: Lower"),
    Line2D([0], [0], color=NEW, ls=":", lw=2, label="Update: Mid"),
])
legend_handles.extend(
    Line2D(
        [0], [0], ls="None", marker=marker, color=NEW, label=status.replace("_", " ")
    )
    for status, marker in status_markers.items()
)
legend_handles.extend(
    Line2D([0], [0], ls="None", marker=marker, color=NEW, label=f"tank GI: {kind}")
    for kind, marker in (("hardware", "*"), ("design study", "o"))
)
axes1.flat[-1].legend(handles=legend_handles, loc="center", fontsize="small", ncols=2)
fig1.suptitle(
    "Aircraft technology parameters: main paper (grey) and update (blue)",
    fontsize="x-large",
)
fig1.savefig("./aircraft_update_technology.png", dpi=150)

# %%
# LH2 tank gravimetric index versus tank size
# -------------------------------------------
# The tank-only gravimetric index increases with the LH2 mass per tank (the tank
# mass scales with its area, as m^(2/3)) and with the maturity of the technology at
# the entry-into-service. The curves are compared with the GI dataset: design
# studies, projections, hardware and targets, plotted at their LH2 mass per tank.

gi_tanks = gi_data[gi_data.gi_scope.eq("tank") & gi_data.m.notna()]
gi_years = (2020, 2035, 2050, 2065, 2080)
gi_colors = ("#c6dbef", "#6baed6", "#3182bd", "#08519c", "#08306b")
masses = geomspace(10.0, 1.0e5, 200)
a, b, c = LH2_TANK_SIZE_LAW

fig2, axes2 = subplots(1, 3, figsize=(14, 4.5), layout="constrained", sharey=True)
for index, (ax, scenario) in enumerate(zip(axes2, SCENARIOS)):
    mass_factor = tech_params[index]["lh2tank_mass_factor"]
    for year, color in zip(gi_years, gi_colors):
        k = mass_factor.value_at_entry_into_service(year)
        gi = 1.0 / (1.0 + k * (a + b * masses ** (-1.0 / 3.0) + c / masses))
        ax.plot(masses, 100 * gi, color=color, linewidth=2, label=f"EIS {year}")
    for data_type, marker in gi_markers.items():
        rows = gi_tanks[gi_tanks.data_type.eq(data_type)]
        ax.plot(
            rows.m,
            100 * rows.gi,
            marker,
            ls="None",
            color="dimgray",
            markersize=6,
            alpha=0.6,
            label=data_type.replace("_", " "),
        )
    ax.set_xscale("log")
    ax.set_xlabel("LH2 mass per tank [kg]")
    ax.set_title(f"{scenario} technology", fontsize="medium")
    ax.set_ylim(0.0, 100.0)
axes2[0].set_ylabel("Tank gravimetric index [%]")
axes2[-1].legend(loc="lower right", fontsize="small")
fig2.suptitle("LH2 tank gravimetric index: size and technology", fontsize="large")
fig2.savefig("./aircraft_update_lh2_tank_gi.png", dpi=150)

# %%
# Size effect on thermal engines
# ------------------------------
# GAM V3.0 makes the turboprop and turbofan efficiencies depend on the power per
# engine: the turboshaft PSFC regression was refitted, and turbofans get a scale
# factor accounting for the efficiency loss of small turbomachines. GAM V2.0 (paper
# model) had no size effect for turbofans.

engine_power = geomspace(1.0e5, 5.0e7, 200)
gam2 = GAMV2()
gam3 = gam_v3.GAM()
tas = 0.78 * 296.5  # Mach 0.78 at 35000 ft
sfc2, fhv = gam2.get_turboprop_sfc(engine_power, "kerosene")
turbofan_sfc2, _ = gam2.get_turbofan_sfc(tas, 12.0, "kerosene")

fig3, ax3 = subplots(figsize=(7, 4.5), layout="constrained")
ax3.plot(
    unit.MW_W(engine_power),
    gam2.prop_eff / (sfc2 * fhv),
    color="darkgoldenrod",
    ls=":",
    lw=2,
    label="Turboprop, GAM V2.0 (paper)",
)
ax3.plot(
    unit.MW_W(engine_power),
    gam3.get_turboprop_eff(engine_power),
    color="darkgoldenrod",
    lw=2,
    label="Turboprop, GAM V3.0 (update)",
)
ax3.plot(
    unit.MW_W(engine_power),
    [tas / (turbofan_sfc2 * fhv)] * len(engine_power),
    color="maroon",
    ls=":",
    lw=2,
    label="Turbofan, GAM V2.0 (paper)",
)
ax3.plot(
    unit.MW_W(engine_power),
    gam3.get_turbofan_eff(tas, 12.0, engine_power, "kerosene"),
    color="maroon",
    lw=2,
    label="Turbofan, GAM V3.0 (update)",
)
ax3.set_xscale("log")
ax3.set_xlabel("Maximum power per engine [MW]")
ax3.set_ylabel("Overall efficiency (incl. propeller or fan) [-]")
ax3.set_ylim(ymin=0.0)
ax3.legend(loc="lower right")
ax3.set_title(
    "Thermal engine efficiency versus size\n(kerosene, turbofan BPR 12 at Mach 0.78)"
)
fig3.savefig("./aircraft_update_engine_size.png", dpi=150)

# %%
# Prospective aircraft design
# ---------------------------
# For each year of entry-into-service, an aircraft is designed with the update
# model for each architecture and market, including turboprops. Electric and fuel
# cell designs are only shown where their power per propulsor is available at their
# entry-into-service (constrained in the optimization), and designs that do not
# close (e.g. immature batteries) are not shown: the band then spans the feasible
# scenarios only.

propulsion_colors = {
    "JetA-GasTurbine": "maroon",
    "Battery-Electric": "limegreen",
    "lH2-FuelCell": "royalblue",
    "lH2-GasTurbine": "orangered",
    "JetA-Turboprop": "darkgoldenrod",
    "lH2-Turboprop": "mediumorchid",
}
missions = {**propulsion_mission, **update_comparison_mission}
DESIGN_OUTPUTS = (
    "enrg_consumption",
    "total_energy",
    "mtow",
    "owe",
    "propulsion_system_efficiency",
    "max_power",
    "unit_power_ratio",
    "closed",
    "energy_storage_mass",
    "fuel_cell_system_mass",
)


def design_sweep(category, architecture, tech_idx):
    """Design outputs versus entry-into-service, with the update aircraft model."""
    power_system = update_power_system(architecture, category)
    mission = {
        **categories_mission[category],
        **missions[architecture],
        "category": category,
    }
    params = aircraft_tech_params(tech_idx, "update")
    electric = power_system["engine_type"] == "emotor"
    hydrogen = power_system["energy_type"] == "liquid_h2"

    def design(eis):
        gam = gam_v3.GAM(
            **{p.name: p.value_at_entry_into_service(eis) for p in params},
            **UPDATE_GAM_OPTIONS,
        )
        result = gam.design_airplane(dict(power_system), dict(mission))
        outputs = {name: result[name] for name in DESIGN_OUTPUTS}
        if hydrogen:
            outputs["gi_tank"] = result["gi_tank"]
        if electric:
            outputs["electric_chain_efficiency"] = result["electric_chain_efficiency"]
            outputs["emotor_specific_power"] = gam.emotor_power_density(
                result["max_power"]
            )
            outputs["max_unit_power"] = gam.max_unit_power
        if electric and not hydrogen:
            # efficiency at the reference power per propulsor, which does not
            # require the design to close
            outputs["open_propulsion_efficiency"] = gam.get_emotor_eff(
                power_system["energy_type"],
                power_system["thruster_type"],
                gam.reference_unit_power,
            )
        if electric and hydrogen:
            outputs["fuel_cell_efficiency"] = gam.fuel_cell_mission_efficiency(
                result["max_power"]
            )
            outputs["fuel_cell_gross_power"] = result["fuel_cell_gross_power"]
        return outputs

    outputs = {name: asarray(v) for name, v in jit(vmap(design))(years).items()}
    feasible = outputs["closed"] > 0.5
    if electric:
        feasible &= outputs["unit_power_ratio"] <= 1.0
    outputs["feasible"] = feasible
    return outputs


designs = {
    (category, architecture, tech_idx): design_sweep(category, architecture, tech_idx)
    for category in categories_mission
    for architecture in propulsion_colors
    for tech_idx in range(3)
}


def masked(values, feasible):
    """Values where feasible, NaN elsewhere."""
    return where(feasible & isfinite(values), values, nan)


def plot_architecture(ax, curves, color):
    """Band of the feasible scenarios, Lower (solid) and Mid (dotted) curves."""
    stacked = vstack(curves)
    band_low = fmin.reduce(stacked)
    band_high = fmax.reduce(stacked)
    ax.fill_between(
        years,
        band_low,
        band_high,
        where=isfinite(band_low),
        alpha=0.2,
        color=color,
        linewidth=0.0,
    )
    ax.plot(years, curves[0], color=color, linestyle="-", linewidth=3)
    ax.plot(years, curves[1], color=color, linestyle=":", linewidth=3)


def market_figure(
    title, metric, architectures=tuple(propulsion_colors), ymax=None, unmasked=()
):
    """One panel per market, one band per architecture, versus EIS.

    The architectures of ``unmasked`` are shown even where their design does not close.
    """
    fig, axes = subplots(2, 3, layout="constrained", figsize=(12, 10))
    fig.suptitle(title, fontsize="x-large")
    for ax, category in zip(axes.flat, categories_mission):
        seat = categories_mission[category]["npax"]
        max_range = 1e-3 * categories_mission[category]["range"]
        for architecture in architectures:
            curves = []
            for tech_idx in range(3):
                outputs = designs[category, architecture, tech_idx]
                feasible = True if architecture in unmasked else outputs["feasible"]
                curves.append(masked(metric(outputs, category), feasible))
            plot_architecture(ax, curves, propulsion_colors[architecture])
        ax.set_title(
            f"{category.replace('_', ' ')}\n({seat} seat, {max_range} km)",
            fontsize="large",
        )
        ax.set_ylim(ymin=0.0, ymax=ymax)
    axes[-1, -1].clear()
    axes[-1, -1].set_axis_off()
    axes[-1, 0].set_xlabel("Entry-Into-Service")
    return fig, axes


def market_legend(axes, architectures=tuple(propulsion_colors), extra=()):
    """Legend of a market figure, in its last panel."""
    handles = [
        Patch(color=propulsion_colors[name], label=name) for name in architectures
    ]
    handles.extend([
        Line2D([0], [0], color="k", ls="-", lw=3, label="Lower"),
        Line2D([0], [0], color="k", ls=":", lw=3, label="Mid"),
        *extra,
    ])
    axes[-1, -1].legend(handles=handles, loc="center")


def seat_km(category):
    """Seat-km of the design mission."""
    return categories_mission[category]["npax"] * categories_mission[category]["range"]


fig4, axes4 = market_figure(
    "Aircraft Energy Efficiency, update model\n[seat km / MJ]",
    lambda outputs, category: 1.0 / (outputs["enrg_consumption"] * 1.0e-3),
    ymax=5.0,
)
for ax, category in zip(axes4.flat, categories_mission):
    current_energy_per_ask = category_conso[category]
    ax.hlines(
        1.0 / current_energy_per_ask[0], years[0], years[-1], colors="dimgray", lw=1.5
    )
    ax.hlines(
        1.0 / current_energy_per_ask[1],
        years[0],
        years[-1],
        colors="dimgray",
        linestyles=":",
        lw=1.5,
    )
    ax.fill_between(
        [years[0], years[-1]],
        [1.0 / current_energy_per_ask[0]] * 2,
        [1.0 / current_energy_per_ask[2]] * 2,
        alpha=0.4,
        color="dimgray",
        lw=0,
    )
market_legend(
    axes4, extra=[Patch(color="dimgray", alpha=0.4, label="Current Technology")]
)
fig4.savefig("./aircraft_update_prospective_energy.png", dpi=150)

fig5, axes5 = market_figure(
    "Aircraft Empty-Mass Efficiency, update model\n[seat km / kg]",
    lambda outputs, category: 1e-3 * seat_km(category) / outputs["owe"],
)
market_legend(axes5)
fig5.savefig("./aircraft_update_prospective_mass.png", dpi=150)

# %%
# Embarked energy and overall propulsion efficiency
# -------------------------------------------------
# The embarked energy is the energy stored on board for the design mission,
# including reserves, per seat-km of the design mission. The overall propulsion
# efficiency is the ratio of the thrust power to the power drawn from the energy
# carrier at cruise: thermal, propulsive and propeller or fan efficiencies for
# thermal engines, and fuel cell, electric chain and propeller or fan efficiencies
# for electric aircraft. Battery-electric aircraft are shown at the reference power
# per propulsor (1 MW), also where their design does not close.

fig6, axes6 = market_figure(
    "Embarked Energy (design mission with reserves), update model\n[MJ / seat km]",
    lambda outputs, category: 1e-3 * outputs["total_energy"] / seat_km(category),
    ymax=6.0,
)
# Current aircraft: energy per seat-km of the 2019 fleet, scaled by the ratio of the
# embarked energy to the mission energy (reserves) of a present-day kerosene design.
for ax, category in zip(axes6.flat, categories_mission):
    jet = designs[category, "JetA-GasTurbine", 1]
    reserves = float(
        jet["total_energy"][0] / (jet["enrg_consumption"][0] * seat_km(category))
    )
    low, mid, high = (reserves * value for value in category_conso[category])
    ax.hlines(low, years[0], years[-1], colors="dimgray", lw=1.5)
    ax.hlines(mid, years[0], years[-1], colors="dimgray", linestyles=":", lw=1.5)
    ax.fill_between(
        [years[0], years[-1]], [low] * 2, [high] * 2, alpha=0.4, color="dimgray", lw=0
    )
market_legend(
    axes6, extra=[Patch(color="dimgray", alpha=0.4, label="Current Technology")]
)
fig6.savefig("./aircraft_update_embarked_energy.png", dpi=150)

fig7, axes7 = market_figure(
    "Overall Propulsion Efficiency at cruise, update model\n[%]",
    lambda outputs, category: 100.0
    * outputs.get(
        "open_propulsion_efficiency", outputs["propulsion_system_efficiency"]
    ),
    ymax=100.0,
    unmasked=("Battery-Electric",),
)
market_legend(axes7)
fig7.savefig("./aircraft_update_propulsion_efficiency.png", dpi=150)

# %%
# Primary energy per seat-km
# --------------------------
# The energy per seat-km of each architecture (final energy, design mission) is
# multiplied by the primary energy per final energy of its carrier at the EIS year
# (:mod:`noads.application.primary_energy`), which depends on time only: the bands
# come from aircraft technology only. The
# architectures are compared per primary resource: oil for the Jet-A aircraft, biomass
# for the Jet-A aircraft on each biofuel pathway, and electricity for the Jet-A
# aircraft on e-fuel, the battery-electric and the two LH2 aircraft. Bands span the
# Lower to Upper scenarios (solid line for Lower and dotted line for Mid); turboprops,
# which are not in the optimized fleets, are shown by their Mid curve only
# (dash-dotted).

primary_factors = primary_energy_factors(years)


def primary_curves(category, architecture, carrier):
    """Primary energy per seat-km (MJ) of an architecture on a carrier, per scenario."""
    return [
        masked(
            1e-3
            * designs[category, architecture, index]["enrg_consumption"]
            * primary_factors[carrier],
            designs[category, architecture, index]["feasible"],
        )
        for index in range(3)
    ]


def primary_figure(title, specs, ymax, reference=None):
    """One panel per market; each spec is (label, architecture, carrier, color)."""
    fig, axes = subplots(2, 3, layout="constrained", figsize=(12, 10))
    fig.suptitle(title, fontsize="x-large")
    for ax, category in zip(axes.flat, categories_mission):
        seat = categories_mission[category]["npax"]
        max_range = 1e-3 * categories_mission[category]["range"]
        if reference is not None:
            low, mid, high = (
                current * primary_factors["Fossil kerosene"][0]
                for current in category_conso[category]
            )
            ax.fill_between(
                [years[0], years[-1]], [low] * 2, [high] * 2,
                color="dimgray", alpha=0.4, lw=0,
            )  # fmt: skip
            ax.hlines(mid, years[0], years[-1], colors="dimgray", ls=":", lw=1.5)
        for _label, architecture, carrier, color in specs:
            curves = primary_curves(category, architecture, carrier)
            if "Turboprop" in architecture:
                ax.plot(years, curves[1], color=color, ls="-.", lw=1.5)
            else:
                plot_architecture(ax, curves, color)
        ax.set_title(
            f"{category.replace('_', ' ')}\n({seat} seat, {max_range} km)",
            fontsize="large",
        )
        ax.set_ylim(0.0, ymax)
    axes[-1, -1].clear()
    axes[-1, -1].set_axis_off()
    axes[-1, 0].set_xlabel("Entry-Into-Service")
    axes[0, 0].set_ylabel("Primary energy per seat-km [MJ]")
    handles = [
        Patch(color=color, label=label)
        for label, architecture, _, color in specs
        if "Turboprop" not in architecture
    ]
    handles.extend([
        Line2D([0], [0], color="k", ls="-", lw=3, label="Lower"),
        Line2D([0], [0], color="k", ls=":", lw=3, label="Mid"),
        Line2D([0], [0], color="k", ls="-.", lw=1.5, label="Turboprop (Mid)"),
    ])
    if reference is not None:
        handles.append(Patch(color="dimgray", alpha=0.4, label=reference))
    axes[-1, -1].legend(handles=handles, loc="center")
    return fig, axes


fig_oil, _ = primary_figure(
    "Primary energy: oil, update model\n[MJ oil / seat km]",
    [
        ("Jet-A", "JetA-GasTurbine", "Fossil kerosene", "maroon"),
        ("Jet-A turboprop", "JetA-Turboprop", "Fossil kerosene", "maroon"),
    ],
    ymax=4.0,
    reference="Current fleet",
)
fig_oil.savefig("./aircraft_update_primary_oil.png", dpi=150)

fig_biomass, _ = primary_figure(
    "Primary energy: biomass, update model\n[MJ biomass / seat km]",
    [
        (f"Jet-A {pathway}", architecture, pathway, color)
        for pathway, color in (
            ("HEFA", "olivedrab"),
            ("ATJ", "yellowgreen"),
            ("FT", "darkgreen"),
        )
        for architecture in ("JetA-GasTurbine", "JetA-Turboprop")
    ],
    ymax=16.0,
)
fig_biomass.savefig("./aircraft_update_primary_biomass.png", dpi=150)

fig_electricity, _ = primary_figure(
    "Primary energy: electricity, update model\n[MJ electricity / seat km]",
    [
        ("Jet-A e-fuel", "JetA-GasTurbine", "E-fuel", "darkorange"),
        ("Jet-A turboprop e-fuel", "JetA-Turboprop", "E-fuel", "darkorange"),
        ("Battery-Electric", "Battery-Electric", "Battery", "limegreen"),
        ("lH2-GasTurbine", "lH2-GasTurbine", "LH2", "orangered"),
        ("lH2-FuelCell", "lH2-FuelCell", "LH2", "royalblue"),
    ],
    ymax=8.0,
)
fig_electricity.savefig("./aircraft_update_primary_electricity.png", dpi=150)

# %%
# LH2 tank gravimetric index per market
# -------------------------------------
# The realized tank gravimetric index of the LH2 aircraft depends on their tank
# size, hence on their market and architecture, and on the tank technology at the
# entry-into-service. The grey band recalls the size-independent gravimetric index
# of the main paper.

lh2_architectures = ("lH2-FuelCell", "lH2-GasTurbine", "lH2-Turboprop")
fig8, axes8 = market_figure(
    "LH2 tank Gravimetric Index per market, update model\n[%]",
    lambda outputs, category: 100.0 * outputs["gi_tank"],
    architectures=lh2_architectures,
    ymax=100.0,
)
paper_gi = paper_curves("lh2tank_gravimetric_index")
for ax in axes8.flat[: len(categories_mission)]:
    plot_band(ax, paper_years, paper_gi, OLD, band_alpha=0.25)
market_legend(
    axes8,
    lh2_architectures,
    extra=[Patch(color=OLD, alpha=0.25, label="Main paper (all sizes)")],
)
fig8.savefig("./aircraft_update_tank_gi_per_market.png", dpi=150)

# %%
# Electric and fuel cell design point
# -----------------------------------
# To understand the performance of fuel cell aircraft, the main parameters of
# their design point are shown for each market in the Mid scenario (thick where the
# power per propulsor is available at the entry-into-service, thin dashed beyond):
# the power per propulsor, the efficiencies of the electric chain and of the fuel
# cell system (net of the TMS parasitic power, at cruise), the resulting overall
# propulsion efficiency, the specific powers realized at the design power per
# propulsor, and the mass fractions of the fuel cell system and of the LH2 storage.

market_colors = {
    "general": "tab:green",
    "commuter": "gold",
    "regional": "darkorange",
    "short_medium": "darkviolet",
    "long_range": "tab:red",
}
fuel_cell_panels = [
    (
        "Power per Propulsor\n(black dashed: maximum available)",
        "MW",
        lambda o: 1e-6 * o["max_power"],
    ),
    (
        "Electric chain Efficiency",
        "%",
        lambda o: 100.0 * o["electric_chain_efficiency"],
    ),
    (
        "Fuel cell system Efficiency\n(cruise, net of TMS parasitic power)",
        "%",
        lambda o: 100.0 * o["fuel_cell_efficiency"],
    ),
    (
        "Overall Propulsion Efficiency\n(H2 to thrust power, cruise)",
        "%",
        lambda o: 100.0 * o["propulsion_system_efficiency"],
    ),
    (
        "Fuel cell system Specific Power\n(gross power / mass incl. TMS)",
        "kW/kg",
        lambda o: o["fuel_cell_gross_power"] / o["fuel_cell_system_mass"] * 1e-3,
    ),
    (
        "E-motor Specific Power\n(at the design power per propulsor)",
        "kW/kg",
        lambda o: 1e-3 * o["emotor_specific_power"],
    ),
    (
        "Fuel cell system Mass\n(incl. TMS, % of MTOW)",
        "%",
        lambda o: 100.0 * o["fuel_cell_system_mass"] / o["mtow"],
    ),
    (
        "LH2 tank and fuel system Mass\n(% of MTOW)",
        "%",
        lambda o: 100.0 * o["energy_storage_mass"] / o["mtow"],
    ),
]
fig9, axes9 = subplots(2, 4, layout="constrained", figsize=(16, 8.5))
for ax, (title, unit_label, metric) in zip(axes9.flat, fuel_cell_panels):
    for category, color in market_colors.items():
        outputs = designs[category, "lH2-FuelCell", 1]
        values = metric(outputs)
        ax.plot(years, values, color=color, lw=1, ls="--")
        ax.plot(years, masked(values, outputs["feasible"]), color=color, lw=3)
    if "Power per Propulsor" in title:
        ax.plot(years, 1e-6 * outputs["max_unit_power"], "k--", lw=2)
        ax.set_yscale("log")
    else:
        ax.set_ylim(ymin=0.0)
    ax.set_title(title, fontsize="medium")
    ax.set_ylabel(unit_label)
    ax.set_xlabel("Entry-Into-Service")
fig9.legend(
    handles=[
        Line2D([0], [0], color=color, lw=3, label=category.replace("_", " "))
        for category, color in market_colors.items()
    ],
    loc="outside lower center",
    ncols=5,
)
fig9.suptitle(
    "Fuel cell aircraft design point, update model (Mid scenario)", fontsize="x-large"
)
fig9.savefig("./aircraft_update_fuel_cell_design.png", dpi=150)
