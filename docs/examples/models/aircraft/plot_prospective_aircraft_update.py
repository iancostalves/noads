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
since GAM V3.0 changed how their efficiency varies with size.
"""

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
from pandas import read_csv

from noads._data import data_file
from noads.application import lh2_tank_calibration
from noads.application.base_objects import aircraft_tech_params
from noads.application.base_objects import categories_mission
from noads.application.base_objects import category_conso
from noads.application.base_objects import lh2_tank_tech_params_lower_mid_upper
from noads.application.base_objects import propulsion_mission
from noads.application.base_objects import update_comparison_mission
from noads.application.base_objects import update_power_system
from noads.core.models.fleet.aircraft_design import AircraftDesign
from noads.core.models.fleet.aircraft_operation import AircraftOperation
from noads.core.models.fleet.aircraft_tech_parameter import LogisticTechParameter
from noads.gam_jax.models import gam_v3
from noads.gam_jax.models.gam_v3 import LH2_TANK_SIZE_LAW
from noads.gam_jax.models.generic_airplane_model import GAM as GAMV2
from noads.gam_jax.utils import unit

SCENARIOS = ("Lower", "Mid", "Upper")
years = linspace(2020, 2080, 61)

# %%
# Aircraft technology evolution
# -----------------------------
# The technology parameters of the update model are given for 2020, 2040, 2060 and
# 2080. The Mid scenario is interpolated with a monotone cubic spline (PCHIP), and
# the Lower and Upper scenarios add a monotone gap to it: all scenarios start from
# the same value in 2020, and the Upper-to-Lower band only widens. The powertrain
# parameters are given at a reference power of 1 MW per propulsor, the fuel cell
# specific power excludes its thermal management system (TMS), and the maximum power
# per propulsor bounds the electric and fuel cell designs. They are compared with
# the data points of the powertrain calibration dataset. The LH2 tank mass factor
# ``k`` multiplies the tank-to-fuel mass ratio of present-day aluminium tanks, and
# the fuel system adds ``s`` kg per kg of LH2.

powertrain_data = read_csv(
    data_file(
        "noads.application",
        "aircraft_tech_data",
        "powertrain",
        "powertrain_tech_data.csv",
    )
)
tech_panels = {
    "battery_specific_energy": ("Battery Specific Energy", "Wh/kg", []),
    "emotor_specific_power": (
        "E-motor Specific Power\n(at 1 MW)",
        "kW/kg",
        [("emotor", "specific_power", 1.0)],
    ),
    "emotor_efficiency": (
        "Electric chain Efficiency\n(at 1 MW)",
        "%",
        [("emotor", "efficiency", 100.0)],
    ),
    "electronics_specific_power": (
        "Power electronics Specific Power",
        "kW/kg",
        [("inverter", "specific_power", 1.0), ("dcdc", "specific_power", 1.0)],
    ),
    "fuelcell_specific_power": (
        "Fuel cell Specific Power\n(stack + BoP, without TMS)",
        "kW/kg",
        [("fuelcell", "stack_specific_power", 1.0)],
    ),
    "fuelcell_efficiency": (
        "Fuel cell Efficiency",
        "%",
        [("fuelcell", "system_efficiency", 100.0)],
    ),
    "fuelcell_tms_heat_rejection": (
        "Fuel cell TMS\nSpecific Heat Rejection",
        "kW heat/kg",
        [
            ("tms", "specific_heat_rejection", 1.0),
            ("tms", "radiator_specific_heat_rejection", 1.0),
        ],
    ),
    "fuelcell_tms_power_loss": (
        "Fuel cell TMS\nParasitic Power",
        "kW / kW heat",
        [("tms", "radiator_power_loss_ratio", 1.0)],
    ),
    "max_unit_power": (
        "Maximum Power per Propulsor",
        "MW",
        [
            ("emotor", "max_unit_power_demonstrated", 1e-3),
            ("emotor", "max_unit_power_certified", 1e-3),
            ("fuelcell", "max_unit_power_demonstrated", 1e-3),
            ("fuelcell", "max_unit_power_certified", 1e-3),
        ],
    ),
    "struct_weight_factor": ("Structural weight\n(relative to today)", "%", []),
    "lh2tank_mass_factor": ("LH2 tank Mass Factor k\n(1: aluminium today)", "-", []),
    "lh2_fuel_system_ratio": ("LH2 Fuel System Mass\nper kg of LH2", "kg/kg", []),
}
status_markers = {
    "certified": "*",
    "product": "s",
    "demonstrator": "o",
    "state_of_art": "D",
    "claim": "P",
    "target": "X",
    "projection": "^",
}
tech_params = [
    {param.name: param for param in aircraft_tech_params(index, "update")}
    for index in range(3)
]

fig1, axes1 = subplots(3, 4, figsize=(14, 10), layout="constrained")
for ax, (name, (title, unit_label, data_keys)) in zip(axes1.flat, tech_panels.items()):
    lower, mid, upper = (
        params[name].value_at_entry_into_service(years) for params in tech_params
    )
    ax.fill_between(years, lower, upper, alpha=0.2, color="k")
    ax.plot(years, lower, "k-", linewidth=2)
    ax.plot(years, mid, "k:", linewidth=2)
    for component, metric, factor in data_keys:
        rows = powertrain_data[
            powertrain_data.component.eq(component) & powertrain_data.metric.eq(metric)
        ]
        for status, marker in status_markers.items():
            selected = rows[rows.status.eq(status)]
            ax.plot(
                selected.year,
                selected.value * factor,
                marker,
                color="C0",
                markersize=7,
                alpha=0.7,
            )
    if name == "max_unit_power":
        ax.set_yscale("log")
    ax.set_title(title, fontsize="medium")
    ax.set_ylabel(unit_label)
for ax in axes1[-1]:
    ax.set_xlabel("Entry-Into-Service")
legend_handles = [
    Line2D([0], [0], color="k", ls="-", lw=2, label="Lower"),
    Line2D([0], [0], color="k", ls=":", lw=2, label="Mid"),
    Patch(alpha=0.2, facecolor="k", label="Upper-to-Lower"),
]
legend_handles.extend(
    Line2D([0], [0], ls="None", marker=marker, markersize=7, color="C0", label=status)
    for status, marker in status_markers.items()
)
fig1.legend(handles=legend_handles, loc="outside lower center", ncols=5)
fig1.suptitle(
    "Aircraft Technology Parameters evolution, update model", fontsize="large"
)
fig1.savefig("./aircraft_update_technology.png", dpi=150)

# %%
# LH2 tank gravimetric index versus tank size
# -------------------------------------------
# The tank-only gravimetric index increases with the LH2 mass per tank (the tank
# mass scales with its area, as m^(2/3)) and with the maturity of the technology at
# the entry-into-service. The curves are compared with the GI dataset: design
# studies, projections, hardware and targets, plotted at their LH2 mass per tank.

gi_data = lh2_tank_calibration.load_dataset()
gi_data = gi_data[gi_data.gi_scope.eq("tank") & gi_data.m.notna()]
data_markers = {
    "design_study": "o",
    "projection": "^",
    "hardware_measured": "*",
    "hardware_claim": "P",
    "target": "X",
    "requirement": "D",
}
gi_years = (2025, 2035, 2045, 2060, 2080)
gi_colors = ("#c6dbef", "#6baed6", "#3182bd", "#08519c", "#08306b")
masses = geomspace(10.0, 1.0e5, 200)
a, b, c = LH2_TANK_SIZE_LAW

fig2, axes2 = subplots(1, 3, figsize=(14, 4.5), layout="constrained", sharey=True)
for index, (ax, scenario) in enumerate(zip(axes2, SCENARIOS)):
    mass_factor = LogisticTechParameter(
        "k", lh2_tank_tech_params_lower_mid_upper["lh2tank_mass_factor"][index]
    )
    for year, color in zip(gi_years, gi_colors):
        k = mass_factor.value_at_entry_into_service(year)
        gi = 1.0 / (1.0 + k * (a + b * masses ** (-1.0 / 3.0) + c / masses))
        ax.plot(masses, 100 * gi, color=color, linewidth=2, label=f"EIS {year}")
    for data_type, marker in data_markers.items():
        rows = gi_data[gi_data.data_type.eq(data_type)]
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
# model for each architecture and market, including turboprops. Fuel cell aircraft
# use more propulsors on larger markets. Electric and fuel cell designs are only
# shown where their power per propulsor is available at their entry-into-service
# (constrained in the optimization), and designs that do not close (e.g. immature
# batteries) are not shown: the band then spans the feasible scenarios only.

propulsion_colors = {
    "JetA-GasTurbine": "maroon",
    "Battery-Electric": "limegreen",
    "lH2-FuelCell": "royalblue",
    "lH2-GasTurbine": "orangered",
    "JetA-Turboprop": "darkgoldenrod",
    "lH2-Turboprop": "mediumorchid",
}
missions = {**propulsion_mission, **update_comparison_mission}


def design_sweep(category, architecture, tech_idx):
    """Energy per ASK, OWE and unit power ratio versus entry-into-service."""
    name = f"{category}_{architecture}"
    aircraft = AircraftDesign(
        name=name,
        propulsion=None,
        mission={
            **categories_mission[category],
            **missions[architecture],
            "category": category,
        },
        power_system=update_power_system(architecture, category),
        aircraft_tech_params=aircraft_tech_params(tech_idx, "update"),
        reference_aircraft=AircraftOperation(name="ref", propulsion=None),
        aircraft_model="update",
    )
    model = aircraft.design_model()
    model.discipline.jax_out_func = vmap(model.discipline.jax_out_func)
    outputs = model.discipline.execute({f"{name}.entry_into_service": years})
    return (
        asarray(outputs[f"{name}.energy_per_ask"]),
        asarray(outputs[f"{name}.owe"]),
        asarray(outputs[f"{name}.unit_power_ratio"]),
    )


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


fig4, axes4 = subplots(2, 3, layout="constrained", figsize=(12, 10))
fig4.suptitle(
    "Aircraft Energy Efficiency, update model\n[seat km / MJ]", fontsize="x-large"
)
fig5, axes5 = subplots(2, 3, layout="constrained", figsize=(12, 10))
fig5.suptitle(
    "Aircraft Empty-Mass Efficiency, update model\n[seat km / kg]",
    fontsize="x-large",
)
ymax = 5.0

for cat, category in enumerate(categories_mission):
    max_range = 1e-3 * categories_mission[category]["range"]
    seat = categories_mission[category]["npax"]
    ax_e = axes4.flat[cat]
    ax_m = axes5.flat[cat]

    current_energy_per_ask = category_conso[category]
    ax_e.hlines(
        y=1.0 / current_energy_per_ask[0],
        xmin=years[0],
        xmax=years[-1],
        colors="dimgray",
        linestyles="-",
        linewidth=1.5,
    )
    ax_e.hlines(
        y=1.0 / current_energy_per_ask[1],
        xmin=years[0],
        xmax=years[-1],
        colors="dimgray",
        linestyles=":",
        linewidth=1.5,
    )
    ax_e.fill_between(
        [years[0], years[-1]],
        [1.0 / current_energy_per_ask[0]] * 2,
        [1.0 / current_energy_per_ask[2]] * 2,
        alpha=0.4,
        color="dimgray",
    )

    for architecture, color in propulsion_colors.items():
        electric = (
            update_power_system(architecture, category)["engine_type"] == "emotor"
        )
        energy, mass = [], []
        for tech_idx in range(3):
            energy_per_ask, owe, ratio = design_sweep(category, architecture, tech_idx)
            feasible = ratio <= 1.0 if electric else isfinite(ratio)
            energy.append(masked(1.0 / energy_per_ask, feasible))
            mass.append(masked(seat * max_range / owe, feasible))
        plot_architecture(ax_e, energy, color)
        plot_architecture(ax_m, mass, color)

    for ax in (ax_e, ax_m):
        ax.set_title(
            f"{category.replace('_', ' ')}\n({seat} seat, {max_range} km)",
            fontsize="large",
        )
        ax.set_ylim(ymin=0.0)
    ax_e.set_ylim(ymax=ymax)

handles = [Patch(color=color, label=name) for name, color in propulsion_colors.items()]
handles.extend([
    Line2D([0], [0], color="k", ls="-", lw=3, label="Lower"),
    Line2D([0], [0], color="k", ls=":", lw=3, label="Mid"),
])
current = Patch(color="dimgray", alpha=0.4, label="Current Technology")
for fig_axes, fig_handles in ((axes4, [*handles, current]), (axes5, handles)):
    fig_axes[-1, -1].clear()
    fig_axes[-1, -1].set_axis_off()
    fig_axes[-1, -1].legend(handles=fig_handles, loc="center")
    fig_axes[-1, 0].set_xlabel("Entry-Into-Service")

fig4.savefig("./aircraft_update_prospective_energy.png", dpi=150)
fig5.savefig("./aircraft_update_prospective_mass.png", dpi=150)
