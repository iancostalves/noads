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

"""Hydrogen aircraft per market: fuel cell versus gas turbine, update versus paper.

Usage, from this directory: ``python h2_design_study.py``. It designs the LH2 fuel
cell and LH2 gas turbine aircraft of every market, for every entry into service
(EIS) from 2035 to 2060 and every technology scenario, with the update model, exactly
as the fleet does (same missions, power systems and GAM options). It also designs:

- the same aircraft with the paper model (GAM V2.0), every 5 years;
- fuel cell aircraft on missions other than the NOADS markets, among which one close
  to the Airbus ZEROe fuel cell concept (100 seats, 1000 nmi, 4 propulsors);
- a step-by-step change from the paper model to the update model, at EIS 2045, Mid
  technology, to attribute the change of the energy per seat-km.

The results are cached in ``h2_design_study.json``; the figures are written to
``figures/``.
"""

import json
from pathlib import Path

import jax
import numpy as np

from noads.application.base_objects import UPDATE_GAM_OPTIONS
from noads.application.base_objects import aircraft_tech_params
from noads.application.base_objects import categories_mission
from noads.application.base_objects import category_conso
from noads.application.base_objects import propulsion_architectures
from noads.application.base_objects import propulsion_mission
from noads.application.base_objects import update_power_system
from noads.gam_jax.models import gam_v3
from noads.gam_jax.models.generic_airplane_model import GAM as GAM_V2

HERE = Path(__file__).parent
CACHE = HERE / "h2_design_study.json"
PARTIAL = HERE / ".h2_design_study.partial.json"
FIGURES = HERE / "figures"
RESULTS = HERE.parents[1] / "examples/optimization/results"

MARKETS = list(categories_mission)
TECHS = {"Lower": 0, "Mid": 1, "Upper": 2}
ARCHITECTURES = {"lH2-FuelCell": "LH2 fuel cell", "lH2-GasTurbine": "LH2 gas turbine"}
YEARS = list(range(2035, 2061))
PAPER_YEARS = list(range(2035, 2061, 5))
FIRST_ALTERNATIVE_EIS = 2040
"""Lower bound of the EIS of alternative aircraft in the update model."""

ZEROE_CASES = {
    # name: (host market of the 2019 fleet reference, seats, range [m], propulsors)
    "NOADS commuter (50 seats, 1500 km, 2 prop.)": ("commuter", 50, 1500e3, 2),
    "NOADS regional (80 seats, 4500 km, 2 prop.)": ("regional", 80, 4500e3, 2),
    "Regional with 4 prop. (80 seats, 4500 km)": ("regional", 80, 4500e3, 4),
    "80 seats, 1850 km, 2 prop.": ("regional", 80, 1850e3, 2),
    "ZEROe-like (100 seats, 1850 km, 4 prop.)": ("regional", 100, 1850e3, 4),
}
"""Fuel cell missions compared with the NOADS markets. 1850 km is about 1000 nmi."""

OUTPUTS = (
    "energy_per_ask",
    "max_power",
    "total_power",
    "unit_power_ratio",
    "mtow",
    "owe",
    "propulsion_mass",
    "fuel_cell_system_mass",
    "energy_storage_mass",
    "total_fuel",
    "payload",
    "closure_gap",
    "closed",
)


def _scalars(design, max_unit_power=None):
    """The outputs of a GAM design as floats, in MJ/seat-km, MW and t."""
    out = {}
    for key in OUTPUTS:
        if key == "energy_per_ask":
            out[key] = 1.0e-3 * float(design["enrg_consumption"])
        elif key in ("max_power", "total_power"):
            out[key] = 1.0e-6 * float(design[key])
        elif key in ("unit_power_ratio", "closure_gap", "closed"):
            out[key] = float(design[key]) if key in design else None
        else:
            out[key] = 1.0e-3 * float(design[key]) if key in design else None
    if max_unit_power is not None:
        out["max_unit_power"] = 1.0e-6 * float(max_unit_power)
    return out


def _mission(prop, market, npax=None, design_range=None):
    mission = {
        **categories_mission[market],
        **propulsion_mission[prop],
        "category": market,
    }
    if npax is not None:
        mission["npax"] = npax
    if design_range is not None:
        mission["range"] = design_range
    return mission


def design_update(
    prop, market, tech, eis, npax=None, design_range=None, engines=None, **overrides
):
    """An aircraft of the update model, designed as in the fleet.

    ``overrides`` replace technology parameters, for sensitivity studies.
    """
    power_system = update_power_system(prop, market)
    if engines is not None:
        power_system["engine_count"] = engines
    gam = gam_v3.GAM(
        **{
            p.name: p.value_at_entry_into_service(float(eis))
            for p in aircraft_tech_params(tech, "update")
        }
        | overrides,
        **UPDATE_GAM_OPTIONS,
    )
    design = gam.design_airplane(
        power_system, _mission(prop, market, npax, design_range)
    )
    return _scalars(design, gam.max_unit_power)


def design_paper(prop, market, tech, eis):
    """An aircraft of the paper model (GAM V2.0), designed as in the fleet."""
    gam = GAM_V2(**{
        p.name: p.value_at_entry_into_service(float(eis))
        for p in aircraft_tech_params(tech, "paper")
    })
    design = gam.design_airplane(
        dict(propulsion_architectures[prop]), _mission(prop, market)
    )
    return _scalars(design)


# Step-by-step change from the paper model to the update model: each step adds the
# update value of a group of technology parameters (or option) to the previous one.
UPDATE_STEPS = {
    "GAM V3.0 (paper technology)": (),
    "+ structure mass factor": ("struct_weight_factor",),
    "+ thermal engine efficiency": ("engine_efficiency_factor",),
    "+ fuel cell efficiency": (
        "fuelcell_efficiency",
        "fuelcell_rated_efficiency_ratio",
        "fuelcell_efficiency_exponent",
    ),
    # the update specific power excludes the heat rejection, sized separately
    "+ fuel cell specific power and TMS mass": (
        "fuelcell_specific_power",
        "fuelcell_power_exponent",
        "fuelcell_tms_heat_rejection",
    ),
    "+ TMS parasitic power": ("fuelcell_tms_power_loss",),
    "+ LH2 tank size law": ("lh2tank_mass_factor", "lh2_fuel_system_ratio"),
    "+ e-motors, electronics, scale laws": (
        "emotor_specific_power",
        "emotor_efficiency",
        "electronics_specific_power",
        "emotor_power_exponent",
        "emotor_loss_exponent",
        "battery_specific_energy",
        "reference_unit_power",
    ),
    "+ climb at powertrain efficiency": ("options",),
    "+ propulsor count (update)": ("engine_count",),
}
STEP_ARCHITECTURES = {
    "JetA-GasTurbine": "Jet-A turbofan",
    "lH2-GasTurbine": "LH2 gas turbine",
    "lH2-FuelCell": "LH2 fuel cell",
}
STEP_EIS = 2045


def update_steps(prop, market, tech=1, eis=STEP_EIS):
    """Energy per seat-km and power per propulsor along the update steps."""
    rows = {"Paper (GAM V2.0)": design_paper(prop, market, tech, eis)}
    paper = {
        p.name: p.value_at_entry_into_service(float(eis))
        for p in aircraft_tech_params(tech, "paper")
    }
    update = {
        p.name: p.value_at_entry_into_service(float(eis))
        for p in aircraft_tech_params(tech, "update")
    }
    kwargs = dict(paper)
    options = {}
    power_system = dict(propulsion_architectures[prop])
    for step, keys in UPDATE_STEPS.items():
        for key in keys:
            if key == "options":
                options = dict(UPDATE_GAM_OPTIONS)
            elif key == "engine_count":
                power_system = update_power_system(prop, market)
            elif key in update:
                kwargs[key] = update[key]
                if key == "lh2tank_mass_factor":
                    kwargs.pop("lh2tank_gravimetric_index", None)
        gam = gam_v3.GAM(**kwargs, **options)
        rows[step] = _scalars(
            gam.design_airplane(dict(power_system), _mission(prop, market)),
            gam.max_unit_power,
        )
    return rows


def compute(partial=None):
    """All the designs of the study, resuming from a partial result.

    JAX compiles a program for every GAM instance; its caches are cleared after each
    design series, and the partial result is saved, to bound the memory.
    """
    result = partial or {"update": {}, "paper": {}, "zeroe": {}, "steps": {}}

    def done(block, key, function):
        if key not in result[block]:
            result[block][key] = function()
            jax.clear_caches()
            PARTIAL.write_text(json.dumps(result))

    for tech, index in TECHS.items():
        for prop in ARCHITECTURES:
            for market in MARKETS:
                key = f"{tech}|{prop}|{market}"
                done(
                    "update",
                    key,
                    lambda p=prop, m=market, i=index: {
                        str(y): design_update(p, m, i, y) for y in YEARS
                    },
                )
                done(
                    "paper",
                    key,
                    lambda p=prop, m=market, i=index: {
                        str(y): design_paper(p, m, i, y) for y in PAPER_YEARS
                    },
                )
            print("designed", tech, prop, flush=True)  # noqa: T201
        for name, (market, npax, design_range, engines) in ZEROE_CASES.items():
            done(
                "zeroe",
                f"{tech}|{name}",
                lambda m=market, i=index, n=npax, r=design_range, e=engines: {
                    str(y): design_update("lH2-FuelCell", m, i, y, n, r, e)
                    for y in YEARS
                },
            )
    for prop in STEP_ARCHITECTURES:
        for market in MARKETS:
            done(
                "steps", f"{prop}|{market}", lambda p=prop, m=market: update_steps(p, m)
            )
    return result


def load():
    """The cached study, computed first if needed."""
    if not CACHE.is_file():
        partial = json.loads(PARTIAL.read_text()) if PARTIAL.is_file() else None
        CACHE.write_text(json.dumps(compute(partial)))
        PARTIAL.unlink()
    return json.loads(CACHE.read_text())


CLOSURE_TOLERANCE = 1.0e-4


def series(block, key, output):
    """Years and values of an output of a design series.

    The values of the designs that do not close are NaN: they are meaningless.
    """
    rows = block[key]
    years = np.array([float(y) for y in rows])
    values = np.array([
        np.nan
        if r[output] is None
        or (
            output != "max_unit_power"
            and r["closure_gap"] is not None
            and r["closure_gap"] > CLOSURE_TOLERANCE
        )
        else r[output]
        for r in rows.values()
    ])
    return years, values


def crossing(years, values, threshold):
    """First year from which ``values <= threshold``, linearly interpolated.

    Returns ``None`` if never reached, and the first year if always reached.
    """
    ok = values <= threshold
    if ok[0]:
        return float(years[0])
    for i in range(1, len(years)):
        if ok[i]:
            y0, y1, v0, v1 = years[i - 1], years[i], values[i - 1], values[i]
            if not np.isfinite(v0):  # the previous design does not close
                return float(y1)
            return float(y0 + (threshold - v0) * (y1 - y0) / (v1 - v0))
    return None


def feasibility_years(study, tech, market):
    """First EIS satisfying each design constraint of the fuel cell aircraft.

    Returns the first EIS of the power limit (power per propulsor <= maximum), of
    the efficiency constraint (energy per seat-km <= 2019 fleet) and of both.
    """
    key = f"{tech}|lH2-FuelCell|{market}"
    years, ratio = series(study["update"], key, "unit_power_ratio")
    _, energy = series(study["update"], key, "energy_per_ask")
    fleet = category_conso[market][TECHS[tech]]
    power = crossing(years, ratio, 1.0)
    efficiency = crossing(years, energy / fleet, 1.0)
    both = crossing(years, np.maximum(ratio, energy / fleet), 1.0)
    return power, efficiency, both


def optimum_eis(model, prop, market, tech="Mid", base="SSP2-26"):
    """EIS of an aircraft in a pre-computed optimum, or None if not in the fleet."""
    suffix = {"Lower": "lowTech", "Mid": "midTech", "Upper": "upTech"}[tech]
    folder = RESULTS / ("update" if model == "update" else "") / f"{base}-{suffix}"
    with (folder / "opt_result.json").open() as file:
        inputs = json.load(file)["inputs"]
    value = inputs.get(f"constant.{prop}_{market}.entry_into_service")
    return None if value is None else float(np.ravel(value)[0])


if __name__ == "__main__":
    load()
