# ruff: noqa: E501, INP001, D103, E401, PTH123, SIM115, BLE001, T201, E702, E731
"""Run upstream numpy GAM V3.0 on the NOADS missions to build a reference.

Usage: python make_gam_v3_reference.py <clone of gitlab.com/m6029/genericairplanemodel> <output json>

Used to generate ``tests/data/gam_v3_reference.json`` (upstream commit 0b50bf5d).
"""

import contextlib
import io
import json
import sys

from gam_v3 import GAM
from utils import unit

sys.path[:0] = [sys.argv[1], sys.argv[1] + "/gam"]

MISSIONS = {
    "general": {"npax": 19, "range": 500e3},
    "commuter": {"npax": 50, "range": 1500e3},
    "regional": {"npax": 80, "range": 4500e3},
    "short_medium": {"npax": 120, "range": 8000e3},
    "long_range": {"npax": 250, "range": 15000e3},
}
FAN = {"speed": 0.8 * 340, "altitude": 28000 * 0.3048}
PROP = {"speed": 0.5 * 340, "altitude": 20000 * 0.3048}
ARCH = {
    "JetA-GasTurbine": (
        {
            "engine_count": 2,
            "engine_type": "turbofan",
            "thruster_type": "fan",
            "energy_type": "kerosene",
            "bpr": 12.0,
        },
        FAN,
    ),
    "lH2-GasTurbine": (
        {
            "engine_count": 2,
            "engine_type": "turbofan",
            "thruster_type": "fan",
            "energy_type": "liquid_h2",
            "bpr": 12.0,
        },
        FAN,
    ),
    "lH2-FuelCell": (
        {
            "engine_count": 2,
            "engine_type": "emotor",
            "thruster_type": "propeller",
            "energy_type": "liquid_h2",
        },
        PROP,
    ),
    "Battery-Electric": (
        {
            "engine_count": 2,
            "engine_type": "emotor",
            "thruster_type": "propeller",
            "energy_type": "battery",
        },
        PROP,
    ),
    "JetA-Turboprop": (
        {
            "engine_count": 2,
            "engine_type": "turboprop",
            "thruster_type": "propeller",
            "energy_type": "kerosene",
        },
        PROP,
    ),
    "lH2-Turboprop": (
        {
            "engine_count": 2,
            "engine_type": "turboprop",
            "thruster_type": "propeller",
            "energy_type": "liquid_h2",
        },
        PROP,
    ),
}
TECH = {
    "battery_specific_energy": 575.0,
    "emotor_specific_power": 15.0,
    "electronics_specific_power": 20.0,
    "fuelcell_specific_power": 2.5,
    "lh2tank_gravimetric_index": 47.5,
    "fuelcell_efficiency": 50.0,
    "struct_weight_factor": 78.0,
}


def upstream():
    with contextlib.redirect_stdout(io.StringIO()):
        gam = GAM()
    gam.battery_enrg_density = unit.J_Wh(TECH["battery_specific_energy"])
    gam.power_density[gam.emotor] = unit.W_kW(TECH["emotor_specific_power"])
    gam.power_density[gam.power_elec] = unit.W_kW(TECH["electronics_specific_power"])
    gam.power_density[gam.fuel_cell] = unit.W_kW(TECH["fuelcell_specific_power"])
    gam.lh2_tank_gravimetric_index = 1e-2 * TECH["lh2tank_gravimetric_index"]
    gam.fuel_cell_eff = 1e-2 * TECH["fuelcell_efficiency"]
    gam.stdm_factor = 1e-2 * TECH["struct_weight_factor"]
    return gam


out = {"tech": TECH, "designs": {}}
KEYS = [
    "mtow",
    "owe",
    "mission_fuel",
    "total_fuel",
    "mission_enrg",
    "enrg_consumption",
    "total_power",
    "energy_storage_mass",
    "fuel_cell_system_mass",
    "propulsion_mass",
    "payload_max",
    "propulsion_system_efficiency",
]
for cat, m in MISSIONS.items():
    for name, (ps, extra) in ARCH.items():
        mission = {**m, **extra, "category": cat}
        try:
            d = upstream().design_airplane(dict(ps), dict(mission))
            out["designs"][f"{cat}|{name}"] = {
                "power_system": ps,
                "mission": mission,
                **{k: float(d[k]) for k in KEYS},
            }
        except Exception:
            pass
json.dump(out, open(sys.argv[2], "w"), indent=1)
