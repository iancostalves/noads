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

"""Design performance of the new aircraft of the main paper and of the update.

Every new aircraft of the fleets of both aircraft models is designed (GAM V2.0 for the
paper, GAM V3.0 for the update) for several years of entry-into-service and the three
technology scenarios. The results are cached in ``design_sweep.json``.

Usage: ``python design_sweep.py`` (recomputes the cache if missing).
"""

import json
from pathlib import Path

import numpy as np

from noads.application.base_objects import category_conso
from noads.application.base_objects import initialize_base_objects

CACHE = Path(__file__).with_name("design_sweep.json")
EIS_YEARS = [2035.0, 2040.0, 2050.0, 2060.0]
TECHS = {"Lower": 0, "Mid": 1, "Upper": 2}


def compute():
    """Design every aircraft: {model: {tech: {aircraft: {eis: outputs}}}}."""
    result = {}
    for model in ("paper", "update"):
        result[model] = {}
        for tech, index in TECHS.items():
            _, fleet = initialize_base_objects(
                technology_index=index, aircraft_model=model
            )
            result[model][tech] = {"fleet_energy": {}, "designs": {}}
            for fleet_i in fleet.fleets:
                result[model][tech]["fleet_energy"][fleet_i.name] = float(
                    category_conso[fleet_i.name][index]
                )
                for aircraft in fleet_i.operating_aircraft:
                    if not hasattr(aircraft, "_gam_design"):
                        continue
                    rows = {}
                    for eis in EIS_YEARS:
                        out = aircraft._gam_design({
                            f"{aircraft.name}.entry_into_service": np.array(eis)
                        })
                        row = {
                            key: float(out[f"{aircraft.name}.{key}"])
                            for key in ("energy_per_ask", "mtow", "owe")
                        }
                        for key in ("unit_power_ratio", "closure_gap"):
                            if f"{aircraft.name}.{key}" in out:
                                row[key] = float(out[f"{aircraft.name}.{key}"])
                        rows[str(int(eis))] = row
                    result[model][tech]["designs"][aircraft.name] = rows
            print(model, tech, len(result[model][tech]["designs"]), flush=True)  # noqa: T201
    return result


def load():
    """The cached design sweep, computed first if needed."""
    if not CACHE.is_file():
        CACHE.write_text(json.dumps(compute()))
    return json.loads(CACHE.read_text())


if __name__ == "__main__":
    load()
