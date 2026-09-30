# Copyright 2025 ISAE-SUPAERO, https://www.isae-supaero.fr/en/
# Copyright 2025 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""Reference designs of the paper aircraft model (GAM V2.0 port).

Run this module to regenerate ``data/gam_paper_baseline.json``. It must only be
regenerated on purpose: the paper results depend on these designs.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax

from noads.application.base_objects import categories_mission
from noads.application.base_objects import propulsion_architectures
from noads.application.base_objects import propulsion_mission
from noads.application.base_objects import tech_params_lower_mid_upper_2020_2040_2060
from noads.core.models.fleet.aircraft_design import AircraftDesign
from noads.core.models.fleet.aircraft_operation import AircraftOperation
from noads.core.models.fleet.aircraft_operation import PropulsionSystem
from noads.core.models.fleet.aircraft_tech_parameter import AircraftTechParameter

BASELINE_FILE = Path(__file__).parent / "data" / "gam_paper_baseline.json"

TECHNOLOGY_INDICES = (0, 1, 2)
ENTRY_INTO_SERVICE = (2025.0, 2040.0, 2060.0)
OUTPUTS = ("energy_per_ask", "mission_energy", "energy_mass", "mtow", "owe")


def _design(category, architecture, technology_index):
    tech_params = [
        AircraftTechParameter(name, tuple(values[technology_index]))
        for name, values in tech_params_lower_mid_upper_2020_2040_2060.items()
    ]
    reference = AircraftOperation(
        "ref", PropulsionSystem("ref", {}), energy_per_ask=1.0
    )
    return AircraftDesign(
        name="design",
        propulsion=PropulsionSystem("prop", {}),
        mission={
            **categories_mission[category],
            **propulsion_mission[architecture],
            "category": category,
        },
        power_system=dict(propulsion_architectures[architecture]),
        aircraft_tech_params=tech_params,
        reference_aircraft=reference,
    )


def compute_baseline():
    """Compute the design outputs and d(energy_per_ask)/d(EIS) over the grid."""
    results = {}
    for category in categories_mission:
        for architecture in propulsion_architectures:
            for index in TECHNOLOGY_INDICES:
                design = _design(category, architecture, index)

                def outputs(eis, design=design):
                    data = design._gam_design({"design.entry_into_service": eis})
                    return {name: data[f"design.{name}"] for name in OUTPUTS}

                gradient = jax.jit(
                    jax.grad(lambda eis, f=outputs: f(eis)["energy_per_ask"])
                )
                values = jax.jit(outputs)
                for eis in ENTRY_INTO_SERVICE:
                    key = f"{category}|{architecture}|{index}|{eis:.0f}"
                    entry = {name: float(v) for name, v in values(eis).items()}
                    entry["d_energy_per_ask_d_eis"] = float(gradient(eis))
                    results[key] = entry
    return results


if __name__ == "__main__":
    BASELINE_FILE.parent.mkdir(exist_ok=True)
    BASELINE_FILE.write_text(json.dumps(compute_baseline(), indent=1, sort_keys=True))
