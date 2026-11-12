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
"""Tests of the JAX port of GAM V3.0 against the upstream numpy implementation."""

import json
from pathlib import Path

import jax
import numpy as np
import pytest

from noads.gam_jax.models.gam_v3 import GAM

REFERENCE = json.loads(
    (Path(__file__).parent / "data" / "gam_v3_reference.json").read_text()
)
QUANTITIES = (
    "mtow",
    "owe",
    "mission_fuel",
    "mission_enrg",
    "enrg_consumption",
    "energy_storage_mass",
    "fuel_cell_system_mass",
    "payload_max",
    "propulsion_system_efficiency",
)


@pytest.mark.parametrize("key", list(REFERENCE["designs"]))
def test_matches_upstream(key):
    """Designs match upstream GAM V3.0 (scipy fsolve) on the NOADS missions."""
    reference = REFERENCE["designs"][key]
    design = GAM(**REFERENCE["tech"]).design_airplane(
        dict(reference["power_system"]), dict(reference["mission"])
    )
    for quantity in QUANTITIES:
        assert float(design[quantity]) == pytest.approx(
            reference[quantity], rel=1e-8, abs=1e-8
        ), quantity


def test_turboprop_efficiency_increases_with_power():
    gam = GAM()
    powers = np.array([2.0e5, 1.0e6, 5.0e6])
    efficiencies = np.asarray(gam.get_turboprop_eff(powers))
    assert np.all(np.diff(efficiencies) > 0.0)


@pytest.mark.parametrize(
    "key",
    ["general|lH2-FuelCell", "regional|JetA-Turboprop", "long_range|lH2-GasTurbine"],
)
def test_gradient_with_respect_to_technology(key):
    """Designs are differentiable with respect to the technology parameters."""
    reference = REFERENCE["designs"][key]

    def energy(struct_weight_factor):
        tech = {**REFERENCE["tech"], "struct_weight_factor": struct_weight_factor}
        return GAM(**tech).design_airplane(
            dict(reference["power_system"]), dict(reference["mission"])
        )["enrg_consumption"]

    gradient = float(jax.grad(energy)(78.0))
    assert np.isfinite(gradient)
    assert gradient > 0.0


def test_category_guess():
    design = GAM().design_airplane(
        {
            "engine_count": 2,
            "engine_type": "turbofan",
            "thruster_type": "fan",
            "energy_type": "kerosene",
            "bpr": 9.0,
        },
        {"npax": 150, "range": 5000e3},
    )
    assert design["airplane_type"] == "short_medium"
    assert float(design["cruise_speed"]) == pytest.approx(0.78)
