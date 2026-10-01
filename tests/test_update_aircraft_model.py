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
"""Tests of the update aircraft model: powertrain scale effects and LH2 tanks."""

import json
from pathlib import Path

import jax
import numpy as np
import pytest

from noads._data import data_file
from noads.application import lh2_tank_calibration as calibration
from noads.application.base_objects import aircraft_tech_params
from noads.application.base_objects import fuelcell_engine_count
from noads.application.base_objects import initialize_base_objects
from noads.application.base_objects import lh2_tank_tech_params_lower_mid_upper
from noads.application.base_objects import tech_params_lower_mid_upper_2020_2040_2060
from noads.application.base_objects import (
    update_tech_params_lower_mid_upper_2020_2040_2060_2080,
)
from noads.application.scenario_setup import single_scenario_setup
from noads.core.models.fleet.aircraft_design import AircraftDesign
from noads.core.models.fleet.aircraft_tech_parameter import AircraftTechParameter
from noads.core.models.fleet.aircraft_tech_parameter import LogisticTechParameter
from noads.core.models.fleet.aircraft_tech_parameter import ScenarioTechParameter
from noads.gam_jax.models.gam_v3 import GAM
from noads.gam_jax.models.gam_v3 import LH2_TANK_SIZE_LAW

FC_SYSTEM = {
    "engine_count": 4,
    "engine_type": "emotor",
    "thruster_type": "propeller",
    "energy_type": "liquid_h2",
}
LH2_TURBOFAN = {
    "engine_count": 2,
    "engine_type": "turbofan",
    "thruster_type": "fan",
    "energy_type": "liquid_h2",
    "bpr": 12.0,
}
REGIONAL = {
    "npax": 80,
    "range": 4500e3,
    "speed": 170.0,
    "altitude": 6096.0,
    "category": "regional",
}
SHORT_MEDIUM = {
    "npax": 120,
    "range": 8000e3,
    "speed": 272.0,
    "altitude": 8534.4,
    "category": "short_medium",
}
TECH = {
    "emotor_specific_power": 15.0,
    "electronics_specific_power": 20.0,
    "fuelcell_specific_power": 3.5,
    "fuelcell_efficiency": 50.0,
    "lh2tank_gravimetric_index": 50.0,
    "struct_weight_factor": 78.0,
}
SCENARIOS = ("lower", "mid", "upper")


# ------------------------------------------------------------------------------
# Powertrain scaling (ported from the powertrain scaling patch)
# ------------------------------------------------------------------------------
def test_default_arguments_are_legacy():
    """Explicit legacy values give the same design as the defaults."""
    legacy = GAM(
        **TECH,
        emotor_efficiency=90.0,
        emotor_power_exponent=0.0,
        emotor_loss_exponent=0.0,
        fuelcell_tms_heat_rejection=None,
        fuelcell_power_exponent=0.0,
        fuelcell_efficiency_exponent=0.0,
    ).design_airplane(dict(FC_SYSTEM), dict(REGIONAL))
    default = GAM(**TECH).design_airplane(dict(FC_SYSTEM), dict(REGIONAL))
    assert float(legacy["mtow"]) == pytest.approx(float(default["mtow"]))
    assert float(default["electric_chain_efficiency"]) == pytest.approx(0.9)


def test_engine_count_only_matters_with_scale_laws():
    """Without scale laws the propulsor count has no effect on the design."""
    mtow = []
    for n in (2, 8):
        system = {**FC_SYSTEM, "engine_count": n}
        mtow.append(float(GAM(**TECH).design_airplane(system, dict(REGIONAL))["mtow"]))
    assert mtow[0] == pytest.approx(mtow[1])

    motor_mass = []
    for n in (2, 8):
        gam = GAM(**TECH, emotor_power_exponent=0.25)
        system = {**FC_SYSTEM, "energy_type": "battery", "engine_count": n}
        motor_mass.append(float(gam.propulsion_mass(system, 4.0e6)[0]))
    assert motor_mass[1] < motor_mass[0]


def test_scale_laws():
    gam = GAM(
        **TECH,
        emotor_power_exponent=0.25,
        emotor_loss_exponent=0.25,
        emotor_efficiency=95.0,
    )
    ref = gam.reference_unit_power
    assert float(gam.emotor_power_density(16 * ref)) == pytest.approx(
        0.5 * float(gam.emotor_power_density(ref))
    )
    assert 1.0 - float(gam.electric_chain_efficiency(16 * ref)) == pytest.approx(
        0.5 * 0.05
    )


def test_fuel_cell_tms_sizing():
    """Gross power, heat load and mass follow the closed-form sizing."""
    gam = GAM(
        **TECH,
        fuelcell_tms_heat_rejection=10.0,
        fuelcell_tms_power_loss=0.1,
        fuelcell_rated_efficiency_ratio=80.0,
    )
    total_power = 4.0e6
    gross, heat, mass = gam.fuel_cell_sizing(FC_SYSTEM, total_power)
    eta_rated = 0.5 * 0.8
    heat_to_power = 1.0 / eta_rated - 1.0
    expected_gross = total_power / 0.9 / (1.0 - 0.1 * heat_to_power)
    assert float(gross) == pytest.approx(expected_gross)
    assert float(heat) == pytest.approx(expected_gross * heat_to_power)
    assert float(mass) == pytest.approx(
        expected_gross / 3500.0 + expected_gross * heat_to_power / 10.0e3
    )
    assert float(gam.fuel_cell_mission_efficiency(1.0e6)) == pytest.approx(
        0.5 - 0.1 * 0.5
    )


def test_unit_power_ratio():
    design = GAM(**TECH, max_unit_power=2.0).design_airplane(
        dict(FC_SYSTEM), dict(REGIONAL)
    )
    assert float(design["unit_power_ratio"]) == pytest.approx(
        float(design["max_power"]) / 2.0e6
    )


def test_log_scale_tech_parameter():
    parameter = AircraftTechParameter("p", (0.06, 3.0, 12.0), log_scale=True)
    values = np.asarray(
        parameter.value_at_entry_into_service(np.linspace(2020.0, 2060.0, 41))
    )
    np.testing.assert_allclose(values[[0, 20, 40]], [0.06, 3.0, 12.0], rtol=1e-6)
    assert np.all(np.diff(values) > 0.0)


UPDATE_PARAMS = {
    param.name: [
        {p.name: p for p in aircraft_tech_params(index, "update")}[param.name]
        for index in range(3)
    ]
    for param in aircraft_tech_params(1, "update")
}
EIS = np.linspace(2020.0, 2100.0, 321)


@pytest.mark.parametrize("name", list(UPDATE_PARAMS))
def test_scenario_band_starts_at_zero_and_widens(name):
    """Scenarios share their 2020 value, and Upper-to-Lower only widens."""
    lower, mid, upper = (
        np.asarray(param.value_at_entry_into_service(EIS))
        for param in UPDATE_PARAMS[name]
    )
    sign = 1.0 if upper[-1] >= lower[-1] else -1.0
    gap = sign * (upper - lower)
    assert gap[0] == pytest.approx(0.0, abs=1e-9)
    assert np.all(np.diff(gap) >= -1e-9)
    assert np.all(sign * (mid - lower) >= -1e-9)
    assert np.all(sign * (upper - mid) >= -1e-9)


@pytest.mark.parametrize("name", list(UPDATE_PARAMS))
def test_scenario_curves_are_monotonic(name):
    for param in UPDATE_PARAMS[name]:
        values = np.asarray(param.value_at_entry_into_service(EIS))
        steps = np.diff(values)
        assert np.all(steps >= -1e-9) or np.all(steps <= 1e-9)


def test_scenario_tech_parameter_rejects_inconsistent_scenarios():
    with pytest.raises(ValueError, match="2020 value"):
        ScenarioTechParameter("p", ((1, 2, 3, 4), (2, 3, 4, 5), (2, 4, 5, 6)), 0)
    with pytest.raises(ValueError, match="must not shrink"):
        ScenarioTechParameter("p", ((1, 2, 3, 4), (1, 3, 4, 5), (1, 5, 5, 6)), 0)


def test_update_params_match_calibration():
    """2040 and 2060 values are those of the calibrations, unless revised."""
    calibrated = json.loads(
        Path(
            data_file(
                "noads.application",
                "aircraft_tech_data",
                "powertrain",
                "calibrated_params.json",
            )
        ).read_text()
    )
    names = {
        "emotor_specific_power": "emotor_specific_power",
        "emotor_efficiency": "electric_chain_efficiency",
        "max_unit_power": "max_unit_power_MW",
    }
    # the fuel cell and TMS values are revised from the 2026 literature review
    revised = {
        "fuelcell_specific_power",
        "fuelcell_efficiency",
        "fuelcell_tms_heat_rejection",
        "fuelcell_tms_power_loss",
    }
    params = update_tech_params_lower_mid_upper_2020_2040_2060_2080
    for name, calibrated_name in names.items():
        for index, scenario in enumerate(SCENARIOS):
            expected = calibrated[scenario][calibrated_name][1:]
            np.testing.assert_allclose(params[name][index][1:3], expected)
    # the other parameters are those of the paper
    for name, values in tech_params_lower_mid_upper_2020_2040_2060.items():
        if name in params and name not in names and name not in revised:
            for index in range(3):
                np.testing.assert_allclose(params[name][index][1:3], values[index][1:])


# ------------------------------------------------------------------------------
# LH2 tanks
# ------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def fitted():
    return calibration.calibrate()


def test_calibration_reproduces_hard_coded_values(fitted):
    np.testing.assert_allclose(fitted.size_law, LH2_TANK_SIZE_LAW, atol=1e-4)
    for index, scenario in enumerate(SCENARIOS):
        np.testing.assert_allclose(
            fitted.mass_factor[scenario],
            lh2_tank_tech_params_lower_mid_upper["lh2tank_mass_factor"][index],
            rtol=1e-3,
            atol=0.05,
        )
        np.testing.assert_allclose(
            fitted.fuel_system_ratio[scenario],
            lh2_tank_tech_params_lower_mid_upper["lh2_fuel_system_ratio"][index],
            rtol=1e-3,
            atol=0.05,
        )


@pytest.mark.parametrize("index", [0, 1, 2])
def test_tank_gi_monotonic(index):
    """GI_tank increases with tank size and does not decrease with EIS."""
    k = LogisticTechParameter(
        "k", lh2_tank_tech_params_lower_mid_upper["lh2tank_mass_factor"][index]
    )
    masses = np.geomspace(10.0, 5.0e4, 30)
    years = np.linspace(2020.0, 2100.0, 81)
    a, b, c = LH2_TANK_SIZE_LAW
    ratio = np.asarray(k.value_at_entry_into_service(years))[:, None] * (
        a + b * masses[None, :] ** (-1.0 / 3.0) + c / masses[None, :]
    )
    gi = 1.0 / (1.0 + ratio)
    assert np.all(np.diff(gi, axis=1) > 0.0)
    assert np.all(np.diff(gi, axis=0) >= 0.0)


def test_legacy_constant_gi_is_reproduced():
    """b = c = s = 0 with 1 / (1 + k a) = GI gives the constant-GI design."""
    legacy = GAM(**TECH).design_airplane(dict(LH2_TURBOFAN), dict(SHORT_MEDIUM))
    sized = GAM(
        **TECH,
        lh2tank_mass_factor=1.0,
        lh2tank_size_law=(1.0 / 0.5 - 1.0, 0.0, 0.0),
    ).design_airplane(dict(LH2_TURBOFAN), dict(SHORT_MEDIUM))
    for name in ("mtow", "owe", "energy_storage_mass", "mission_fuel"):
        assert float(sized[name]) == pytest.approx(float(legacy[name]), rel=1e-8)
    assert float(sized["gi_tank"]) == pytest.approx(0.5)


def test_flyzero_narrowbody_2050(fitted):
    """The Upper model reaches the FlyZero narrowbody 2050 tanks within 0.05.

    Both aft tanks hold the same (estimated) LH2 mass, so the size law is compared
    with their mean GI.
    """
    df = calibration.load_dataset()
    rows = df[df.id.str.startswith("FZ-narrowbody-afttank") & df.tech_year.eq(2050)]
    gi = calibration.tank_gravimetric_index(
        fitted, "upper", rows.m.iloc[0], 2050.0 + calibration.FLYZERO_DELAY
    )
    assert gi == pytest.approx(rows.gi.mean(), abs=0.05)


def test_tank_count_changes_gi():
    """Splitting the fuel into more, smaller tanks lowers the tank GI."""
    gi = []
    for count in (1, 4):
        design = GAM(**TECH, lh2tank_mass_factor=1.0).design_airplane(
            {**LH2_TURBOFAN, "tank_count": count}, dict(SHORT_MEDIUM)
        )
        gi.append(float(design["gi_tank"]))
    assert gi[1] < gi[0]


# ------------------------------------------------------------------------------
# Application wiring
# ------------------------------------------------------------------------------
def test_paper_tech_params_unchanged():
    names = [param.name for param in aircraft_tech_params(1)]
    assert "lh2tank_gravimetric_index" in names
    update = {param.name: param for param in aircraft_tech_params(1, "update")}
    assert "lh2tank_gravimetric_index" not in update
    assert isinstance(update["lh2tank_mass_factor"], LogisticTechParameter)
    assert update["max_unit_power"].log_scale


def test_base_objects_with_update_model():
    _, fleet = initialize_base_objects(technology_index=1, aircraft_model="update")
    designs = {
        aircraft.name: aircraft
        for aircraft in fleet.operating_aircraft
        if isinstance(aircraft, AircraftDesign)
    }
    for category, count in fuelcell_engine_count.items():
        design = designs[f"lH2-FuelCell_{category}"]
        assert design.aircraft_model == "update"
        assert design.power_system["engine_count"] == count
        assert design.power_system["tank_count"] == 2
        names = {parameter.name for parameter in design.technology_evolution}
        assert {"max_unit_power", "fuelcell_tms_heat_rejection"} <= names
    assert designs["JetA-GasTurbine-v1_regional"].power_system["engine_count"] == 2


def test_unknown_aircraft_model():
    with pytest.raises(ValueError, match="aircraft_model"):
        initialize_base_objects(aircraft_model="v3")


@pytest.mark.parametrize(
    "name",
    ["lH2-FuelCell_regional", "lH2-FuelCell_general", "lH2-GasTurbine_long_range"],
)
def test_design_outputs_and_eis_gradient(name):
    """The update outputs are differentiable in the EIS."""
    _, fleet = initialize_base_objects(technology_index=1, aircraft_model="update")
    design = next(
        aircraft for aircraft in fleet.operating_aircraft if aircraft.name == name
    )

    def outputs(eis):
        return design._gam_design({f"{name}.entry_into_service": eis})

    data = outputs(2045.0)
    assert 0.0 < float(data[f"{name}.gi_tank"]) < 1.0
    for output in ("energy_per_ask", "unit_power_ratio", "gi_tank"):
        gradient = float(
            jax.grad(lambda eis, o=output: outputs(eis)[f"{name}.{o}"])(2045.0)
        )
        assert np.isfinite(gradient), output
    assert (
        float(jax.grad(lambda eis: outputs(eis)[f"{name}.unit_power_ratio"])(2045.0))
        < 0.0
    )


def test_non_closing_design_is_nan():
    """An immature battery aircraft does not close: NaN instead of an error."""
    tech = {
        param.name: param.value_at_entry_into_service(2030.0)
        for param in aircraft_tech_params(0, "update")
    }
    design = GAM(**tech).design_airplane(
        {
            "engine_count": 2,
            "engine_type": "emotor",
            "thruster_type": "propeller",
            "energy_type": "battery",
        },
        {"npax": 19, "range": 500e3, "speed": 170.0, "category": "general"},
    )
    assert np.isnan(float(design["mtow"]))


def test_scenario_setup_unit_power_constraints():
    _, _, constraints, _, fleet = single_scenario_setup(
        "update",
        "SSP2-26",
        technology_index=1,
        aircraft_model="update",
        compile_jit=False,
    )
    for aircraft in fleet.operating_aircraft:
        name = f"{aircraft.name}.unit_power_ratio"
        if "FuelCell" in aircraft.name or "Electric" in aircraft.name:
            assert constraints[name] == (1.0, False)
        else:
            assert name not in constraints
