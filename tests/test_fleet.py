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

import numpy as np
import pytest

from noads.application.base_objects import categories_mission
from noads.application.base_objects import propulsion_architectures
from noads.application.base_objects import propulsion_mission
from noads.application.base_objects import tech_params_lower_mid_upper_2020_2040_2060
from noads.core.models.fleet.aircraft_design import AircraftDesign
from noads.core.models.fleet.aircraft_operation import AircraftOperation
from noads.core.models.fleet.aircraft_operation import PropulsionSystem
from noads.core.models.fleet.aircraft_tech_parameter import AircraftTechParameter
from noads.core.models.fleet.fleet import Fleet
from noads.core.models.fleet.fleet import FleetAssembly


@pytest.fixture
def hybrid_aircraft(kerosene, battery):
    hybrid = PropulsionSystem("hybrid", {kerosene: 0.7, battery: 0.3})
    return AircraftOperation(
        "Hyb", hybrid, energy_per_ask=1.0, lifetime=25.0, recent=False
    )


def test_aircraft_tech_parameter():
    parameter = AircraftTechParameter("p", (1.0, 2.0, 4.0))
    assert parameter.name == "p"
    assert float(parameter.value_at_entry_into_service(2020.0)) == pytest.approx(1.0)
    assert float(parameter.value_at_entry_into_service(2040.0)) == pytest.approx(2.0)
    assert float(parameter.value_at_entry_into_service(2060.0)) == pytest.approx(4.0)
    assert 1.0 < float(parameter.value_at_entry_into_service(2030.0)) < 2.0

    values = parameter.value_at_entry_into_service(np.array([2020.0, 2060.0]))
    np.testing.assert_allclose(values, [1.0, 4.0], atol=1e-9)


def test_propulsion_system(kerosene, battery):
    propulsion = PropulsionSystem("hybrid", {kerosene: 0.7, battery: 0.3})
    assert propulsion.name == "hybrid"
    assert propulsion.energy_carrier_mix[kerosene] == pytest.approx(0.7)
    assert propulsion.energy_carrier_mix[battery] == pytest.approx(0.3)


def test_aircraft_operation_models_property(old_aircraft, new_aircraft):
    assert len(old_aircraft.models) == 1  # recent: consumption only
    assert len(new_aircraft.models) == 2  # share + consumption


def test_aircraft_operation_share_model(new_aircraft, execute_model):
    model = new_aircraft.share_model()
    eis = 2000.0
    overrides = {"New.entry_into_service": eis, "New.max_share": 0.4}

    # At the inflection point (eis + lifetime / 2) the sigmoid is half-way.
    at_inflection = execute_model(model, {**overrides, "year": eis + 12.5})
    assert float(at_inflection["New.share"]) == pytest.approx(0.2, rel=1e-9)

    # Far beyond introduction the share saturates at max_share.
    saturated = execute_model(model, {**overrides, "year": eis + 100.0})
    assert float(saturated["New.share"]) == pytest.approx(0.4, rel=1e-6)

    # Below the sigmoid cut-off the share is clamped to zero.
    before = execute_model(model, {**overrides, "year": eis - 20.0})
    assert float(before["New.share"]) == pytest.approx(0.0)


def test_aircraft_operation_consumption_model(hybrid_aircraft, execute_model):
    model = hybrid_aircraft.consumption_model()
    output_data = execute_model(model, {"Hyb.ask": 10.0})
    assert float(output_data["Hyb.KEROSENE.consumption"]) == pytest.approx(7.0)
    assert float(output_data["Hyb.BATTERY.consumption"]) == pytest.approx(3.0)


def test_fleet_models(old_aircraft, new_aircraft, hybrid_aircraft, execute_model):
    fleet = Fleet("cat", [old_aircraft, new_aircraft])
    assert len(fleet.models) == 8  # 5 fleet models + 1 (recent) + 2 aircraft models

    last_share = execute_model(fleet.last_share_model(), {"New.share": 0.3})
    assert float(last_share["Old.share"]) == pytest.approx(0.7)

    ask = execute_model(
        fleet.ask_model(), {"cat.ask": 10.0, "Old.share": 0.7, "New.share": 0.3}
    )
    assert float(ask["Old.ask"]) == pytest.approx(7.0)
    assert float(ask["New.ask"]) == pytest.approx(3.0)

    # Aircraft not consuming a carrier contribute zero to its aggregation.
    mixed = Fleet("mix", [old_aircraft, hybrid_aircraft])
    consumption = execute_model(
        mixed.consumption_model(),
        {
            "Old.KEROSENE.consumption": 5.0,
            "Hyb.KEROSENE.consumption": 3.0,
            "Hyb.BATTERY.consumption": 2.0,
        },
    )
    assert float(consumption["mix.KEROSENE.consumption"]) == pytest.approx(8.0)
    assert float(consumption["mix.BATTERY.consumption"]) == pytest.approx(2.0)

    mean = execute_model(
        mixed.mean_consumption_impacts_model(),
        {
            "mix.ask": 10.0,
            "mix.KEROSENE.consumption": 8.0,
            "mix.BATTERY.consumption": 2.0,
        },
    )
    assert float(mean["mix.KEROSENE.mean_consumption_per_ask"]) == pytest.approx(0.8)
    assert float(mean["mix.BATTERY.mean_consumption_per_ask"]) == pytest.approx(0.2)
    assert float(mean["mix.mean_energy_per_ask"]) == pytest.approx(1.0)

    avoidance = execute_model(
        fleet.demand_avoidance_model(),
        {
            "cat.demand_shift_ratio": 0.2,
            "cat.ask_trend": 100.0,
            "year": 2030.0,
            "discount_rate": 0.03,
            "start_year": 2025.0,
            "price_elasticity": -0.5,
        },
    )
    discount_factor = 1.03 ** (2025.0 - 2030.0)
    assert float(avoidance["cat.ask_avoided"]) == pytest.approx(20.0)
    assert float(avoidance["cat.ask"]) == pytest.approx(80.0)
    assert float(avoidance["cat.relative_price_change"]) == pytest.approx(0.8**-2)
    assert float(avoidance["cat.discounted_relative_price_change"]) == pytest.approx(
        0.8**-2 * discount_factor
    )
    assert float(avoidance["cat.discounted_ask_avoided"]) == pytest.approx(
        20.0 * discount_factor
    )


def test_fleet_assembly(
    old_aircraft, hybrid_aircraft, kerosene, battery, execute_model
):
    fleet_a = Fleet("A", [old_aircraft])
    fleet_b = Fleet("B", [hybrid_aircraft])
    assembly = FleetAssembly([fleet_a, fleet_b])

    assert assembly.name == "Global fleet"
    assert set(assembly.consumed_carriers) == {kerosene, battery}
    assert len(assembly.models) == 5 + len(fleet_a.models) + len(fleet_b.models)

    last_share = execute_model(assembly.last_share_model(), {"B.share": 0.4})
    assert float(last_share["A.share"]) == pytest.approx(0.6)

    ask = execute_model(
        assembly.ask_model(), {"ask_trend": 100.0, "A.share": 0.6, "B.share": 0.4}
    )
    assert float(ask["A.ask_trend"]) == pytest.approx(60.0)
    assert float(ask["B.ask_trend"]) == pytest.approx(40.0)

    consumption = execute_model(
        assembly.consumption_model(),
        {
            "A.KEROSENE.consumption": 5.0,
            "B.KEROSENE.consumption": 3.0,
            "B.BATTERY.consumption": 2.0,
        },
    )
    assert float(consumption["KEROSENE.consumption"]) == pytest.approx(8.0)
    assert float(consumption["BATTERY.consumption"]) == pytest.approx(2.0)

    mean = execute_model(
        assembly.mean_consumption_impacts_model(),
        {"ask": 10.0, "KEROSENE.consumption": 8.0, "BATTERY.consumption": 2.0},
    )
    assert float(mean["KEROSENE.mean_consumption_per_ask"]) == pytest.approx(0.8)
    assert float(mean["mean_energy_per_ask"]) == pytest.approx(1.0)

    avoidance = execute_model(
        assembly.demand_avoidance_model(),
        {
            "ask_trend": 100.0,
            "year": 2030.0,
            "discount_rate": 0.03,
            "start_year": 2025.0,
            "load_factor": 80.0,
            "A.ask": 40.0,
            "B.ask": 30.0,
            "A.ask_trend": 60.0,
            "B.ask_trend": 40.0,
            "A.relative_price_change": 1.2,
            "B.relative_price_change": 1.5,
        },
    )
    discount_factor = 1.03 ** (2025.0 - 2030.0)
    assert float(avoidance["ask"]) == pytest.approx(70.0)
    assert float(avoidance["ask_avoided"]) == pytest.approx(30.0)
    assert float(avoidance["rpk"]) == pytest.approx(56.0)
    assert float(avoidance["relative_price_change"]) == pytest.approx(1.32)
    assert float(avoidance["discount_factor"]) == pytest.approx(discount_factor)
    assert float(avoidance["discounted_relative_price_change"]) == pytest.approx(
        1.32 * discount_factor
    )
    assert float(avoidance["discounted_ask_avoided"]) == pytest.approx(
        30.0 * discount_factor
    )
    assert float(avoidance["discounted_ask_ratio"]) == pytest.approx(
        0.3 * discount_factor
    )


def test_aircraft_design_control_model(turbofan, old_aircraft, execute_model):
    design = AircraftDesign(
        name="Des",
        propulsion=turbofan,
        mission={},
        power_system={},
        aircraft_tech_params=[],
        reference_aircraft=old_aircraft,
    )
    assert design.reference_aircraft is old_aircraft
    assert design.energy_per_ask == pytest.approx(1.5)

    control = execute_model(
        design.control_model(),
        {
            "Des.ramp_up_duration": 5.0,
            "Des.ramp_down_duration": 5.0,
            "Des.entry_into_service": 2030.0,
            "Des.max_share": 0.4,
            "Des.lifetime": 25.0,
        },
    )
    np.testing.assert_allclose(
        control["control.Des.share.times"], [2030.0, 2035.0, 2055.0, 2060.0]
    )
    np.testing.assert_allclose(
        control["control.Des.share.values"], [0.0, 0.4, 0.4, 0.0]
    )

    # The design variant consumes the energy_per_ask input, not the attribute.
    consumption = execute_model(
        design.consumption_model(), {"Des.ask": 10.0, "Des.energy_per_ask": 2.0}
    )
    assert float(consumption["Des.KEROSENE.consumption"]) == pytest.approx(20.0)


def test_aircraft_design_design_model_smoke(turbofan, old_aircraft, execute_model):
    tech_params = [
        AircraftTechParameter(name, tuple(lower_mid_upper[1]))
        for name, lower_mid_upper in (
            tech_params_lower_mid_upper_2020_2040_2060.items()
        )
    ]
    mission = {
        **categories_mission["short_medium"],
        **propulsion_mission["JetA-GasTurbine"],
        "category": "short_medium",
    }
    design = AircraftDesign(
        name="JetA",
        propulsion=turbofan,
        mission=mission,
        power_system=propulsion_architectures["JetA-GasTurbine"],
        aircraft_tech_params=tech_params,
        reference_aircraft=old_aircraft,
    )
    assert len(design.models) == 3

    output_data = execute_model(design.design_model())
    for name in design.design_model().discipline.output_grammar.names:
        assert np.all(np.isfinite(output_data[name]))
    mtow = float(output_data["JetA.mtow"])
    owe = float(output_data["JetA.owe"])
    energy_per_ask = float(output_data["JetA.energy_per_ask"])
    assert mtow > owe > 0.0
    assert 0.1 < energy_per_ask < 5.0
    assert float(output_data["JetA.relative_efficiency_gain"]) == pytest.approx(
        old_aircraft.energy_per_ask / energy_per_ask, rel=1e-9
    )
