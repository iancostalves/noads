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

import pytest

from noads.core.models.energy.energy import Energy
from noads.core.models.energy.energy import ProducedEnergy
from noads.core.models.energy.energy_mix import EnergyMix
from noads.core.models.energy.production_pathway import ProductionPathway
from noads.core.models.energy.streams import Impact
from noads.core.models.energy.streams import MaterialStream
from noads.core.models.energy.streams import Stream


@pytest.fixture
def pathway(co2, electricity):
    return ProductionPathway("Fossil", impacts=[co2], input_streams=[electricity])


def test_streams(co2):
    stream = Stream("WATER", "m3")
    assert stream.name == "WATER"
    assert stream.unit == "m3"

    assert co2.name == "CO2"
    assert co2.unit == "gCO2"
    assert co2.budget == pytest.approx(1.0e3)

    material = MaterialStream("BIO", density=0.5)
    assert material.unit == "kg"
    assert material.volume_to_mass(2.0) == pytest.approx(1.0)
    assert material.mass_to_volume(1.0) == pytest.approx(2.0)


def test_energy_carrier_conversions(kerosene):
    assert kerosene.unit == "MJ"
    assert kerosene.mass_to_energy(2.0) == pytest.approx(88.0)
    assert kerosene.energy_to_mass(88.0) == pytest.approx(2.0)
    assert kerosene.volume_to_energy(1.0) == pytest.approx(0.8 * 44.0)
    assert kerosene.energy_to_volume(0.8 * 44.0) == pytest.approx(1.0)


def test_production_pathway_impact_index_model(pathway, execute_model):
    assert len(pathway.models) == 2

    output_data = execute_model(
        pathway.impact_index_model(),
        {
            "Fossil.direct.CO2_index": 10.0,
            "Fossil.ELECTRICITY.efficiency": 2.0,
            "ELECTRICITY.CO2_index": 4.0,
        },
    )
    assert float(output_data["Fossil.ELECTRICITY.consumption_index"]) == pytest.approx(
        0.5
    )
    assert float(output_data["Fossil.indirect.CO2_index"]) == pytest.approx(2.0)
    assert float(output_data["Fossil.CO2_index"]) == pytest.approx(12.0)


def test_production_pathway_consumption_model(pathway, execute_model):
    output_data = execute_model(
        pathway.consumption_model(),
        {"Fossil.production": 10.0, "Fossil.ELECTRICITY.consumption_index": 0.5},
    )
    assert float(output_data["Fossil.ELECTRICITY.consumption"]) == pytest.approx(5.0)


def test_produced_energy(produced_carrier, co2, electricity, execute_model):
    assert set(produced_carrier.impacts) == {co2}
    assert set(produced_carrier.input_streams) == {electricity}
    assert len(produced_carrier.models) == 3

    # The last pathway share is the residual; indices aggregate by share.
    impact_index = execute_model(
        produced_carrier.impact_index_model(),
        {
            "Fossil.share": 0.3,
            "Fossil.CO2_index": 10.0,
            "Efuel.CO2_index": 2.0,
            "Fossil.ELECTRICITY.consumption_index": 0.5,
            "Efuel.ELECTRICITY.consumption_index": 2.0,
        },
    )
    assert float(impact_index["Efuel.share"]) == pytest.approx(0.7)
    assert float(impact_index["FUEL.CO2_index"]) == pytest.approx(
        0.3 * 10.0 + 0.7 * 2.0
    )
    assert float(impact_index["FUEL.ELECTRICITY.consumption_index"]) == pytest.approx(
        0.3 * 0.5 + 0.7 * 2.0
    )

    production = execute_model(
        produced_carrier.production_model(),
        {"FUEL.production": 10.0, "Fossil.share": 0.3, "Efuel.share": 0.7},
    )
    assert float(production["Fossil.production"]) == pytest.approx(3.0)
    assert float(production["Efuel.production"]) == pytest.approx(7.0)

    # Output streams registration is idempotent.
    other = Energy("OTHER")
    produced_carrier.set_output_streams([other])
    produced_carrier.add_output_stream(other)
    assert produced_carrier.output_streams == [other]


def test_produced_energy_consumption(co2, electricity, execute_model):
    mid_pathway = ProductionPathway("P1", impacts=[co2], input_streams=[electricity])
    mid = ProducedEnergy("MID", pathways=[mid_pathway])
    fuel = Energy("FUEL")
    mid.set_output_streams([fuel])

    output_data = execute_model(
        mid.consumption_model(),
        {"FUEL.MID.consumption": 8.0, "MID.ELECTRICITY.consumption_index": 0.5},
    )
    assert float(output_data["MID.production"]) == pytest.approx(8.0)
    assert float(output_data["MID.ELECTRICITY.consumption"]) == pytest.approx(4.0)


def test_produced_energy_carrier_consumption(produced_carrier, execute_model):
    other = Energy("OTHER")
    produced_carrier.add_output_stream(other)

    # Production covers both direct and indirect consumption of the carrier.
    output_data = execute_model(
        produced_carrier.consumption_model(),
        {
            "FUEL.consumption": 10.0,
            "OTHER.FUEL.consumption": 5.0,
            "FUEL.ELECTRICITY.consumption_index": 2.0,
        },
    )
    assert float(output_data["FUEL.production"]) == pytest.approx(15.0)
    assert float(output_data["FUEL.ELECTRICITY.consumption"]) == pytest.approx(30.0)

    # Conversions are inherited from EnergyCarrier.
    assert produced_carrier.mass_to_energy(1.0) == pytest.approx(44.0)


def test_energy_mix(produced_carrier, co2, electricity, execute_model):
    land = Impact(name="LAND", unit="m2", budget=1.0e2)
    mix = EnergyMix(
        [produced_carrier], extra_impacts=[land], inputs_to_constrain=[electricity]
    )

    # Extra impacts are back-filled into every energy and pathway.
    assert land in mix.impacts
    assert land in produced_carrier.impacts
    for pathway in produced_carrier.pathways:
        assert land in pathway.impacts

    assert mix.final_energies == [produced_carrier]
    assert mix.input_streams == [electricity]
    assert mix.constrained_inputs == [electricity]
    # aggregate + impacts + constraints + 3 energy models + 2 pathways x 2 models
    assert len(mix.models) == 10

    streams = execute_model(
        mix.input_streams_model(), {"FUEL.ELECTRICITY.consumption": 7.0}
    )
    assert float(streams["ELECTRICITY.consumption"]) == pytest.approx(7.0)

    impacts = execute_model(
        mix.total_impacts_model(),
        {"FUEL.CO2_index": 2.0, "FUEL.LAND_index": 0.1, "FUEL.consumption": 10.0},
    )
    assert float(impacts["CO2"]) == pytest.approx(20.0)
    assert float(impacts["LAND"]) == pytest.approx(1.0)

    # Negative residual pathway share violates the constraint.
    constraint_model = mix.constraint_model()
    negative_share = execute_model(constraint_model, {"Efuel.share": -0.2})
    assert float(negative_share["FUEL.constraint"]) == pytest.approx(0.2)
    assert float(negative_share["FUEL.constraint_violation"]) == pytest.approx(0.2)

    positive_share = execute_model(constraint_model, {"Efuel.share": 0.3})
    assert float(positive_share["FUEL.constraint"]) == pytest.approx(-0.3)
    assert float(positive_share["FUEL.constraint_violation"]) == pytest.approx(0.0)

    # Fair-share constraint on the constrained input stream.
    over_share = execute_model(
        constraint_model,
        {
            "ELECTRICITY.consumption": 5.0,
            "ELECTRICITY.global_production": 20.0,
            "ELECTRICITY.fair_share": 0.1,
        },
    )
    assert float(over_share["ELECTRICITY.consumed_share"]) == pytest.approx(0.25)
    assert float(over_share["ELECTRICITY.constraint"]) == pytest.approx(0.15)
    assert float(over_share["ELECTRICITY.constraint_violation"]) == pytest.approx(0.15)

    under_share = execute_model(
        constraint_model,
        {
            "ELECTRICITY.consumption": 5.0,
            "ELECTRICITY.global_production": 20.0,
            "ELECTRICITY.fair_share": 0.5,
        },
    )
    assert float(under_share["ELECTRICITY.constraint"]) == pytest.approx(-0.25)
    assert float(under_share["ELECTRICITY.constraint_violation"]) == pytest.approx(0.0)


def test_energy_mix_without_constraints(co2, electricity):
    single = ProductionPathway("Only", impacts=[co2], input_streams=[electricity])
    produced = ProducedEnergy("HEAT", pathways=[single])
    mix = EnergyMix([produced], plot_coupling_graph=True)

    assert mix.constrained_inputs == []
    # aggregate + impacts + 3 energy models + 1 pathway x 2 models, no constraints
    assert len(mix.models) == 7
