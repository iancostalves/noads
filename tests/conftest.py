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
from noads.core.models.energy.energy import EnergyCarrier
from noads.core.models.energy.energy import ProducedEnergyCarrier
from noads.core.models.energy.production_pathway import ProductionPathway
from noads.core.models.energy.streams import Impact
from noads.core.models.fleet.aircraft_operation import AircraftOperation
from noads.core.models.fleet.aircraft_operation import PropulsionSystem


@pytest.fixture
def execute_model():
    """Execute a Model's discipline with default inputs, optionally overridden."""

    def _execute(model, overrides=None):
        data = dict(model.discipline.default_input_data)
        if overrides:
            data.update(overrides)
        return model.discipline.execute(data)

    return _execute


# EnergyMix constructors mutate the impacts of pathways and energies, so all
# energy-related fixtures are function-scoped to keep tests independent.
@pytest.fixture
def co2():
    return Impact(name="CO2", unit="gCO2", budget=1.0e3)


@pytest.fixture
def electricity():
    return Energy("ELECTRICITY")


@pytest.fixture
def kerosene():
    return EnergyCarrier("KEROSENE", density=0.8, specific_energy=44.0)


@pytest.fixture
def battery():
    return EnergyCarrier("BATTERY", density=1.8, specific_energy=1.29)


@pytest.fixture
def produced_carrier(co2, electricity):
    """A two-pathway carrier consuming electricity and emitting CO2."""
    fossil = ProductionPathway("Fossil", impacts=[co2], input_streams=[electricity])
    efuel = ProductionPathway("Efuel", impacts=[co2], input_streams=[electricity])
    return ProducedEnergyCarrier(
        "FUEL", pathways=[fossil, efuel], density=0.8, specific_energy=44.0
    )


@pytest.fixture
def turbofan(kerosene):
    return PropulsionSystem("turbofan", {kerosene: 1.0})


@pytest.fixture
def old_aircraft(turbofan):
    return AircraftOperation(
        "Old", turbofan, energy_per_ask=1.2, lifetime=20.0, recent=True
    )


@pytest.fixture
def new_aircraft(turbofan):
    return AircraftOperation(
        "New", turbofan, energy_per_ask=0.8, lifetime=25.0, recent=False
    )
