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

"""Operate an existing aircraft."""

from __future__ import annotations

from typing import TYPE_CHECKING

from jax.numpy import exp
from jax.numpy import log
from jax.numpy import where
from jax.numpy import zeros

from noads.core.model import JAXModel

if TYPE_CHECKING:
    from collections.abc import Mapping

    from noads.core.models.energy.energy import EnergyCarrier


class PropulsionSystem:  # noqa: B903
    """A propulsion system, mapping an architecture to the carriers it embarks.

    Associates a power-system type (e.g. ``turbofan``, ``lh2_fuel_cell``,
    ``electric_propulsion``) with the energy carriers it consumes and their shares
    of the embarked energy, linking each aircraft to the energy-mix models.
    """

    name: str
    """Name of the propulsion architecture."""

    energy_carrier_mix: Mapping[EnergyCarrier:float]
    """Energy carriers and relative part in the mission energy."""

    def __init__(self, name: str, energy_carrier_mix: Mapping[EnergyCarrier:float]):
        """Initialize PropulsionSystem from name and EnergyCarrier's share."""
        self.name = name
        self.energy_carrier_mix = energy_carrier_mix


class AircraftOperation:
    """An aircraft operated with a fixed, exogenous energy consumption.

    Used for the current fleet of each market, whose energy consumption per ASK is
    initialized from the 2019 AeroSCOPE statistics (quartile depending on the
    technology scenario) instead of being designed. Provides the operation models
    computing covered supply and energy consumption per embarked carrier.
    """

    name: str
    """Aircraft name."""

    propulsion: PropulsionSystem
    """Aircraft propulsion system."""

    energy_per_ask: float
    """Aircraft energy consumption."""

    lifetime: float
    """Aircraft lifetime."""

    recent: bool
    """Boolean to indicate if aircraft is recent, or a new design."""

    def __init__(
        self,
        name: str,
        propulsion: PropulsionSystem,
        energy_per_ask: float = 0.87,
        lifetime: float = 25.0,
        recent: bool = False,
    ):
        """Initialize AircraftOperation."""
        self.name = name
        self.propulsion = propulsion
        self.energy_per_ask = energy_per_ask
        self.lifetime = lifetime
        self.recent = recent

    @property
    def models(self):
        """List of models."""
        if self.recent:
            return [self.consumption_model()]
        return [self.share_model(), self.consumption_model()]

    def _sigmoid_fleet_share(self, years_since_introduction, max_share, cut_below=2):
        growth_rate = log(100 / cut_below - 1) / (self.lifetime / 2)
        year_inflection = 0 if self.recent else self.lifetime / 2
        calc_share = (
            1e2
            * max_share
            / (1 + exp(-growth_rate * (years_since_introduction - year_inflection)))
        )
        calc_share_max = 100 / (
            1 + exp(-growth_rate * (years_since_introduction - year_inflection))
        )
        return 1e-2 * where(
            calc_share_max < cut_below, zeros(calc_share.shape), calc_share
        )

    def _fleet_share(self, input_data):
        year = input_data["year"]
        entry_into_service = input_data[f"{self.name}.entry_into_service"]
        max_share = input_data[f"{self.name}.max_share"]
        return {
            f"{self.name}.share": self._sigmoid_fleet_share(
                year - entry_into_service, max_share
            )
        }

    def share_model(self):
        """Aircraft market share based on sigmoid functions."""
        default_inputs = {
            "year": 2025.0,
            f"{self.name}.entry_into_service": 2035.0,
            f"{self.name}.max_share": 0.0,
        }
        output_names = [f"{self.name}.share"]
        return JAXModel(
            function=self._fleet_share,
            input_names=list(default_inputs.keys()),
            output_names=output_names,
            default_inputs=default_inputs,
            name=f"{self.name} ASK share",
        )

    def _energy_consumption(self, input_data):
        ask = input_data[f"{self.name}.ask"]
        return {
            f"{self.name}.{carrier.name}.consumption": carrier_ratio
            * ask
            * self.energy_per_ask
            for carrier, carrier_ratio in self.propulsion.energy_carrier_mix.items()
        }

    def consumption_model(self):
        """Energy carrier consumption model."""
        default_inputs = {f"{self.name}.ask": 0.0}
        output_names = [
            f"{self.name}.{carrier.name}.consumption"
            for carrier in self.propulsion.energy_carrier_mix
        ]
        return JAXModel(
            function=self._energy_consumption,
            input_names=list(default_inputs.keys()),
            output_names=output_names,
            default_inputs=default_inputs,
            name=f"{self.name} consumption",
        )
