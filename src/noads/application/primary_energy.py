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


"""Primary energy per final energy of the energy carriers.

The primary energies are those of the paper: oil, biomass and (grid) electricity.
Each factor chains the pathway efficiencies of
:func:`~noads.application.scenario_setup.single_scenario_setup` (MJ produced per MJ of
input), which depend on time only, assuming that all hydrogen is electrolytic.
"""

from numpy import asarray
from numpy import interp

PATHWAY_YEARS = (2025.0, 2035.0, 2050.0)
"""Years of the time-dependent pathway efficiencies (constant outside)."""

TIME_EFFICIENCIES = {
    "Electrolysis.ELECTRICITY.efficiency": (0.71, 0.71 * 1.03, 0.71 * 1.06),
    "Power_to_liquid.ELECTRICITY.efficiency": (1.53, 1.53 * 1.08, 1.53 * 1.16),
    "Power_to_liquid.GAS-H2.efficiency": (0.53, 0.53 * 1.06, 0.53 * 1.12),
    "H2_liquefaction.ELECTRICITY.efficiency": (4.54, 4.54 * 1.2, 4.54 * 1.4),
}
"""Time-dependent pathway efficiencies, as in the scenario setup."""

CONSTANT_EFFICIENCIES = {
    "Refinery.OIL.efficiency": 0.865,
    "HEFA.BIOMASS.efficiency": 0.59,
    "ATJ.BIOMASS.efficiency": 0.30,
    "FT.BIOMASS.efficiency": 0.20,
    "H2_liquefaction.GAS-H2.efficiency": 1.0,
    "Charging.ELECTRICITY.efficiency": 0.98,
}
"""Constant pathway efficiencies, as in the scenario setup."""

CARRIERS = {
    "Fossil kerosene": "OIL",
    "HEFA": "BIOMASS",
    "ATJ": "BIOMASS",
    "FT": "BIOMASS",
    "E-fuel": "ELECTRICITY",
    "LH2": "ELECTRICITY",
    "Battery": "ELECTRICITY",
}
"""Energy carriers (or biofuel pathways) and their primary energy."""


def pathway_efficiency(name, years):
    """Efficiency of a pathway input (MJ produced per MJ of input) versus year."""
    if name in CONSTANT_EFFICIENCIES:
        return CONSTANT_EFFICIENCIES[name] + 0.0 * asarray(years, dtype=float)
    return interp(years, PATHWAY_YEARS, TIME_EFFICIENCIES[name])


def primary_energy_factors(years):
    """Primary energy per final energy (MJ/MJ) of each carrier versus year.

    Args:
        years: The years.

    Returns:
        The primary energy factor of each carrier of :data:`CARRIERS`.
    """

    def inverse(name):
        return 1.0 / pathway_efficiency(name, years)

    electrolysis = inverse("Electrolysis.ELECTRICITY.efficiency")
    return {
        "Fossil kerosene": inverse("Refinery.OIL.efficiency"),
        "HEFA": inverse("HEFA.BIOMASS.efficiency"),
        "ATJ": inverse("ATJ.BIOMASS.efficiency"),
        "FT": inverse("FT.BIOMASS.efficiency"),
        "E-fuel": inverse("Power_to_liquid.ELECTRICITY.efficiency")
        + inverse("Power_to_liquid.GAS-H2.efficiency") * electrolysis,
        "LH2": inverse("H2_liquefaction.ELECTRICITY.efficiency")
        + inverse("H2_liquefaction.GAS-H2.efficiency") * electrolysis,
        "Battery": inverse("Charging.ELECTRICITY.efficiency"),
    }
