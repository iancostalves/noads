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

"""Metrics of the optimization results of the main paper and of the update.

The pre-computed optima are read from ``docs/examples/optimization/results``: the
main paper in ``<name>/opt_result.json`` and the update in ``update/<name>/``.
"""

import json
from pathlib import Path

import numpy as np

from noads.application.background_scenario_data import co2_budget_2p0deg_66percent
from noads.application.visualization import CO2_2019_MT
from noads.application.visualization import RPK_2019_TRIL

RESULTS = Path(__file__).resolve().parents[2] / "examples/optimization/results"
BUDGET = co2_budget_2p0deg_66percent * 3.0e-2
"""Carbon budget allocated to aviation (3 % of the 2.0 degree budget), in gCO2."""

TECHS = {"Lower": "lowTech", "Mid": "midTech", "Upper": "upTech"}
MARKETS = ["general", "commuter", "regional", "short_medium", "long_range"]
ARCHITECTURES = {
    "Current fleet": "Current_fleet",
    "Jet-A v1": "JetA-GasTurbine-v1",
    "Jet-A v2": "JetA-GasTurbine-v2",
    "LH2 gas turbine": "lH2-GasTurbine",
    "LH2 fuel cell": "lH2-FuelCell",
    "Battery-electric": "Battery-Electric",
}

# (family, variant) -> result-name pattern, with the technology suffix appended
SCENARIOS = {
    ("Baseline SSP1", ""): "SSP1-19-Fossil",
    ("Baseline SSP2", ""): "SSP2-26-Fossil",
    ("Baseline SSP5", ""): "SSP5-45-Fossil",
    ("Drop-in", "trend"): "SSP2-26-DropIn",
    ("Drop-in", "availability"): "SSP2-26-DropIn-Availability",
    ("Drop-in", "low-demand"): "SSP2-26-DropIn-LowDemand",
    ("Breakthrough", "trend"): "SSP2-26",
    ("Breakthrough", "availability"): "SSP2-26-Availability",
    ("Breakthrough", "low-demand"): "SSP2-26-LowDemand",
}


def load(model, base, tech):
    """The outputs and inputs of an optimum, as NumPy arrays.

    Args:
        model: ``"paper"`` or ``"update"``.
        base: The scenario name without the technology suffix.
        tech: The technology scenario (``"Lower"``, ``"Mid"`` or ``"Upper"``).

    Returns:
        The merged inputs and outputs.
    """
    folder = RESULTS / ("update" if model == "update" else "") / f"{base}-{TECHS[tech]}"
    with (folder / "opt_result.json").open() as file:
        data = json.load(file)
    return {k: np.array(v) for k, v in {**data["inputs"], **data["outputs"]}.items()}


def metrics(data):
    """Summary metrics of a single-scenario optimum.

    Args:
        data: The merged inputs and outputs of the optimum.

    Returns:
        The metrics.
    """
    years = data["year"]
    co2 = data["CO2"] * 1e-12  # Mt / year
    ask_total = sum(
        v
        for k, v in data.items()
        if k.endswith(".ask")
        and k.count(".") == 1
        and any(k.startswith(prefix) for prefix in ARCHITECTURES.values())
    )
    result = {
        "cum_co2_pct": 100.0 * float(data["cumulative.CO2"].ravel()[-1]) / BUDGET,
        "peak_co2": float(co2.max()) / CO2_2019_MT,
        "peak_year": float(years[co2.argmax()]),
        "co2_2050": float(co2[years == 2050][0]) / CO2_2019_MT,
        "co2_2075": float(co2[-1]) / CO2_2019_MT,
        "rpk_2075": float(data["rpk"][-1]) / (RPK_2019_TRIL * 1e12),
        "rpk_avoided_2075": 100.0
        * (1.0 - float(data["rpk"][-1] / data["rpk_trend"][-1])),
        "fossil_2075": float(data["Fossil.production"][-1]) * 1e-12,
        "fossil_cum": float(np.trapz(data["Fossil.production"], years)) * 1e-12,
        "biofuel_2075": float(data["Biofuel.production"][-1]) * 1e-12,
        "efuel_2075": float(data["Electrofuel.production"][-1]) * 1e-12,
        "lh2_2075": float(data["LIQUID-H2.production"][-1]) * 1e-12
        if "LIQUID-H2.production" in data
        else 0.0,
        "battery_2075": float(data["BATTERY.production"][-1]) * 1e-12
        if "BATTERY.production" in data
        else 0.0,
        "biomass_peak": float(data["BIOMASS.consumption"].max()) * 1e-12,
        "electricity_peak": float(data["ELECTRICITY.consumption"].max()) * 1e-12,
        "mean_energy_per_ask_2075": float(data["mean_energy_per_ask"][-1]),
    }
    fossil = data["Fossil.production"]
    below = np.where(fossil < 0.05 * fossil.max())[0]
    below = below[years[below] > years[fossil.argmax()]]
    result["fossil_phaseout_year"] = float(years[below[0]]) if below.size else np.nan
    # share of the ASK of 2075 per architecture, in total and per market
    for label, key in ARCHITECTURES.items():
        asks = {
            m: sum(
                float(v[-1])
                for k, v in data.items()
                if k.endswith(f"_{m}.ask") and k.startswith(key)
            )
            for m in MARKETS
        }
        result[f"share_{label}"] = 100.0 * sum(asks.values()) / float(ask_total[-1])
        for m in MARKETS:
            total_m = sum(
                float(v[-1])
                for k, v in data.items()
                if k.endswith(f"_{m}.ask")
                and k.count(".") == 1
                and any(k.startswith(prefix) for prefix in ARCHITECTURES.values())
            )
            result[f"share_{label}_{m}"] = 100.0 * asks[m] / total_m if total_m else 0.0
    return result


def table():
    """Metrics of all single-scenario optima: {(model, family, variant, tech): dict}."""
    out = {}
    for (family, variant), base in SCENARIOS.items():
        for tech in TECHS:
            for model in ("paper", "update"):
                out[model, family, variant, tech] = metrics(load(model, base, tech))
    return out
