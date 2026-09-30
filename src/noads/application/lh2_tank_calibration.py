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
r"""Calibration of the size- and time-dependent LH2 tank gravimetric index (GI).

The tank-to-fuel mass ratio of a tank holding ``m`` kg of LH2, for an aircraft
entering into service in year ``t``, is

.. math::

    r_{tank}(m, t) = k(t) (a + b m^{-1/3} + c / m),

with a volume term ``a`` (pressure vessel), an area term ``b`` (insulation, jacket,
minimum gauge) and a fixed-mass term ``c``. The technology multiplier ``k(t)`` is 1
for a present-day aluminium design-study tank, and the fuel system (pipes, pumps,
conditioning) adds ``s(t)`` kg per kg of LH2. Both follow logistic curves in time
(:class:`~noads.core.models.fleet.aircraft_tech_parameter.LogisticTechParameter`):

.. math::

    GI_{tank} = 1 / (1 + r_{tank}), \\qquad GI_{system} = 1 / (1 + r_{tank} + s).

The fit follows the procedure of the LH2 tank GI handoff note, on the dataset
``aircraft_tech_data/lh2_tank_gi/lh2_tank_gi_dataset.csv`` (105 values from 32
sources):

1. The size law ``(a, b, c)`` is fitted on tank-only design studies with
   aluminium (or unspecified) walls, published up to 2025.
2. ``k_0`` (before maturation) is the geometric mean of the ``k`` implied by
   hardware values and claims of 2020-2025, for tanks above 10 kg of LH2 (smaller
   UAV tanks are outside the range of the size law). The unverified GTL vendor
   claims (k ~ 0.4) are left out: they are only consistent with the ``k_inf`` of
   the Upper scenario.
3. The fuel system ratio ``s(t)`` is fitted on the FlyZero tank/system pairs and
   IZEA 2025, with the FlyZero technology years.
4. ``k_inf`` and ``t_50`` are fitted on the ``k`` implied by the 2020-2025
   hardware and by the FlyZero tank-only projections. FlyZero values are given at
   technology-readiness years (TRL6 in 2026 for an EIS in 2035), so they are
   delayed by 9 years in the Upper scenario, and by 8 more years in the Mid
   scenario (the Airbus ZEROe delay of 5 to 10 years). The Lower scenario only
   reaches the present-day aluminium design-study tanks (``k_inf = 1``), with
   ``k = 1.25`` (between 1 and 1.5) by 2050. The McKinsey system targets are not
   used: they are nearly independent of the tank size, hence inconsistent with the
   size law. ``s(t)`` follows the timing of each scenario, FlyZero being the reference.

Run this module to print the fitted parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from noads._data import data_file

TIME_SCALE = 5.0
"""Logistic transition time scale tau (years), common to all curves."""

SCENARIOS = ("lower", "mid", "upper")

BASIS_WEIGHT = {"stated": 1.0, "derived": 1.0, "estimated": 0.5}
CONFIDENCE_WEIGHT = {"high": 1.0, "medium": 0.7, "low": 0.4}
INTERVAL_WEIGHT = 0.5
"""Weight factor for values given as a range (fitted at their midpoint)."""

FLYZERO_DELAY = {"upper": 9.0, "mid": 17.0}
"""Delay (years) between FlyZero technology year and entry-into-service."""

LOWER_ANCHOR = (2050.0, 1.25)
"""(year, k) reached by the Lower scenario, converging to k_inf = 1."""

MIN_HARDWARE_MASS = 10.0
"""Minimum LH2 mass per tank (kg) of the hardware values used for k_0."""


@dataclass
class LH2TankCalibration:
    """Fitted LH2 tank model parameters."""

    size_law: tuple[float, float, float]
    """Size law coefficients (a, b, c), with m in kg of LH2 per tank."""

    size_law_rmse: float
    """Root mean square GI error of the size law on its data."""

    mass_factor: dict[str, tuple[float, float, float, float]]
    """Tank mass factor k logistic (k_0, k_inf, t_50, tau) per scenario."""

    fuel_system_ratio: dict[str, tuple[float, float, float, float]]
    """Fuel system ratio s logistic (s_0, s_inf, t_50, tau) per scenario."""


def load_dataset() -> pd.DataFrame:
    """Load the LH2 tank GI dataset with numeric columns."""
    df = pd.read_csv(
        data_file(
            "noads.application",
            "aircraft_tech_data",
            "lh2_tank_gi",
            "lh2_tank_gi_dataset.csv",
        )
    )
    for column in ("gi", "gi_low", "gi_high", "m_h2_kg_per_tank", "tech_year"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["m"] = df["m_h2_kg_per_tank"]
    return df


def size_ratio(size_law, m):
    """Tank-to-fuel mass ratio of the baseline (k = 1) tank holding m kg of LH2."""
    a, b, c = size_law
    return a + b * np.power(m, -1.0 / 3.0) + c / m


def logistic(values, t):
    """Logistic curve (v_0, v_inf, t_50, tau) at year t."""
    v_0, v_inf, t_50, tau = values
    return v_inf + (v_0 - v_inf) / (1.0 + np.exp((t - t_50) / tau))


def _row_weight(row) -> float:
    weight = BASIS_WEIGHT.get(row.m_h2_basis, 1.0)
    weight *= CONFIDENCE_WEIGHT.get(str(row.confidence).split()[0], 0.7)
    if np.isfinite(row.gi_low) and np.isfinite(row.gi_high):
        weight *= INTERVAL_WEIGHT
    return weight


def fit_size_law(df: pd.DataFrame) -> tuple[tuple[float, float, float], float]:
    """Fit (a, b, c) >= 0 on present-day aluminium tank-only design studies."""
    wall = df.wall_material.fillna("").str.lower()
    selection = df[
        df.data_type.eq("design_study")
        & df.gi_scope.eq("tank")
        & (wall.eq("") | wall.str.contains("alumin") | wall.str.startswith("al "))
        & (df.tech_year <= 2025)
        & df.m.notna()
    ]
    weights = np.sqrt(np.array([_row_weight(row) for row in selection.itertuples()]))
    m = selection.m.to_numpy()
    gi = selection.gi.to_numpy()

    def residuals(p):
        return weights * (1.0 / (1.0 + size_ratio(p, m)) - gi)

    result = least_squares(
        residuals, x0=[0.2, 2.0, 5.0], bounds=([0.0, 0.0, 0.0], [5.0, 50.0, 1.0e4])
    )
    fit = 1.0 / (1.0 + size_ratio(result.x, m))
    return tuple(float(x) for x in result.x), float(np.sqrt(np.mean((fit - gi) ** 2)))


def implied_mass_factor(size_law, gi, m, fuel_system_ratio=0.0):
    """Mass factor k implied by a (tank or system) GI for m kg of LH2 per tank."""
    return (1.0 / gi - 1.0 - fuel_system_ratio) / size_ratio(size_law, m)


def _flyzero_tank_ratio(df):
    """Mass-weighted tank-to-fuel ratio of the FlyZero concepts per year."""
    tanks = df[df.id.str.startswith("FZ-") & df.gi_scope.eq("tank") & df.m.notna()]
    ratios = {}
    for (concept, year), group in tanks.groupby(["aircraft_class", "tech_year"]):
        ratios[concept, year] = float(
            ((1.0 / group.gi - 1.0) * group.m).sum() / group.m.sum()
        )
    return ratios


def fit_fuel_system_ratio(df: pd.DataFrame) -> tuple[float, float, float]:
    """Fit (s_0, s_inf, t_50) on FlyZero tank/system pairs and IZEA 2025."""
    years, values = [], []
    for (concept, year), tank_ratio in _flyzero_tank_ratio(df).items():
        system = df[df.id.eq(f"FZ-{concept}-total-{int(year)}")].gi.iloc[0]
        years.append(year)
        values.append(1.0 / system - 1.0 - tank_ratio)
    izea = df.set_index("id").gi
    years.append(2025.0)
    values.append((1.0 / izea["IZ25-system"]) - (1.0 / izea["IZ25-tank"]))
    years, values = np.array(years), np.array(values)

    def residuals(p):
        return logistic((*p, TIME_SCALE), years) - values

    result = least_squares(
        residuals,
        x0=[0.35, 0.12, 2030.0],
        bounds=([0.0, 0.0, 2000.0], [1.0, 1.0, 2060.0]),
    )
    return tuple(float(x) for x in result.x)


def initial_mass_factor(df, size_law):
    """Geometric mean of the mass factor k implied by the 2020-2025 hardware."""
    _years, k = _hardware_anchors(df, size_law)
    return float(np.exp(np.mean(np.log(k))))


def _hardware_anchors(df, size_law):
    """(year, k) points implied by the 2020-2025 hardware values and claims."""
    hardware = df[
        df.data_type.isin(["hardware_measured", "hardware_claim"])
        & df.gi_scope.isin(["tank", "system"])
        & df.tech_year.between(2020, 2025)
        & (df.m >= MIN_HARDWARE_MASS)
        & ~df.source_key.eq("gtl2024")
    ]
    k = implied_mass_factor(size_law, hardware.gi.to_numpy(), hardware.m.to_numpy())
    return hardware.tech_year.to_numpy(), k


def _flyzero_anchors(df, size_law, delay):
    """(year, k) points implied by the FlyZero tank-only projections."""
    rows = df[df.id.str.startswith("FZ-") & df.gi_scope.eq("tank") & df.m.notna()]
    years = rows.tech_year.to_numpy() + delay
    return years, implied_mass_factor(size_law, rows.gi.to_numpy(), rows.m.to_numpy())


def calibrate() -> LH2TankCalibration:
    """Fit the LH2 tank GI model on the dataset."""
    df = load_dataset()
    size_law, rmse = fit_size_law(df)
    s_0, s_inf, s_t50 = fit_fuel_system_ratio(df)
    k_0 = initial_mass_factor(df, size_law)
    hardware_years, hardware_k = _hardware_anchors(df, size_law)
    mass_factor = {}
    for scenario, delay in FLYZERO_DELAY.items():
        years, k = _flyzero_anchors(df, size_law, delay)
        years = np.concatenate([years, hardware_years])
        k = np.concatenate([k, hardware_k])

        def residuals(p, years=years, k=k):
            return np.log(logistic((k_0, p[0], p[1], TIME_SCALE), years) / k)

        result = least_squares(
            residuals, x0=[0.8, 2035.0], bounds=([0.05, 2000.0], [k_0, 2070.0])
        )
        mass_factor[scenario] = (k_0, *(float(x) for x in result.x), TIME_SCALE)
    year, k_year = LOWER_ANCHOR
    t_50 = year - TIME_SCALE * np.log((k_0 - 1.0) / (k_year - 1.0) - 1.0)
    mass_factor["lower"] = (k_0, 1.0, float(t_50), TIME_SCALE)
    # The fuel system matures with the timing of the tanks of each scenario. s(t)
    # is fitted on FlyZero technology years, delayed like the tanks of the Upper
    # scenario.
    fuel_system_ratio = {
        scenario: (
            s_0,
            s_inf,
            s_t50
            + FLYZERO_DELAY["upper"]
            + mass_factor[scenario][2]
            - mass_factor["upper"][2],
            TIME_SCALE,
        )
        for scenario in SCENARIOS
    }
    mass_factor = {scenario: mass_factor[scenario] for scenario in SCENARIOS}
    return LH2TankCalibration(size_law, rmse, mass_factor, fuel_system_ratio)


def tank_gravimetric_index(calibration, scenario, m, t, system=False):
    """Tank-only (or system) GI for m kg of LH2 per tank and an EIS year t."""
    k = logistic(calibration.mass_factor[scenario], t)
    r = k * size_ratio(calibration.size_law, m)
    if system:
        r = r + logistic(calibration.fuel_system_ratio[scenario], t)
    return 1.0 / (1.0 + r)


def summary(calibration: LH2TankCalibration) -> str:
    """Text summary of the fitted parameters and of the resulting tank GI."""
    lines = [
        f"size law (a, b, c) = {np.round(calibration.size_law, 4).tolist()}",
        f"size law RMSE (GI) = {calibration.size_law_rmse:.4f}",
    ]
    for scenario in SCENARIOS:
        k = np.round(calibration.mass_factor[scenario], 3).tolist()
        s = np.round(calibration.fuel_system_ratio[scenario], 3).tolist()
        lines.append(f"{scenario}: k = {k}, s = {s}")
        for m in (75.0, 1250.0, 10500.0):
            gi = [
                float(tank_gravimetric_index(calibration, scenario, m, t))
                for t in (2020, 2040, 2060)
            ]
            lines.append(
                f"    {m:>7.0f} kg per tank: GI_tank 2020/2040/2060 = "
                f"{np.round(gi, 3).tolist()}"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    sys.stdout.write(summary(calibrate()) + "\n")
