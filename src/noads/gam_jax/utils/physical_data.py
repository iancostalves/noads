#!/usr/bin/env python3
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

"""Created on Thu Jan 24 23:22:21 2019.

@author: DRUOT Thierry : original Scilab implementation
         PETEILH Nicolas : portage to Python
"""

# ruff: noqa: D205, E501, N806, TRY002, N802, D103, N803

import numpy as np
from jax.numpy import array
from jax.numpy import interp
from jax.numpy import sqrt
from scipy.optimize import fsolve


def gravity():
    """Reference gravity acceleration."""
    return 9.80665  # Gravity acceleration at sea level


def sea_level_density():
    """Reference air density at sea level."""
    return 1.225  # (kg/m3) Air density at sea level


def sea_level_pressure():
    """Reference air pressure at sea level."""
    return 101325.0  # Pascals


def sea_level_temperature():
    """Reference air temperature at sea level."""
    return 288.15  # Kelvins


def sea_level_sound_speed():
    """Reference sound speed at sea level."""
    return 340.29  # m/s


def gas_data(gas="air"):
    """Gas data for a single gas."""
    r = {
        "air": 287.053,
        "argon": 208.0,
        "carbon_dioxide": 188.9,
        "carbon_monoxide": 297.0,
        "helium": 2077.0,
        "hydrogen_examples": 4124.0,
        "methane": 518.3,
        "nitrogen": 296.8,
        "oxygen": 259.8,
        "propane": 189.0,
        "sulphur_dioxide": 130.0,
        "steam": 462.0,
    }.get(gas, "Erreur: type of gas is unknown")

    gam = {
        "air": 1.40,
        "argon": 1.66,
        "carbon_dioxide": 1.30,
        "carbon_monoxide": 1.40,
        "helium": 1.66,
        "hydrogen_examples": 1.41,
        "methane": 1.32,
        "nitrogen": 1.40,
        "oxygen": 1.40,
        "propane": 1.13,
        "sulphur_dioxide": 1.29,
        "steam": 1.33,
    }.get(gas, "Erreur: type of gas is unknown")

    cv = r / (gam - 1.0)
    cp = gam * cv
    return r, gam, cp, cv


def gas_viscosity(tamb, gas="air"):
    """Mixed gas dynamic viscosity, Sutherland's formula
    WARNING : result will not be accurate if gas is mixing components of too different molecular weights.
    """
    data = {
        "air": [1.715e-5, 273.15, 110.4],
        "ammonia": [0.92e-5, 273.15, 382.9],
        "argon": [2.10e-5, 273.15, 155.6],
        "benzene": [0.70e-5, 273.15, 173.1],
        "carbon_dioxide": [1.37e-5, 273.15, 253.4],
        "carbon_monoxide": [1.66e-5, 273.15, 94.0],
        "chlorine": [1.23e-5, 273.15, 273.0],
        "chloroform": [0.94e-5, 273.15, 284.2],
        "ethylene": [0.97e-5, 273.15, 163.7],
        "helium": [1.87e-5, 273.15, 69.7],
        "hydrogen_examples": [0.84e-5, 273.15, 60.4],
        "methane": [1.03e-5, 273.15, 166.3],
        "neon": [2.98e-5, 273.15, 80.8],
        "nitrogen": [1.66e-5, 273.15, 110.9],
        "nitrous oxide": [1.37e-5, 273.15, 253.4],
        "oxygen": [1.95e-5, 273.15, 57.9],
        "steam": [0.92e-5, 273.15, 154.8],
        "sulphur_dioxide": [1.16e-5, 273.15, 482.3],
        "xenon": [2.12e-5, 273.15, 302.6],
    }  # mu0      T0      S
    # gas={"nitrogen":0.80, "oxygen":0.20}
    # mu = 0.
    # for g in list(gas.keys()):
    #     [mu0,T0,S] = data[g]
    #     mu = mu + gas[g]*(mu0*((T0+S)/(tamb+S))*(tamb/T0)**1.5)
    mu0, T0, S = data[gas]
    return mu0 * ((T0 + S) / (tamb + S)) * (tamb / T0) ** 1.5


def reynolds_number(pamb, tamb, mach):
    """Reynolds number based on Sutherland viscosity model."""
    vsnd = sound_speed(tamb)
    rho, _sig = air_density(pamb, tamb)
    mu = gas_viscosity(tamb)
    return rho * vsnd * mach / mu


def atmosphere(altp, disa):
    """Pressure and temperature from pressure altitude from ground to 50 km."""
    g = gravity()
    r, _gam, _Cp, _Cv = gas_data()

    Z = np.array([0.0, 11000.0, 20000.0, 32000.0, 47000.0, 50000.0])
    dtodz = np.array([-0.0065, 0.0, 0.0010, 0.0028, 0.0])

    P = np.array([sea_level_pressure(), 0.0, 0.0, 0.0, 0.0, 0.0])
    T = np.array([sea_level_temperature(), 0.0, 0.0, 0.0, 0.0, 0.0])

    if Z[-1] < altp:
        msg = "atmosphere, altitude cannot exceed 50km"
        raise Exception(msg)

    j = 0

    while Z[1 + j] <= altp:
        T[j + 1] = T[j] + dtodz[j] * (Z[j + 1] - Z[j])
        if np.abs(dtodz[j]) > 0.0:
            P[j + 1] = P[j] * (1.0 + (dtodz[j] / T[j]) * (Z[j + 1] - Z[j])) ** (
                -g / (r * dtodz[j])
            )
        else:
            P[j + 1] = P[j] * np.exp(-(g / r) * ((Z[j + 1] - Z[j]) / T[j]))
        j = j + 1

    if np.abs(dtodz[j]) > 0.0:
        pamb = P[j] * (1 + (dtodz[j] / T[j]) * (altp - Z[j])) ** (-g / (r * dtodz[j]))
    else:
        pamb = P[j] * np.exp(-(g / r) * ((altp - Z[j]) / T[j]))
    tstd = T[j] + dtodz[j] * (altp - Z[j])
    tamb = tstd + disa

    return pamb, tamb, tstd, dtodz[j]


def atmosphere_P_T():
    # g = gravity()
    # r, gam, Cp, Cv = gas_data()
    #
    # z_dtdz = array([
    #     [0., -0.0065],
    #     [11e3, 0.],
    #     [20e3, 0.0010],
    #     [32e3, 0.0028],
    #     [47e3, 0.0],
    #     [50e3, 0.0]
    # ])
    # p0 = 101325.
    # t0 = 288.15
    #
    # def compute_p_t_z_from_z(last_p_t_z, alt_dtdz):
    #     last_p, last_t, last_z = last_p_t_z
    #     alt, dtdz = alt_dtdz[0], alt_dtdz[1]
    #
    #     t = last_t + dtdz * (alt - last_z)
    #
    #     def compute_p_no_gradient(z):
    #         return last_p * exp(-(g / r) * (z - last_z) / t)
    #
    #     def compute_p_with_gradient(z):
    #         return last_p * (1 + (dtdz / t) * (z - last_z)) ** (-g / (r * dtdz))
    #
    #     p = cond(dtdz == 0., compute_p_no_gradient, compute_p_with_gradient, alt)
    #
    #     carry = p, t, alt
    #     y = array([[p, t]])
    #     print(carry)
    #     return carry, y
    #
    # first_carry = p0, t0, z_dtdz[0][0]
    # last_carry, p_t_matrix = scan(compute_p_t_z_from_z, first_carry, z_dtdz)
    # print(p_t_matrix)
    # Z = z_dtdz[:, 0]
    # P, T = p_t_matrix[:, 0].T[0], p_t_matrix[:, 1].T[0]
    Z = array([0, 11e3, 20e3, 32e3, 47e3, 50e3])
    P = array([
        101325.0,
        22632.05545875,
        5474.88465973,
        868.01764776,
        110.90611578,
        0.0,
    ])
    T = array([288.15, 216.65, 216.65, 228.65, 270.65, 0.0])

    return Z, P, T


def atmosphere_g(altp, disa=0.0):
    """Ambiant data from pressure altitude from ground to 50 km according to Standard Atmosphere."""
    g = gravity()

    Z, P, T = atmosphere_P_T()

    pamb = interp(altp, Z, P)
    tamb = interp(altp, Z, T) + disa

    return pamb, tamb, g


def altg_from_altp(altp, disa):
    """Geometrical altitude from pressure altitude."""

    def fct(altg, altp, disa):
        pamb, _tamb, _dtodz = atmosphere_geo(altg, disa)
        zp = pressure_altitude(pamb)
        return altp - zp

    output_dict = fsolve(fct, x0=altp, args=(altp, disa), full_output=True)

    altg = output_dict[0][0]
    if output_dict[2] != 1:
        msg = "Convergence problem"
        raise Exception(msg)

    return altg


def atmosphere_geo(altg, disa):
    """Pressure and temperature from geometrical altitude from ground to 50 km."""
    g = gravity()
    r, _gam, _Cp, _Cv = gas_data()

    Zi = np.array([0.0, 11000.0, 20000.0, 32000.0, 47000.0, 50000.0])
    dtodzi = np.array([-0.0065, 0.0, 0.0010, 0.0028, 0.0])

    Z = np.zeros_like(Zi)
    dtodz = np.zeros_like(dtodzi)

    P = np.array([sea_level_pressure(), 0.0, 0.0, 0.0, 0.0, 0.0])
    T = np.array([sea_level_temperature(), 0.0, 0.0, 0.0, 0.0, 0.0])

    K = 1 + disa / T[0]
    dtodz[0] = dtodzi[0] / K
    Z[1] = Z[0] + (Zi[1] - Zi[0]) * K

    n = len(P) - 1
    j = 0

    while j < n and Z[1 + j] <= altg:
        T[j + 1] = T[j] + dtodz[j] * (Z[j + 1] - Z[j])
        if np.abs(dtodz[j]) > 0.0:
            P[j + 1] = P[j] * (
                1.0 + (dtodz[j] / (T[j] + disa)) * (Z[j + 1] - Z[j])
            ) ** (-g / (r * dtodz[j]))
        else:
            P[j + 1] = P[j] * np.exp(-(g / r) * ((Z[j + 1] - Z[j]) / (T[j] + disa)))
        j = j + 1
        K = 1 + disa / T[j]
        dtodz[j] = dtodzi[j] / K
        Z[j + 1] = Z[j] + (Zi[j + 1] - Zi[j]) * K

    if Z[1 + j] < altg:
        msg = "atmosphere_geo, altitude cannot exceed 50km"
        raise Exception(msg)

    if np.abs(dtodz[j]) > 0.0:
        pamb = P[j] * (1 + (dtodz[j] / (T[j] + disa)) * (altg - Z[j])) ** (
            -g / (r * dtodz[j])
        )
    else:
        pamb = P[j] * np.exp(-(g / r) * ((altg - Z[j]) / (T[j] + disa)))
    tamb = T[j] + dtodz[j] * (altg - Z[j]) + disa

    return pamb, tamb, dtodz[j]


def pressure_altitude(pamb):
    """Pressure altitude from ground to 50 km."""
    g = gravity()
    r, _gam, _Cp, _Cv = gas_data()

    Z = np.array([0.0, 11000.0, 20000.0, 32000.0, 47000.0, 50000.0])
    dtodz = np.array([-0.0065, 0.0, 0.0010, 0.0028, 0.0])

    P = np.array([sea_level_pressure(), 0.0, 0.0, 0.0, 0.0, 0.0])
    T = np.array([sea_level_temperature(), 0.0, 0.0, 0.0, 0.0, 0.0])

    j = 0
    n = len(P) - 1
    P[1] = P[0] * (1.0 + (dtodz[0] / T[0]) * (Z[1] - Z[0])) ** (-g / (r * dtodz[0]))
    T[1] = T[0] + dtodz[0] * (Z[1] - Z[0])

    while j < n and pamb < P[j + 1]:
        j = j + 1
        T[j + 1] = T[j] + dtodz[j] * (Z[j + 1] - Z[j])
        if np.abs(dtodz[j]) > 0.0:
            P[j + 1] = P[j] * (1.0 + (dtodz[j] / T[j]) * (Z[j + 1] - Z[j])) ** (
                -g / (r * dtodz[j])
            )
        else:
            P[j + 1] = P[j] * np.exp(-(g / r) * ((Z[j + 1] - Z[j]) / T[j]))

    if pamb < P[j + 1]:
        msg = "pressure_altitude, altitude cannot exceed 50km"
        raise Exception(msg)

    if np.abs(dtodz[j]) > 0.0:
        altp = Z[j] + ((pamb / P[j]) ** (-(r * dtodz[j]) / g) - 1) * (T[j] / dtodz[j])
    else:
        altp = Z[j] - (T[j] / (g / r)) * np.log(pamb / P[j])

    return altp


def pressure(altp):
    """Pressure from pressure altitude from ground to 50 km."""
    g = gravity()
    r, _gam, _Cp, _Cv = gas_data()

    Z = np.array([0.0, 11000.0, 20000.0, 32000.0, 47000.0, 50000.0])
    dtodz = np.array([-0.0065, 0.0, 0.0010, 0.0028, 0.0])

    P = np.array([sea_level_pressure(), 0.0, 0.0, 0.0, 0.0, 0.0])
    T = np.array([sea_level_temperature(), 0.0, 0.0, 0.0, 0.0, 0.0])

    if Z[-1] < altp:
        msg = "pressure, altitude cannot exceed 50km"
        raise Exception(msg)

    j = 0

    while Z[1 + j] <= altp:
        T[j + 1] = T[j] + dtodz[j] * (Z[j + 1] - Z[j])
        if np.abs(dtodz[j]) > 0.0:
            P[j + 1] = P[j] * (1.0 + (dtodz[j] / T[j]) * (Z[j + 1] - Z[j])) ** (
                -g / (r * dtodz[j])
            )
        else:
            P[j + 1] = P[j] * np.exp(-(g / r) * ((Z[j + 1] - Z[j]) / T[j]))
        j = j + 1

    if np.abs(dtodz[j]) > 0.0:
        pamb = P[j] * (1 + (dtodz[j] / T[j]) * (altp - Z[j])) ** (-g / (r * dtodz[j]))
    else:
        pamb = P[j] * np.exp(-(g / r) * ((altp - Z[j]) / T[j]))

    return pamb


def air_density(pamb, tamb):
    """Ideal gas density."""
    r, _gam, _Cp, _Cv = gas_data()
    rho0 = sea_level_density()
    rho = pamb / (r * tamb)
    sig = rho / rho0
    return rho, sig


def sound_speed(tamb):
    """Sound speed for ideal gas."""
    r, gam, _Cp, _Cv = gas_data()
    return sqrt(gam * r * tamb)


def total_temperature(tamb, mach):
    """Stagnation temperature."""
    _r, gam, _Cp, _Cv = gas_data()
    return tamb * (1.0 + ((gam - 1.0) / 2.0) * mach**2)


def total_pressure(pamb, mach):
    """Stagnation pressure."""
    _r, gam, _Cp, _Cv = gas_data()
    return pamb * (1 + ((gam - 1.0) / 2.0) * mach**2) ** (gam / (gam - 1.0))


def vtas_from_mach(altp, disa, mach):
    """True air speed from Mach number, subsonic only."""
    _pamb, tamb, _tstd, _dtodz = atmosphere(altp, disa)
    vsnd = sound_speed(tamb)
    return vsnd * mach


def mach_from_vtas(altp, disa, vtas):
    """True air speed from Mach number, subsonic only."""
    _pamb, tamb, _tstd, _dtodz = atmosphere(altp, disa)
    vsnd = sound_speed(tamb)
    return vtas / vsnd


def mach_from_vcas(pamb, Vcas):
    """Mach number from calibrated air speed, subsonic only."""
    _r, gam, _Cp, _Cv = gas_data()
    P0 = sea_level_pressure()
    vc0 = sea_level_sound_speed()
    fac = gam / (gam - 1.0)
    return np.sqrt(
        (
            (
                ((((gam - 1.0) / 2.0) * (Vcas / vc0) ** 2 + 1) ** fac - 1.0) * P0 / pamb
                + 1.0
            )
            ** (1.0 / fac)
            - 1.0
        )
        * (2.0 / (gam - 1.0))
    )


def vcas_from_mach(pamb, mach):
    """Calibrated air speed from Mach number, subsonic only."""
    _r, gam, _Cp, _Cv = gas_data()
    P0 = sea_level_pressure()
    vc0 = sea_level_sound_speed()
    fac = gam / (gam - 1.0)
    return vc0 * np.sqrt(
        (2.0 / (gam - 1.0))
        * (
            (((pamb / P0) * ((1.0 + ((gam - 1.0) / 2.0) * mach**2) ** fac - 1.0)) + 1.0)
            ** (1.0 / fac)
            - 1.0
        )
    )


def vtas_from_vcas(altp, disa, vcas):
    """True air speed from calibrated air speed, subsonic only."""
    pamb, tamb, _tstd, _dtodz = atmosphere(altp, disa)
    mach = mach_from_vcas(pamb, vcas)
    vsnd = sound_speed(tamb)
    return vsnd * mach


def cross_over_altp(Vcas, mach):
    """Altitude where constant calibrated air speed meets constant Mach number, subsonic only."""
    _r, gam, _Cp, _Cv = gas_data()
    P0 = sea_level_pressure()
    vc0 = sea_level_sound_speed()
    fac = gam / (gam - 1)

    pamb = (
        ((1.0 + ((gam - 1.0) / 2.0) * (Vcas / vc0) ** 2) ** fac - 1.0)
        * P0
        / ((1.0 + ((gam - 1.0) / 2.0) * mach**2) ** fac - 1.0)
    )

    return pressure_altitude(pamb)


def climb_mode(speed_mode, mach, dtodz, tstd, disa):
    """Acceleration factor depending on speed driver ('cas': constant CAS, 'mach': constant Mach)
    WARNING : input is mach number whatever speed_mode.
    """
    g = gravity()
    r, gam, _Cp, _Cv = gas_data()

    if speed_mode == "cas":
        fac = (gam - 1.0) / 2.0
        acc_factor = (
            1.0
            + (
                ((1.0 + fac * mach**2) ** (gam / (gam - 1.0)) - 1.0)
                / (1.0 + fac * mach**2) ** (1.0 / (gam - 1.0))
            )
            + ((gam * r) / (2.0 * g)) * (mach**2) * (tstd / (tstd + disa)) * dtodz
        )
    elif speed_mode == "mach":
        acc_factor = (
            1.0 + ((gam * r) / (2.0 * g)) * (mach**2) * (tstd / (tstd + disa)) * dtodz
        )
    else:
        msg = "climb_mode key is unknown"
        raise Exception(msg)

    return acc_factor


def fuel_density(fuel_type, press=101325.0):
    """Reference fuel density."""
    if fuel_type == "kerosene":
        fuel_density = 803.0  # Kerosene : between 775-840 kg/m3
    elif fuel_type == "petrol":
        fuel_density = 803.0  # Same as kerosene
    elif fuel_type == "gasoline":
        fuel_density = 800.0
    elif fuel_type == "liquid_h2":
        fuel_density = 70.8  # Liquid hydrogene
    elif fuel_type == "compressed_h2":
        p = press * 1.0e-5
        fuel_density = (
            -3.11480362e-05 * p + 7.82320891e-02
        ) * p + 1.03207822e-01  # Compressed hydrogen_examples at 293.15 K
    elif fuel_type == "liquid_ch4":
        fuel_density = 422.6  # Liquid methane
    elif fuel_type == "liquid_nh3":
        fuel_density = 681.0  # Liquid ammonia (239.85 K or 20 bar max)
    elif fuel_type == "solid_nh3":
        fuel_density = 817.0  # Solid ammonia (193.15 K)
    elif fuel_type == "battery":
        fuel_density = 2800.0  # Lithium-ion
    else:
        msg = "fuel_type key is unknown"
        raise Exception(msg)
    return fuel_density


def fuel_heat(fuel_type):
    """Reference fuel lower heating value or battery energy density."""
    if (fuel_type == "kerosene") or (fuel_type == "petrol"):
        fuel_heat = 43.1e6  # J/kg, kerosene
    elif fuel_type == "gasoline":
        fuel_heat = 43.2e6  # J/kg, kerosene
    elif fuel_type == "liquid_h2":
        fuel_heat = 121.0e6  # J/kg, liquid hydrogene
    elif fuel_type == "compressed_h2":
        fuel_heat = 121.0e6  # J/kg, compressed hydrogene 700 bars
    elif fuel_type == "liquid_ch4":
        fuel_heat = 50.3e6  # J/kg, Liquid methane
    elif fuel_type == "liquid_nh3":
        fuel_heat = 16.89e6  # J/kg, Liquid ammonia
    else:
        msg = "fuel_type index is out of range"
        raise Exception(msg, fuel_type)
    return fuel_heat


def stoichiometry(oxydizer, fuel):
    if oxydizer == "air":
        if fuel == "hydrogen_examples":
            return 34.5
        msg = "Fuel type is unknown"
        raise Exception(msg)
    msg = "Oxydizer type is unknown"
    raise Exception(msg)


def emission_index(fuel_type, compound):
    """Various emitted compound depending on energy source."""
    if (fuel_type == "kerosene") or (fuel_type == "petrol"):
        index = {
            "CO2": 3140.0 / 1000.0,
            "H2O": 1290.0 / 1000.0,
            "SO2": 0.8 / 1000.0,
            "NOx": 14.0 / 1000.0,
            "CO": 3.0 / 1000.0,
            "HC": 0.4 / 1000.0,
            "sulfuric_acid": 0.04 / 1000.0,
            "nitrous_acid": 0.4 / 1000.0,
            "nitric_acid": 0.2 / 1000.0,
            "soot": 2.5e12,
        }
        return index.get(compound)
    if fuel_type in ["liquid_h2", "Compressed_h2"]:
        index = {
            "CO2": 0.0,
            "H2O": 18000.0 / 1000.0,
            "SO2": 0.0,
            "NOx": 14.0 / 1000.0,
            "CO": 0.0,
            "HC": 0.0,
            "sulfuric_acid": 0.0,
            "nitrous_acid": 0.4 / 1000.0,
            "nitric_acid": 0.2 / 1000.0,
            "soot": 2.0e12,
        }
        return index.get(compound)
    if fuel_type == "battery":
        index = {
            "CO2": 0.0,
            "H2O": 0.0,
            "SO2": 0.0,
            "NOx": 0.0,
            "CO": 0.0,
            "HC": 0.0,
            "sulfuric_acid": 0.0,
            "nitrous_acid": 0.0,
            "nitric_acid": 0.0,
            "soot": 0.0,
        }
        return index.get(compound)
    return None
