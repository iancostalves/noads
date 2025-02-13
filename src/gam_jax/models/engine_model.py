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

import matplotlib.pyplot as plt
import numpy as np
import utils.data_analysis as uda
from scipy import interpolate
from utils import physical_data as phd
from utils import unit


def psfc_thierry(pw):
    c1 = 0.2
    k1 = 10
    p1 = 0.65

    return unit.convert_from("kg/kW/h", c1 + k1 / unit.kW_W(pw) ** p1)


def tsfc_thierry(bpr, vtas):
    fhv = phd.fuel_heat("petrol")

    alpha = 0.02
    # eff_th = 0.474
    eff_th = 0.464

    eff_pr = 1 / (
        0.5 + np.sqrt(0.25 + ((alpha * eff_th * fhv) / (2 * (1 + bpr) * vtas**2)))
    )

    return vtas / (eff_th * eff_pr * fhv)


def tsfc_mattingly_hbpr(tamb, vtas):
    _r, _gam, _cp, _cv = phd.gas_data()
    t0 = phd.sea_level_temperature()
    phd.fuel_heat("petrol")

    a0 = phd.sound_speed(tamb)
    mach = vtas / a0

    return unit.convert_from("lb/lbf/h", (0.4 + 0.45 * mach) * np.sqrt(tamb / t0))


def tsfc_mattingly_lbpr(tamb, vtas):
    _r, _gam, _cp, _cv = phd.gas_data()
    t0 = phd.sea_level_temperature()
    phd.fuel_heat("petrol")

    a0 = phd.sound_speed(tamb)
    mach = vtas / a0

    return unit.convert_from("lb/lbf/h", (1.0 + 0.35 * mach) * np.sqrt(tamb / t0))


def tsfc_mattingly(t41, opr, fpr, bpr, tamb, vtas):
    _r, gam, cp, _cv = phd.gas_data()
    fhv = phd.fuel_heat("petrol")

    def tsfc_mattingly(t41, opr, fpr, bpr, tamb, vtas):
        _r, _gam, _cp, _cv = phd.gas_data()
        phd.fuel_heat("petrol")

    kth = 1
    dth = 0

    a0 = phd.sound_speed(tamb)
    mach = vtas / a0

    tau_r = 1 + ((gam - 1) / 2) * mach**2
    tau_lambda = t41 / tamb
    tau_c = opr ** ((gam - 1) / gam)
    tau_f = fpr ** ((gam - 1) / gam)

    v_core_jet_o_a0 = np.sqrt(
        (2 / (gam - 1))
        * (
            tau_lambda
            - tau_r * (tau_c - 1 + bpr * (tau_f - 1))
            - tau_lambda / (tau_r * tau_c)
        )
    )

    v_fan_jet_o_a0 = np.sqrt((2 / (gam - 1)) * (tau_r * tau_f - 1))

    (v_core_jet_o_a0 - mach) / (v_fan_jet_o_a0 - mach)

    (a0 / (1 + bpr)) * (v_core_jet_o_a0 - mach + bpr * (v_fan_jet_o_a0 - mach))

    (cp * tamb / fhv) * (tau_lambda - tau_r * tau_c)

    # tsfc = f / ((1 + bpr)*thrust_o_air_flow)

    eta_th = kth * (1 - 1 / (tau_r * tau_c)) + dth

    eta_pr = (
        2
        * mach
        * (v_core_jet_o_a0 - mach + bpr * (v_fan_jet_o_a0 - mach))
        / (v_core_jet_o_a0**2 - mach**2 + bpr * (v_fan_jet_o_a0**2 - mach**2))
    )

    eta_total = eta_th * eta_pr

    tsfc = vtas / (eta_total * fhv)

    # tsfc = max(tsfc, unit.convert_from("kg/daN/h", 0.52))

    return tsfc, eta_th, eta_pr


if __name__ == "__main__":
    # disa = 0
    # altp = unit.m_ft(35000)
    #
    # pamb, tamb, g = phd.atmosphere_g(altp, disa)
    #
    # t41 = 1700
    # opr = 30
    # fpr = 1.4
    # bpr = 5
    #
    # tas = 230
    #
    # tsfc, eta_th, eta_pr = tsfc_mattingly(t41, opr, fpr, bpr, tamb, tas)
    #
    # print("")
    # print("bpr = ", bpr)
    # print("opr = ", opr)
    # print("fpr = ", fpr)
    # print("eff_th = ", "%.4f" % eta_th)
    # print("eff_pr = ", "%.4f" % eta_pr)
    # print("tsfc = ", "%.4f" % unit.convert_to("kg/daN/h", tsfc), " kg/daN/h")
    # print("vtas/tsfc = ", "%.4f" % (tas / tsfc * 1e-6), " MJ/kg")
    #

    # -----------------------------------------------------------------------------------------------------------------------
    #
    #  Calibrating turbofan model
    #
    # -----------------------------------------------------------------------------------------------------------------------
    file = "../all_data/turbofan_data.xlsx"

    turbofan_data, unit_data = uda.read_db(file)

    # -----------------------------------------------------------------------------------------------------------------------
    #
    #  Calibrating TSFC
    #
    # -----------------------------------------------------------------------------------------------------------------------

    disa = 0
    altp = unit.m_ft(35000)

    pamb, tamb, g = phd.atmosphere_g(altp, disa)

    t41 = 1650
    opr = 30

    ksfc = 0.9

    for j in range(len(turbofan_data)):
        bpr = turbofan_data.loc[j, "bpr"]
        tas = turbofan_data.loc[j, "speed"]

        fct_fpr = interpolate.interp1d(
            [1, 9, 12, 15, 18], [1.5, 1.3, 1.25, 1.2, 1.15], kind="linear"
        )
        fpr = fct_fpr(bpr)

        tsfc1, eta_th, eta_pr = tsfc_mattingly(t41, opr, fpr, bpr, tamb, tas)
        turbofan_data.loc[j, "tsfc1"] = unit.convert_to("kg/daN/h", tsfc1 * ksfc)

        tsfc2 = tsfc_mattingly_hbpr(tamb, tas)
        turbofan_data.loc[j, "tsfc2"] = unit.convert_to("kg/daN/h", tsfc2)

        # tsfc3 = tsfc_mattingly_lbpr(tamb, tas)
        # turbofan_data.loc[j ,"tsfc3"] = unit.convert_to("kg/daN/h", tsfc3)

        tsfc4 = tsfc_thierry(bpr, tas)
        turbofan_data.loc[j, "tsfc4"] = unit.convert_to("kg/daN/h", tsfc4)

    x = turbofan_data.loc[:, "bpr"]
    y = unit.convert_to("kg/daN/h", turbofan_data.loc[:, "tsfc"])
    plt.scatter(x, y, c="green", marker=".", alpha=0.5, label="Data base")

    z = turbofan_data.loc[:, "tsfc1"]
    plt.scatter(x, z, c="red", marker=".", alpha=0.5, label="Mattingly Ideal")

    z = turbofan_data.loc[:, "tsfc2"]
    plt.scatter(x, z, c="orange", marker=".", alpha=0.5, label="Mattingly HBPR")

    # z = turbofan_data.loc[: ,"tsfc3"]
    # plt.scatter(x, z, c="violet", marker='.', alpha=0.5)

    z = turbofan_data.loc[:, "tsfc4"]
    plt.scatter(x, z, c="blue", marker=".", alpha=0.5, label="Simple model")

    plt.legend()

    title = "TSFC model comparison"

    plt.title(title, fontsize=16)
    plt.ylabel("TSFC (kg/daN/h)")
    plt.xlabel("BPR")
    plt.grid(True)

    plt.savefig("output/engine_model_" + title + ".png")

    plt.show()

    bpr_min = min(turbofan_data["bpr"])
    bpr_max = max(turbofan_data["bpr"])

    tas = 231.3

    turbofan_data1 = (
        turbofan_data[turbofan_data["speed"] > 232].reset_index(drop=True).copy()
    )

    x = turbofan_data1.loc[:, "bpr"]
    y = unit.convert_to("kg/daN/h", turbofan_data1.loc[:, "tsfc"])
    plt.scatter(x, y, c="green", marker="o", alpha=0.5)

    bpr_list = []
    tsfc_list1 = []
    for bpr in np.linspace(bpr_min, bpr_max, 100):
        tsfc1 = tsfc_thierry(bpr, tas)
        bpr_list.append(bpr)
        tsfc_list1.append(unit.convert_to("kg/daN/h", tsfc1))

    plt.plot(
        bpr_list,
        tsfc_list1,
        c="blue",
    )

    title = "TSFC regression, Mach 0.78"

    plt.title(title, fontsize=16)
    plt.ylabel("TSFC (kg/daN/h)", fontsize=16)
    plt.xlabel("BPR", fontsize=16)
    plt.grid(True)

    plt.savefig("output/engine_model_" + title + ".png")

    plt.show()

    tas = 243.1

    turbofan_data1 = (
        turbofan_data[turbofan_data["speed"] < 243].reset_index(drop=True).copy()
    )

    x = turbofan_data1.loc[:, "bpr"]
    y = unit.convert_to("kg/daN/h", turbofan_data1.loc[:, "tsfc"])
    plt.scatter(x, y, c="green", marker="o", alpha=0.5)

    bpr_list = []
    tsfc_list1 = []
    for bpr in np.linspace(bpr_min, bpr_max, 100):
        tsfc1 = tsfc_thierry(bpr, tas)
        bpr_list.append(bpr)
        tsfc_list1.append(unit.convert_to("kg/daN/h", tsfc1))

    plt.plot(
        bpr_list,
        tsfc_list1,
        c="blue",
    )

    title = "TSFC regression, Mach 0.82"

    plt.title(title, fontsize=16)
    plt.ylabel("TSFC (kg/daN/h)", fontsize=16)
    plt.xlabel("BPR", fontsize=16)
    plt.grid(True)

    plt.savefig("output/engine_model_" + title + ".png")

    plt.show()

    # -----------------------------------------------------------------------------------------------------------------------
    #
    #  Calibrating turbofan mass
    #
    # -----------------------------------------------------------------------------------------------------------------------
    x = turbofan_data.loc[:, "max_power"]
    y = turbofan_data.loc[:, "dry_mass"]
    plt.scatter(unit.kW_W(x), y, c="green", marker="o", alpha=0.5)

    gi = 4.3
    power_list = [0, unit.kW_W(max(x))]
    mass_list = [0, power_list[1] / gi]
    plt.plot(
        power_list,
        mass_list,
        c="blue",
    )

    title = "Turbofan mass regression"

    plt.title(title, fontsize=16)
    plt.ylabel("Engine mass (kg)", fontsize=16)
    plt.xlabel("Engine GPI (kW)", fontsize=16)
    plt.grid(True)

    plt.savefig("output/engine_model_" + title + ".png")

    plt.show()

    # -----------------------------------------------------------------------------------------------------------------------
    #
    #  Calibrating turbofan model
    #
    # -----------------------------------------------------------------------------------------------------------------------
    file = "../all_data/turboshaft_data.xlsx"

    turboshaft_data, unit_data = uda.read_db(file)

    # -----------------------------------------------------------------------------------------------------------------------
    #
    #  Calibrating PSFC
    #
    # -----------------------------------------------------------------------------------------------------------------------
    pw_min = min(turboshaft_data["max_power"])
    pw_max = max(turboshaft_data["max_power"])

    pw_list = []
    psfc_list1 = []
    for pw in np.linspace(pw_min, pw_max, 100):
        psfc1 = psfc_thierry(pw)
        pw_list.append(unit.convert_to("kW", pw))
        psfc_list1.append(unit.convert_to("kg/kW/h", psfc1))

    plt.plot(
        pw_list,
        psfc_list1,
        c="blue",
    )

    x = unit.convert_to("kW", turboshaft_data.loc[:, "max_power"])
    y = unit.convert_to("kg/kW/h", turboshaft_data.loc[:, "psfc"])
    plt.scatter(x, y, c="green", marker="o", alpha=0.5)

    title = "PSFC regression"

    plt.title(title, fontsize=16)
    plt.ylabel("PSFC (kg/kW/h)", fontsize=16)
    plt.xlabel("Max power (kW)", fontsize=16)
    plt.grid(True)

    plt.savefig("output/engine_model_" + title + ".png")

    plt.show()

    # -----------------------------------------------------------------------------------------------------------------------
    #
    #  Calibrating turboshaft mass
    #
    # -----------------------------------------------------------------------------------------------------------------------
    x = turboshaft_data.loc[:, "max_power"]
    y = turboshaft_data.loc[:, "dry_mass"]
    plt.scatter(unit.kW_W(x), y, c="green", marker="o", alpha=0.5)

    gi = 4.3
    power_list = [0, unit.kW_W(max(x))]
    mass_list = [0, power_list[1] / gi]
    plt.plot(
        power_list,
        mass_list,
        c="blue",
    )

    title = "Turboshaft mass regression"

    plt.title(title, fontsize=16)
    plt.ylabel("Engine mass (kg)", fontsize=16)
    plt.xlabel("Engine GPI (kW)", fontsize=16)
    plt.grid(True)

    plt.savefig("output/engine_model_" + title + ".png")

    plt.show()

    # -----------------------------------------------------------------------------------------------------------------------
    #
    #  Calibrating piston model
    #
    # -----------------------------------------------------------------------------------------------------------------------
    file = "../all_data/piston_data.xlsx"

    piston_data, unit_data = uda.read_db(file)

    # -----------------------------------------------------------------------------------------------------------------------
    #
    #  Calibrating turboshaft mass
    #
    # -----------------------------------------------------------------------------------------------------------------------
    x = piston_data.loc[:, "max_power"]
    y = piston_data.loc[:, "dry_mass"]
    plt.scatter(unit.kW_W(x), y, c="green", marker="o", alpha=0.5)

    gi = 1.1
    power_list = [0, unit.kW_W(max(x))]
    mass_list = [0, power_list[1] / gi]
    plt.plot(
        power_list,
        mass_list,
        c="blue",
    )

    title = "Piston engine mass regression"

    plt.title(title, fontsize=16)
    plt.ylabel("Engine mass (kg)", fontsize=16)
    plt.xlabel("Engine GPI (kW)", fontsize=16)
    plt.grid(True)

    plt.savefig("output/engine_model_" + title + ".png")

    plt.show()
