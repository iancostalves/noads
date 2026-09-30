"""Calibrate NOADS powertrain technology parameters from powertrain_tech_data.csv.

Outputs lower/mid/upper triplets at EIS 2020/2040/2060, in the format of
tech_params_lower_mid_upper_2020_2040_2060 in noads/application/base_objects.py.
"""
import json
import numpy as np

YEARS = np.array([2020.0, 2040.0, 2060.0])

def quad_spline(x, xs, ys):
    """Same as NOADS AircraftTechParameter: order-2 polynomial through 3 points."""
    return np.polyval(np.polyfit(xs, ys, 2), x)

# ---------------------------------------------------------------- e-motor SP
# Pastra et al. (2023) logistic, with their 30% knockdown, for best-demonstrated
# machines. EIS value = demonstrated value shifted by a delay (demo to certified EIS).
PASTRA = {"lower": (16.3, 0.1411, 2020.0), "mid": (37.8, 0.1213, 2030.0), "upper": (94.3, 0.1134, 2040.0)}
DELAY = {"lower": 10.0, "mid": 8.0, "upper": 6.0}

def pastra(t, L, k, y0):
    return 0.7 * L / (1 + np.exp(-k * (t - y0)))

def emotor_sp_eis(t, scen):
    return pastra(t - DELAY[scen], *PASTRA[scen])

# ---------------------------------------------------------------- e-motor efficiency
# Pastra efficiency model: losses fall by k per year from 4% (0.96) in 2020 for best
# machines. Apply the same delay. Chain = motor x inverter x distribution.
EFF_K = {"lower": 0.01, "mid": 0.025, "upper": 0.045}
def motor_eff_eis(t, scen):
    # best-demonstrated loss 4% in 2020, shifted by the demo-to-EIS delay.
    # gives 0.947 to 0.956 at EIS 2020, close to magniX/Safran/Siemens 0.93 to 0.95
    return 1 - 0.04 * (1 - EFF_K[scen]) ** (t - DELAY[scen] - 2020)
INV_EFF = {"lower": (0.97, 0.98, 0.985), "mid": (0.97, 0.985, 0.99), "upper": (0.97, 0.99, 0.995)}
DIST_EFF = 0.99

# ---------------------------------------------------------------- max unit power (MW)
# certified EIS max per propulsor. 2020: Velis Electro 57.6 kW certified 2020 (EASA).
# Evidence: 125 kW certified 2025, 600 kW targeted 2028-2030, 2 MW demonstrated 2019
# and 2.5 MW claimed 2024, 2 MW superconducting validation 2026.
PMAX = {"lower": (0.06, 1.5, 5.0), "mid": (0.06, 3.0, 12.0), "upper": (0.06, 5.0, 25.0)}
# monotonic on 2020-2060 iff log10(P2060) >= (4 log10(P2040) - log10(P2020)) / 3
for _k, (_a, _b, _c) in PMAX.items():
    assert np.log10(_c) >= (4 * np.log10(_b) - np.log10(_a)) / 3, _k
def pmax_eis(t, scen):
    return 10 ** quad_spline(t, YEARS, np.log10(PMAX[scen]))

# ---------------------------------------------------------------- fuel cell
# system SP (stack+BoP+TMS) consistent with current NOADS and FlyZero delayed ~10 y
FC_SYS = {"lower": (0.75, 2.0, 3.0), "mid": (0.875, 2.5, 4.5), "upper": (1.0, 3.0, 6.0)}
# TMS specific heat rejection, kW heat per kg (FlyZero thermal 5 / 10 LT / 20 HT / >20)
TMS_Q = {"lower": (3.0, 7.0, 10.0), "mid": (4.0, 10.0, 17.5), "upper": (5.0, 15.0, 25.0)}
# radiator parasitic power per unit heat (FlyZero 20 / 15 / 10 %)
TMS_LOSS = {"lower": (0.25, 0.20, 0.15), "mid": (0.225, 0.16, 0.12), "upper": (0.20, 0.12, 0.10)}
# NOADS existing mission fuel cell efficiency (%)
FC_EFF = {"lower": (40, 45, 50), "mid": (40, 50, 57.5), "upper": (40, 55, 65)}
RATED_RATIO = 0.85   # efficiency at rated (hot-day takeoff) / mission efficiency

def main():
    out = {}
    for s in ["lower", "mid", "upper"]:
        sp = emotor_sp_eis(YEARS, s)
        eff_m = motor_eff_eis(YEARS, s)
        chain = eff_m * np.array(INV_EFF[s]) * DIST_EFF
        eta_r = RATED_RATIO * np.array(FC_EFF[s]) / 100
        qp = 1 / eta_r - 1
        core = 1 / (1 / np.array(FC_SYS[s]) - qp / np.array(TMS_Q[s]))
        out[s] = {
            "emotor_specific_power": sp.round(1).tolist(),
            "emotor_efficiency": (100 * eff_m).round(2).tolist(),
            "electric_chain_efficiency": (100 * chain).round(2).tolist(),
            "max_unit_power_MW": list(PMAX[s]),
            "fuelcell_system_specific_power_check": list(FC_SYS[s]),
            "fuelcell_core_specific_power": core.round(2).tolist(),
            "fuelcell_tms_heat_rejection": list(TMS_Q[s]),
            "fuelcell_tms_power_loss": list(TMS_LOSS[s]),
            "fuelcell_efficiency": list(FC_EFF[s]),
            "heat_to_power_at_rated": qp.round(3).tolist(),
        }
    out["scale_exponents"] = {
        "emotor_power_exponent_beta": {"lower": 0.25, "mid": 0.10, "upper": 0.0,
            "evidence": "Dowdle ideal 0.5, optimized 0.25. MW frontier (MIT 1 MW, Wright 2.5 MW, Dowdle 3.6 MW) 0.10. FlyZero 0 over 0.5-4 MW"},
        "emotor_loss_exponent_gamma": {"lower": 0.0, "mid": 0.10, "upper": 0.25,
            "evidence": "IEC 60034-30-1 IE4 fit 0.27 (1.1-160 kW). FlyZero no variation 0.5-4 MW"},
        "fuelcell_power_exponent": {"lower": 0.0, "mid": 0.0, "upper": 0.15,
            "evidence": "OSTI modular BoP scaling implies up to 0.44 over 0.5-4 MW. FlyZero 0. ICCT assumes benefit, unquantified"},
        "reference_unit_power_MW": 1.0,
    }
    out["delays_years"] = DELAY
    json.dump(out, open("calibrated_params.json", "w"), indent=1)
    for s in ["lower", "mid", "upper"]:
        print(s)
        for k, v in out[s].items():
            print("   ", k, v)
    for s in PMAX:
        t = np.arange(2020, 2061, 5)
        print(s, "Pmax MW", dict(zip(t.tolist(), pmax_eis(t, s).round(2).tolist())))


if __name__ == "__main__":
    main()
