"""Preliminary fit of GI(m, t) = 1 / (1 + k(t) * r0(m)),  r0(m) = a + b*m^(-1/3) + c/m.

r0 is the tank-to-fuel mass ratio of a present-day (aluminium, conventional insulation)
*design-study* tank holding m kg of LH2. k(t) is a technology multiplier (k=1 for that baseline).
Starting values only; the downstream session should redo this with proper weighting/UQ.
"""
import numpy as np, pandas as pd
from scipy.optimize import least_squares

df = pd.read_csv("lh2_tank_gi_dataset.csv")
num = lambda s: pd.to_numeric(s, errors="coerce")
df["m"] = num(df.m_h2_kg_per_tank); df["gi"] = num(df.gi)

base_ids = [i for i in df.id if i.startswith(("VE10-","SA22-","IZ25-tank","WI18-cyl","WI18-4tanks","MZ24-7","MZ24-9",
                                               "SV23-ATR42","AE24-","HU21-100m3","HU21-0p5m","HU21-900m3","HU21-300m3","MZ24-UAV-MLI"))]
b = df[df.id.isin(base_ids) & df.m.notna()].copy()
w = np.where(b.m_h2_basis.eq("estimated"), 0.5, 1.0)

def r0(p, m):
    a, bb, c = p
    return a + bb*np.power(m, -1/3) + c/m
def gi(p, m, k=1.0):
    return 1/(1 + k*r0(p, m))
res = least_squares(lambda p: w*(gi(p, b.m.values) - b.gi.values), x0=[0.2, 2.0, 5.0], bounds=([0,0,0],[5,50,1e4]))
p = res.x
b["fit"] = gi(p, b.m.values)
print("baseline params a, b, c =", np.round(p, 4))
print("baseline RMSE (GI points) =", round(float(np.sqrt(np.mean((b.fit-b.gi)**2))), 4))
print("ceiling GI at m->inf =", round(1/(1+p[0]), 3))
for m in [1, 20, 100, 500, 1000, 2000, 5000, 10000, 30000]:
    print(f"  m={m:>6} kg  GI_base={gi(p, m):.3f}")
print(b[["id","m","gi","fit"]].round(3).to_string(index=False))

# Technology multiplier k from FlyZero tank rows and hardware/targets
def k_of(row):
    return (1/row.gi - 1)/r0(p, row.m)
fz = df[df.id.str.startswith("FZ-") & df.gi_scope.eq("tank") & df.m.notna()].copy()
fz["k"] = fz.apply(k_of, axis=1)
fz["year"] = num(fz.tech_year)
print("\nFlyZero implied k(t) (tank-only):")
print(fz.groupby("year").k.agg(["median","min","max"]).round(3))
print(fz[["id","m","gi","k"]].round(3).to_string(index=False))

hw = df[df.data_type.isin(["hardware_measured","hardware_claim","target","requirement"]) & df.m.notna() & df.gi_scope.isin(["tank","system"])].copy()
hw["k"] = hw.apply(k_of, axis=1)
print("\nHardware/targets implied k:")
print(hw[["id","data_type","tech_year","m","gi","k"]].round(3).to_string(index=False))

# FlyZero fuel-system mass fraction s = (1/GI_sys - 1) - (1/GI_tank - 1), using mass-weighted tank GI
fzt = df[df.id.str.startswith("FZ-") & df.gi_scope.eq("tank")].copy()
fzt["mass_ratio"] = (1/fzt.gi - 1)*fzt.m
for concept in ["regional","narrowbody","midsize"]:
    for yr in [2026, 2030, 2050]:
        sub = fzt[fzt.aircraft_class.eq(concept) & num(fzt.tech_year).eq(yr)]
        tank_ratio = sub.mass_ratio.sum()/sub.m.sum()
        sysgi = df[df.id.eq(f"FZ-{concept}-total-{yr}")].gi.astype(float).iloc[0]
        s = (1/sysgi - 1) - tank_ratio
        print(f"fuel-system mass per kg H2  {concept:10s} {yr}: s = {s:.3f}  (tank ratio {tank_ratio:.3f}, system GI {sysgi})")
np.save("prelim_params.npy", p)
