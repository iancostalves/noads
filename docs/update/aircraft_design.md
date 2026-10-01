<!--
 Copyright 2025 ISAE-SUPAERO, https://www.isae-supaero.fr/en/
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

(sec-update-aircraft)=

# Aircraft design

The update aircraft model (`aircraft_model="update"`) changes three things in the
prospective aircraft designs of the [main paper](sec-paper-fleet):

1. the Generic Airplane Model is upgraded from V2.0 to V3.0;
2. the electric and fuel cell powertrains depend on the power per propulsor, which is
   bounded by the maximum power available at the entry-into-service (EIS);
3. the gravimetric index of LH2 tanks depends on the tank size and on the EIS.

The technology parameters of the main paper (battery, power electronics, structure)
are otherwise kept, while those of the fuel cell system, its thermal management and
the LH2 tanks are revised from a 2026 literature review. The paper model is left untouched: its designs are checked by a
regression test.

## Generic Airplane Model V3.0

The aircraft are designed with a JAX port of GAM V3.0 {cite:p}`u-gam_v3`
({mod}`noads.gam_jax.models.gam_v3`), which reproduces the upstream designs to
machine precision on the NOADS missions. Compared with GAM V2.0
{cite:p}`u-kambiri_energy_2024`, used in the main paper:

- the reference power and standard empty mass regressions are refitted;
- the lift-to-drag ratio regression distinguishes fan and propeller aircraft;
- the turboshaft specific fuel consumption regression versus power is refitted;
- turbofans get a scale factor for the efficiency loss of small turbomachines,
  $1 - (1 - 0.2) / (1 + P / 5\,\text{MW})^2$, where $P$ is the maximum power per engine.

```{figure} figures/aircraft_update_engine_size.png
:name: fig-update-engine-size
:width: 70%

Overall efficiency of thermal engines versus maximum power per engine, in GAM V2.0
(paper) and V3.0 (update).
```

The turbofan scale factor has the largest effect: turbofans of the general, commuter
and regional markets, sized below a few MW per engine, are much less efficient than
in the main paper, and are now within the range of the 2019 fleet
({numref}`fig-update-prospective-energy`). The short-medium and long-range designs are
nearly unchanged. Turboprops are also added to the comparison of prospective
aircraft, since their efficiency versus size changed too (they are not part of the
optimized fleets).

## Powertrain scale effects

The e-motor specific power $SP$ and the electric chain (motor, inverter,
distribution) efficiency $\eta$ are given at a reference power $P_{ref} = 1$ MW per
propulsor, and scale with the power per propulsor $P$:

```{math}
SP(P) = SP_{ref} \left(\frac{P}{P_{ref}}\right)^{-\beta}, \qquad
1 - \eta(P) = (1 - \eta_{ref}) \left(\frac{P}{P_{ref}}\right)^{-\gamma}.
```

The fuel cell specific power now excludes the thermal management system (TMS). The
TMS is sized by the heat load at rated power, with a specific heat rejection, and it
draws a parasitic power proportional to the heat it rejects, which lowers the
mission efficiency of the fuel cell system.

Electric motors and fuel cells of the power required by large aircraft do not exist
yet. The maximum power per propulsor available at the EIS is therefore a technology
parameter, interpolated in log space, and the ratio of the power per propulsor to
this maximum is a design output, constrained to be at most 1 in the optimization
(`<aircraft>.unit_power_ratio`). Fuel cell aircraft use the number of propulsors of
credible concepts: 2 up to the regional market (as the ZeroAvia ZA600 and ZA2000
powertrains, or ATR 72 retrofit studies), and 4 for short-medium and long range (as
the 2025 Airbus ZEROe concept, 4 propulsors of 2 MW for about 100 seats). Larger
counts (8 to 12) only appear in research concepts. With 4 propulsors, short-medium
and long-range fuel cell aircraft need 5 to 10 MW per propulsor, which delays
them.

The parameters are calibrated on a dataset of demonstrated, certified, projected and
target powertrain components, with the e-motor specific power following the
logistic projections of {cite:t}`u-pastra_2023` delayed by 10, 8 and 6 years (Lower,
Mid, Upper) between demonstration and certified EIS. The data and calibration script
are bundled in `noads/application/aircraft_tech_data/powertrain/`.

| Scenario | $\beta$ (specific power) | $\gamma$ (losses) | Max power per propulsor 2020 / 2040 / 2060 / 2080 [MW] |
|---|---|---|---|
| Lower | 0.25 | 0 | 0.06 / 1.5 / 5 / 9.1 |
| Mid | 0.1 | 0.1 | 0.06 / 3 / 12 / 24 |
| Upper | 0 | 0.25 | 0.06 / 5 / 25 / 56 |

The scale exponents are those of 2040 onwards; they all start from the Mid value in
2020 (see below).

The fuel cell system parameters of the powertrain calibration rest on component
targets, which proved optimistic against the literature: flown aviation fuel cell
systems reach 0.2 to 0.6 kW/kg, automotive systems 0.6 to 0.86 kW/kg, and the 2030
targets about 2 kW/kg; the cruise efficiency at altitude is about 40 %, the peak
efficiency targets of 65 to 72 % being low-load values; and today's TMS rejects 1.5
to 3 kW of heat per kg, about 5 in the FlyZero concepts, with a parasitic power of
about 0.2 to 0.27 W per W of heat. They are revised downwards accordingly:

| Parameter | 2020 | Lower 2040 / 2060 / 2080 | Mid 2040 / 2060 / 2080 | Upper 2040 / 2060 / 2080 |
|---|---|---|---|---|
| Fuel cell specific power, without TMS [kW/kg] | 0.75 | 1.5 / 2.0 / 2.25 | 2.0 / 2.75 / 3.25 | 2.5 / 3.5 / 4.5 |
| Fuel cell system efficiency at cruise [%] | 40 | 44 / 46 / 48 | 46 / 50 / 52 | 48 / 53 / 56 |
| TMS specific heat rejection [kW/kg] | 2 | 3.5 / 5 / 6 | 5 / 8 / 10 | 7 / 12 / 15 |
| TMS parasitic power per unit heat [-] | 0.225 | 0.20 / 0.18 / 0.16 | 0.17 / 0.14 / 0.12 | 0.14 / 0.10 / 0.08 |

The literature values are bundled in
`noads/application/aircraft_tech_data/literature_data.csv` (197 values from 2020 to
2026, with their source, scope and status). Most of them come from abstracts and
summaries rather than full texts, and should be checked before being quoted.

## LH2 tanks

In the main paper, the LH2 tank gravimetric index (GI, mass of LH2 over mass of LH2
and tank) is the same for all aircraft. The data rather show that it increases with
the tank size, from 0.2 to 0.3 for tanks of tens of kilograms to 0.7 to 0.8 for tanks
of tens of tonnes, since the tank mass is driven by its area (insulation, walls)
{cite:p}`u-huete_parametric_2021,u-verstraete_hydrogen_2010`. Splitting the same fuel
into more tanks lowers the GI {cite:p}`u-winnefeld_2018`. The tank-to-fuel mass ratio of
a tank holding $m$ kg of LH2, for an aircraft entering into service in year $t$, is

```{math}
r_{tank}(m, t) = k(t) \left(a + b\, m^{-1/3} + \frac{c}{m}\right), \qquad
GI_{tank} = \frac{1}{1 + r_{tank}}, \qquad
GI_{system} = \frac{1}{1 + r_{tank} + s(t)},
```

where the size law $(a, b, c)$ describes present-day aluminium tanks from design
studies, the mass factor $k$ the maturity of the technology ($k = 1$ for these tanks),
and $s$ the mass of the fuel system (pipes, pumps, conditioning) per kg of LH2. The
LH2 of each aircraft is split into 2 tanks (a per-market parameter), sized for the
maximum fuel. $k$ and $s$ follow logistic curves of the EIS, anchored at their 2020
value $v_0$:

```{math}
v(t) = v_\infty + (v_0 - v_\infty) \frac{\sigma(t)}{\sigma(2020)}, \qquad
\sigma(t) = \frac{1}{1 + e^{(t - t_{50}) / \tau}}.
```

The model is fitted by {mod}`noads.application.lh2_tank_calibration` on a dataset of
112 GI values from 37 sources, bundled in
`noads/application/aircraft_tech_data/lh2_tank_gi/`:

- the size law is fitted on tank-only aluminium design studies up to 2025:
  $a = 0.123$, $b = 4.97\ \text{kg}^{1/3}$, $c = 0$ (RMSE of 0.11 on the GI, dominated
  by the scatter of design assumptions such as dormancy and vent pressure);
- $k_0 = 5.3$ is implied by the stated state of the art of LH2 aircraft tanks, a GI
  of only 20 % for 500 kg of LH2 (OVERLEAF project, 2024). No aviation tank above
  about 100 kg of LH2 has flown with a published GI, and the GI of 0.24 targeted by
  small additively manufactured tanks in 2024-2025 is consistent with it;
- the scenarios share the timing of the maturity, so that they start from the same
  value in 2020 and then only diverge: half of the gain is reached in 2038, the TRL6
  year of the FlyZero composite tanks {cite:p}`u-ati_cryogenic` (2030) delayed by the
  8 years of the Airbus ZEROe programme;
- the scenarios differ by the tank technology they converge to: the ceiling of large
  integrated tanks in the literature (GI of 0.78 at 10 t of LH2, within 0.75 to 0.80,
  Upper), aluminium design-study tanks with dormancy and integration penalties
  (Lower, $k_\infty = 1.5$), and their geometric mean (Mid). The McKinsey system
  targets {cite:p}`u-mckinsey_hydrogen_2020` are nearly independent of the size,
  hence not used;
- $s$ starts from the mean of the FlyZero concepts at their 2026 TRL6 year, and
  converges to the spread of the FlyZero 2050 concepts: the best one (narrowbody,
  Upper), their mean (Mid), and the worst one (regional, with a fuel system sized for
  fuel cells, Lower).

| Scenario | $k_0$ | $k_\infty$ | $t_{50}$ | $s_0$ | $s_\infty$ | $t_{50}$ ($s$) |
|---|---|---|---|---|---|---|
| Lower | 5.34 | 1.50 | 2038 | 0.271 | 0.201 | 2052.1 |
| Mid | 5.34 | 1.09 | 2038 | 0.271 | 0.153 | 2052.1 |
| Upper | 5.34 | 0.80 | 2038 | 0.271 | 0.115 | 2052.1 |

with $\tau = 5$ years. The tank GI is then:

| LH2 per tank | 2020 | Lower 2040 / 2060 / 2080 | Mid 2040 / 2060 / 2080 | Upper 2040 / 2060 / 2080 |
|---|---|---|---|---|
| 75 kg | 0.13 | 0.20 / 0.33 / 0.34 | 0.21 / 0.40 / 0.41 | 0.22 / 0.47 / 0.49 |
| 1.25 t | 0.24 | 0.36 / 0.53 / 0.53 | 0.38 / 0.60 / 0.61 | 0.39 / 0.67 / 0.68 |
| 10.5 t | 0.35 | 0.48 / 0.65 / 0.65 | 0.50 / 0.71 / 0.72 | 0.52 / 0.77 / 0.78 |

```{figure} figures/aircraft_update_lh2_tank_gi.png
:name: fig-update-lh2-tank-gi
:width: 100%

LH2 tank gravimetric index versus the LH2 mass per tank, for several EIS and
technology scenarios, compared with the tank-only values of the dataset.
```

## Technology parameters

The technology parameters of the update model are given for the Lower, Mid and Upper
scenarios at 2020, 2040, 2060 and 2080, and are constant after 2080. Following the
main paper, the scenarios bracket the uncertainty on the maturing technology, which
is null for today's technology: all scenarios share their 2020 value, and the
Upper-to-Lower band only widens with the EIS. To guarantee this:

- the Mid scenario is interpolated with a monotone cubic spline (PCHIP), which never
  overshoots its values, instead of the quadratic spline through 2020, 2040 and 2060
  of the main paper;
- the Lower and Upper scenarios are the Mid curve plus a PCHIP interpolation of their
  gap to Mid, which starts at zero in 2020 and never shrinks
  ({class}`~noads.core.models.fleet.aircraft_tech_parameter.ScenarioTechParameter`);
- the 2040 and 2060 values are those of the main paper or of the powertrain
  calibration (except the revised fuel cell and TMS values above), and the 2080 values extend the calibrated curves (e-motor specific
  power and efficiency), or add half of the 2040-2060 increase. Where a calibrated
  gap to Mid shrinks (fuel cell TMS parasitic power in 2060), it is held at its
  largest value;
- the scale exponents start from the Mid value in 2020 and reach the scenario
  values in 2040.

With technology parameters up to 2080, new aircraft can enter into service until
2080 in the optimization, and the scenarios of the update model run until 2100
instead of 2075.

```{figure} figures/aircraft_update_technology.png
:name: fig-update-technology
:width: 100%

Technology parameters of the update aircraft model (blue) versus EIS, compared with
those of the main paper (grey). Filled between the Upper and Lower scenarios, solid
line for the Lower scenario and dotted line for the Mid scenario. Blue dots are the
powertrain dataset and the literature review, grey dots the data of the main paper;
hollow markers are values of a narrower scope (cell, stack, radiator only). The LH2
tank GI is shown for 1 t of LH2 per tank.
```

## Prospective aircraft

```{figure} figures/aircraft_update_prospective_energy.png
:name: fig-update-prospective-energy
:width: 100%

Energy efficiency of prospective aircraft designed with the update model, versus
EIS. Filled between the Upper and Lower scenarios, solid line for the Lower and
dotted line for the Mid scenario. Electric and fuel cell designs are shown only where
their power per propulsor is available at their EIS, and designs that do not close
are not shown: the band then spans the feasible scenarios only. The grey band shows
the 2019 fleet quartiles.
```

```{figure} figures/aircraft_update_prospective_mass.png
:name: fig-update-prospective-mass
:width: 100%

Empty-mass efficiency of prospective aircraft designed with the update model, with
the same conventions.
```

Compared with the main paper:

- small turbofans (general, commuter and regional) are far less efficient, due to the
  size effect of GAM V3.0, and turboprops become the most efficient thermal
  architecture on these markets;
- fuel cell aircraft are delayed by the maximum power per propulsor and the revised
  fuel cell technology. They close from 2034 on the general market, 2039 on the
  commuter market, 2045 regional, 2046 short-medium and 2060 long range in the Mid
  scenario (2031 to 2047 in Upper, 2041 to 2069 in Lower, where long-range fuel cell
  aircraft never close);
- battery-electric aircraft only close once batteries are mature enough: in the Mid
  scenario, from 2036 on the general market and 2048 on the commuter market;
- with the size law, the tank GI differs strongly between markets
  ({numref}`fig-update-tank-gi-per-market`): for hydrogen gas turbines in the Mid
  scenario, from 0.27 (general) to 0.55 (long range) for an EIS in 2040, and from 0.46
  to 0.74 in 2060.

```{figure} figures/aircraft_update_embarked_energy.png
:name: fig-update-embarked-energy
:width: 100%

Energy embarked per seat and kilometre (energy of the design mission, including
reserves, over seats times design range) of prospective aircraft, with the same
conventions.
```

```{figure} figures/aircraft_update_propulsion_efficiency.png
:name: fig-update-propulsion-efficiency
:width: 100%

Overall efficiency of the propulsion system (from the stored energy to the
propulsive power) of prospective aircraft, with the same conventions.
```

```{figure} figures/aircraft_update_tank_gi_per_market.png
:name: fig-update-tank-gi-per-market
:width: 100%

LH2 tank gravimetric index of the prospective hydrogen aircraft of each market, with
the same conventions, compared with the constant GI of the main paper (grey).
```

```{figure} figures/aircraft_update_fuel_cell_design.png
:name: fig-update-fuel-cell-design
:width: 100%

Design point of the fuel cell aircraft of each market in the Mid scenario, versus
EIS: power per propulsor and its maximum available; electric chain, fuel cell system
(net of the TMS parasitic power) and overall propulsion efficiencies; fuel cell
system specific power including the TMS; e-motor specific power at the design power;
and masses of the fuel cell system and of the LH2 tanks and fuel system relative to
the MTOW. Thick lines where the design is feasible, thin dashed lines elsewhere.
```

The fuel cell design point shows where the remaining optimism lies. Including its
TMS, the fuel cell system only reaches 1 to 2.3 kW/kg, and the propulsion chain 25 to
36 % efficiency, so the fuel cell system remains 10 to 20 % of the MTOW. The electric
chain (above 94 %) and the e-motors (15 to 29 kW/kg at the design power, from the
logistic projections of {cite:t}`u-pastra_2023`) are the most optimistic
assumptions, but they weigh less than the fuel cell system.

## Reproduce these figures

```{eval-rst}
.. minigallery:: examples/models/aircraft/plot_prospective_aircraft_update.py
```

## References

```{bibliography}
:keyprefix: u-
:labelprefix: U
:filter: docname in docnames
```
