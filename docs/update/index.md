<!--
 Copyright 2025 ISAE-SUPAERO, https://www.isae-supaero.fr/en/
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

(sec-update)=

# Update of the model

This section documents the update of the models of the [main paper](../paper/index.md).
The update revises the aircraft design and some energy and technology assumptions, and
re-runs the optimizations with it. It is opt-in: the main paper and its results remain
the default and stay reproducible.

## Summary

- **Aircraft design** is upgraded from the Generic Airplane Model (GAM) V2.0 to V3.0,
  with powertrain scale effects, a maximum power per propulsor, a revised fuel cell
  system and a size-dependent LH2 tank mass.
- **Technology scenarios** use four time points (2020, 2040, 2060 and 2080) instead of
  three, and the engine efficiency follows a trend with entry into service.
- **Energy pathways** are those of the main paper, and so are the scenarios, demand
  model and optimizer. The technology scenarios vary aircraft technology only.
- **Entry into service** of new aircraft is bounded below by 2035 for Jet-A aircraft and
  2040 for alternative aircraft, and the LH2 gas turbine and LH2 fuel cell start at the
  same date. Turboprops are not part of the optimized fleets.
- **Results** are the 29 optimizations of the main paper, re-run with the update. All
  of them end feasible on the paper timeline (scenarios to 2075, entry into service up
  to 2060).

In short, the update emits more than the main paper in every scenario, and the ranking
of the scenarios is unchanged. The [comparison document](comparison/update_vs_paper_comparison.pdf)
details the differences, organised like the paper's Methods and Results.

## What changes compared with the main paper

| Update | Status | Page |
|---|---|---|
| Aircraft design with GAM V3.0 | available | [Aircraft design](aircraft_design.md) |
| Powertrain scale effects and maximum power per propulsor | available | [Aircraft design](aircraft_design.md) |
| LH2 tank gravimetric index depending on tank size and technology | available | [Aircraft design](aircraft_design.md) |
| Primary energy of energy carriers and aircraft | available | [Energy pathways](energy_pathways.md) |
| Updated optimization results (paper timeline) | available | {ref}`Results <sec-update-results>` |
| Geological hydrogen | planned | [Geological hydrogen](geological_hydrogen.md) |
| Extended timeline (entry into service to 2080, end year 2100) | planned | |

## Selecting the update models

The aircraft design model is selected with the `aircraft_model` argument, `"paper"`
(default) or `"update"`, of
{func}`~noads.application.base_objects.initialize_base_objects`,
{func}`~noads.application.scenario_setup.single_scenario_setup`,
{func}`~noads.application.scenario_setup.multi_scenario_setup` and of the example
drivers of {mod}`noads.application.examples`:

```python
from noads.application.scenario_setup import single_scenario_setup

scenario, design_space, constraints, energy_mix, fleet = single_scenario_setup(
    "SSP2-26-update",
    "SSP2-26",
    technology_index=1,
    aircraft_model="update",
)
```

The update model keeps the timeline of the main paper by default (scenarios until 2075,
new aircraft entering into service until 2060), so that results are comparable. Its
technology parameters are defined until 2080, and `end_year` can be extended up to
2100.

The optimization results of the update model are stored in an `update` subfolder of
the results directory, so that they never overwrite the pre-computed results of the
main paper.

(sec-update-results)=

## Results

### Comparison with the main paper

Cumulative CO₂ is given as a percentage of the 3 % share of the 2.0 °C carbon budget
allocated to aviation (Mid technology). Each cell reads main paper → update.

| Scenario | Cumulative CO₂ [% of budget] | Emissions 2075 [× 2019] | Avoided demand 2075 [%] |
|---|---|---|---|
| Baseline SSP1 | 220 → 276 | 1.49 → 2.04 | 0 → 0 |
| Baseline SSP2 | 231 → 290 | 1.75 → 2.39 | 0 → 0 |
| Baseline SSP5 | 317 → 398 | 2.26 → 3.08 | 0 → 0 |
| Drop-in, trend | 189 → 247 | 1.12 → 1.75 | 0 → 0 |
| Drop-in, availability | 162 → 219 | 0.67 → 1.30 | 0 → 0 |
| Drop-in, low demand | 100 → 100 | 0.29 → 0.36 | 47 → 58 |
| Breakthrough, trend | 166 → 216 | 0.69 → 1.01 | 0 → 0 |
| Breakthrough, availability | 130 → 177 | 0.11 → 0.32 | 0 → 0 |
| Breakthrough, low demand | 100 → 100 | 0.08 → 0.13 | 39 → 45 |

The low-demand variants stay within the budget in both models, but the update needs
more avoided demand to do so. The breakthrough scenarios keep their advantage over the
drop-in scenarios, although a smaller one.

### New aircraft

The energy per seat-km of the new aircraft is higher in the update for the general and
commuter markets, where the GAM V3.0 penalty on small turbofans applies. Fuel cell
aircraft are limited by their maximum power and are unavailable in some markets.

```{figure} figures/design_energy_per_ask.png
:width: 100%

Energy per seat-km of the new aircraft versus entry into service, per market, Mid
technology. Dashed: main paper; solid: update; dotted grey: 2019 fleet. Gaps show
designs that do not close or exceed the power limit.
```

### Fleet mix of the breakthrough variants

Fleet mix of the three variants of the breakthrough case (trend, availability and
low-demand) for each technology scenario, under SSP2-2.6.

```{figure} figures/fleet_mix_update_lower.png
:width: 100%

Fleet mix, Lower technology.
```

```{figure} figures/fleet_mix_update_mid.png
:width: 100%

Fleet mix, Mid technology.
```

```{figure} figures/fleet_mix_update_upper.png
:width: 100%

Fleet mix, Upper technology.
```

### Where to find the results

| What | Where |
|---|---|
| Update optima (27 single-scenario, 2 robust) | `docs/examples/optimization/results/update/<name>/opt_result.json` |
| Main paper optima | `docs/examples/optimization/results/<name>/opt_result.json` |
| Names | `<scenario>-<tech>`, with tech `lowTech`, `midTech` or `upTech` |
| Comparison document and its scripts | `docs/update/comparison/` (PDF, `results_summary.csv`, `build_pdf.py`) |
| Update figures | `docs/update/figures/` |
| Iteration archives and videos | Local only, not in Git (`update/<name>/iterations.npz`, `*.mp4`) |

Each `opt_result.json` holds the optimal `inputs` (controls and entry into service) and
the `outputs` as time series from 2025 to 2075.

### Reproducing the comparison

From the repository root:

```bash
pip install -e ".[test]" reportlab matplotlib
cd docs/update/comparison
python build_pdf.py
```

This regenerates the figures, `results_summary.csv` and the PDF from the committed
optima. To recompute an optimum, run `precompute_update.py` in
`docs/examples/optimization/results`.

```{toctree}
:hidden:

aircraft_design
energy_pathways
geological_hydrogen
```
