# Full Results

This section provides detailed results for all simulated scenarios, organized as an online version of the Supplementary Information.

## Overview

This section presents the complete set of results from the optimization of aviation decarbonization scenarios. For each scenario, detailed plots show:

- Traffic evolution (RPK, ASK)
- Fleet composition by market segment
- Energy mix and production pathways
- Emissions (annual and cumulative)
- Energy and carbon intensity

## Organization

Results are organized by scenario type:

### Baseline Scenarios
- [SSP1-1.9 Fossil](single_policy/ssp1_fossil.md) (Lower, Mid, Upper technology)
- [SSP2-2.6 Fossil](single_policy/ssp2_fossil.md) (Lower, Mid, Upper technology)
- [SSP5-4.5 Fossil](single_policy/ssp5_fossil.md) (Lower, Mid, Upper technology)

### Drop-in Mitigation Scenarios (SSP2-2.6)
- [Drop-in Trend](single_policy/dropin.md) (Lower, Mid, Upper technology)
- [Drop-in Availability](single_policy/dropin_availability.md) (Lower, Mid, Upper technology)
- [Drop-in Low-Demand](single_policy/dropin_lowdemand.md) (Lower, Mid, Upper technology)

### Breakthrough Mitigation Scenarios (SSP2-2.6)
- [Breakthrough Trend](single_policy/breakthrough.md) (Lower, Mid, Upper technology)
- [Breakthrough Availability](single_policy/breakthrough_availability.md) (Lower, Mid, Upper technology)
- [Breakthrough Low-Demand](single_policy/breakthrough_lowdemand.md) (Lower, Mid, Upper technology)

### Scenario-Robust Policies
- [Robust Trend](robust_policy/robust_trend.md) (Mid technology, SSP2 variants)
- [Robust Low-Demand](robust_policy/robust_lowdemand.md) (Mid technology, SSP2 variants)

## Result Files

Results are stored in the following directories:

- `docs/examples/optimization/single_policy/single_policy/` - Single-policy optimization results
- `docs/examples/optimization/single_robust_policy/robust_policy/` - Robust policy results

Each scenario directory contains PDF files with plots for:
- `demand_*.pdf` - Traffic demand evolution (for scenarios with demand management)
- `fleet_*.pdf` - Fleet composition by market segment
- `energy_*.pdf` - Energy production by pathway
- `conso_*.pdf` - Energy consumption by carrier

## Scenario Naming Convention

Scenario names follow the pattern:

```
[SSP]-[RCP]-[Policy]-[Technology]
```

Examples:
- `SSP2-26-Fossil-midTech` - SSP2, RCP 2.6, no mitigation (fossil only), mid technology
- `SSP2-26-DropIn-Availability-upTech` - SSP2, RCP 2.6, drop-in SAF, high availability, upper technology
- `SSP2-26-LowDemand-lowTech` - SSP2, RCP 2.6, breakthrough with demand management, lower technology

Where:
- **SSP**: SSP1, SSP2, or SSP5
- **RCP**: 19 (1.9), 26 (2.6), 34 (3.4), or 45 (4.5)
- **Policy**: Fossil (baseline), DropIn, Availability, LowDemand, or combinations
- **Technology**: lowTech, midTech, or upTech

## Reading the Results

### Fleet Composition Plots

Fleet plots show for each market segment:
- Current aircraft share (retiring over time)
- New conventional aircraft (kerosene turbofan)
- Alternative aircraft:
  - Battery-electric (shortest range only)
  - Hydrogen fuel cell (short to medium range)
  - Hydrogen gas turbine (all ranges)

Color coding indicates aircraft type and energy carrier.

### Energy Mix Plots

Energy plots show:
- Final energy carriers (Jet-A, Liquid H2, Batteries)
- Production pathways (fossil, biofuels, electrofuels, electrolysis, etc.)
- Primary energy inputs (oil, biomass, electricity)

Sankey-style flow diagrams illustrate the energy system structure.

### Consumption Plots

Consumption plots show:
- Energy consumption per carrier over time
- Total energy consumption
- Consumption by market segment

### Demand Plots (Low-Demand Scenarios Only)

Demand plots show:
- Trend demand (no management)
- Actual demand (after supply caps)
- Supply reduction ratio per market

## Key Metrics Summary

For each scenario, key metrics are provided:

| Metric | Description |
|--------|-------------|
| 2070 Traffic vs 2019 | Percentage change in RPK |
| 2070 Emissions vs 2019 | Percentage change in CO2 |
| Peak Emissions Year | Year of maximum annual CO2 |
| Cumulative Emissions 2020-2070 | Total CO2 in Gt |
| Carbon Budget Status | Met / Exceeded by X% |
| Fossil Phase-out | Year of <5% fossil kerosene |
| Alternative Aircraft Share 2070 | % of ASK from H2/battery |

## Comparison Plots

Several comparison plots aggregate across scenarios:

### Baseline Comparison
- All baseline scenarios (SSP1, SSP2, SSP5) with technology sensitivity
- Shows range of no-policy futures

### Technology Sensitivity
- Same policy across Lower/Mid/Upper technology
- Quantifies technology uncertainty impact

### Policy Effectiveness
- Baseline vs Drop-in vs Breakthrough
- Shows incremental benefit of each mitigation lever

### Resource Constraint Impact
- Trend (5%) vs Availability (8.6%)
- Shows value of additional low-carbon energy

### Demand Management
- Trend vs Low-Demand variants
- Shows necessity and impact of demand caps

## Data Access

All underlying data and scripts to generate these plots are available in the repository:
- Raw optimization results: HDF5 format
- Post-processing scripts: Python
- Plot generation: Matplotlib

See the `examples` directory for Jupyter notebooks demonstrating result analysis.

## References

For interpretation and discussion of these results, see:
- [Scenario Trends - Main Results](../scenario_trends/main_results.md)
- [Scenario Trends - Discussion](../scenario_trends/discussion.md)
