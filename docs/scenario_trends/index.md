# Scenario Trends

This section presents an online version of the main paper results, showing the scenario trends for aviation decarbonization.

## Overview

This page presents the main results from numerical optimization of aviation decarbonization scenarios. The scenarios explore:

- **Baseline scenarios**: No-policy projections under different SSP narratives
- **Drop-in scenarios**: Mitigation using Sustainable Aviation Fuels (SAF)
- **Breakthrough scenarios**: Mitigation including alternative aircraft (hydrogen, electric)
- **Robust scenarios**: Policies optimized across multiple future scenarios

For full detailed results with additional plots, see the [Full Results](../full_results/index.md) section.

## Contents

- [Introduction and Methods](intro_methods.md) - Context and methodology
- [Main Results](main_results.md) - Key findings from scenario optimization
- [Discussion](discussion.md) - Interpretation and implications

## Key Findings

1. **Emissions Peak Around 2040**: Across all baseline scenarios, emissions peak around 2040 due to fleet replacement outpacing demand growth

2. **Technology Choice is Scenario-Dependent**: The optimal energy carrier mix depends heavily on assumptions about aircraft technology and energy system constraints

3. **Paris Alignment Requires Demand Management**: Achieving alignment with the Paris Agreement (+2°C target) requires either:
   - Targeted demand management policies, OR
   - Additional low-carbon energy supply beyond fair-share allocations

4. **Efficiency of Gradient-Based Optimization**: Numerical methods enable policy optimization that was previously prohibitive, reducing computation time from hours to minutes

## Scenario Summary

The following scenarios were simulated:

| Scenario | Background | Demand | Energy Carriers | Energy Constraint | Objective |
|----------|-----------|--------|-----------------|-------------------|-----------|
| **Baseline SSP1** | SSP1-1.9 | Trend -10% | Fossil Jet-A | None | Min cumulative CO2 |
| **Baseline SSP2** | SSP2-2.6 | Trend | Fossil Jet-A | None | Min cumulative CO2 |
| **Baseline SSP5** | SSP5-4.5 | Trend +50% | Fossil Jet-A | None | Min cumulative CO2 |
| **Drop-in trend** | SSP2-2.6 | Trend | Jet-A (fossil+SAF) | 5.0% | Min cumulative CO2 |
| **Drop-in availability** | SSP2-2.6 | Trend | Jet-A (fossil+SAF) | 8.6% | Min cumulative CO2 |
| **Drop-in low-demand** | SSP2-2.6 | Variable | Jet-A (fossil+SAF) | 5.0% | Min rel. price increase |
| **Breakthrough trend** | SSP2-2.6 | Trend | Jet-A, LH2, Battery | 5.0% | Min cumulative CO2 |
| **Breakthrough availability** | SSP2-2.6 | Trend | Jet-A, LH2, Battery | 8.6% | Min cumulative CO2 |
| **Breakthrough low-demand** | SSP2-2.6 | Variable | Jet-A, LH2, Battery | 5.0% | Min rel. price increase |
| **Robust trend** | SSP2 (1.9, 2.6, 3.4) | Trend | Jet-A, LH2, Battery | 5.0% | Min mean CO2 |
| **Robust low-demand** | SSP2 (1.9, 2.6, 3.4) | Variable | Jet-A, LH2, Battery | 5.0% | Min mean rel. price |

Notes:
- Energy constraint shows % of global biomass/electricity production allocated to aviation
- Each scenario (except robust) was run with 3 technology assumptions (Lower, Mid, Upper)
- Robust scenarios use Mid technology only

## Visual Summary

### Scenario Trends Comparison

Key trends for baseline, drop-in, and breakthrough scenarios showing:
- Traffic (RPK and ASK)
- Annual CO2 emissions
- Cumulative CO2 emissions
- Carbon intensity (g CO2/RPK)
- Energy intensity (MJ/RPK)

See [Main Results](main_results.md) for detailed figures.

### Energy Mix Evolution

Sankey diagrams showing energy flows from primary sources to final consumption for:
- Baseline: Fossil kerosene only
- Drop-in: Fossil kerosene + SAF (biofuels and electrofuels)
- Breakthrough: Multiple carriers (Jet-A, LH2, batteries)

See [Main Results](main_results.md) for detailed figures.

### Fleet Composition

Aircraft fleet composition by market segment showing:
- Current aircraft retirement
- New conventional aircraft deployment
- Alternative aircraft (hydrogen, electric) deployment
- Carbon intensity by market

See [Main Results](main_results.md) for detailed figures.

## Impact of Technology Assumptions

Technology scenarios significantly impact results:

**Lower Technology**:
- Slower fleet renewal (longer lifetimes)
- Less efficient new aircraft
- Higher emissions for same policy
- Favors earlier deployment of alternative aircraft

**Mid Technology**:
- Moderate renewal rates
- Mid-range efficiency improvements
- Balanced deployment strategies

**Upper Technology**:
- Faster fleet renewal (shorter lifetimes)
- More efficient new aircraft
- Lower emissions for same policy
- Can delay alternative aircraft deployment

## Computational Performance

Numerical methods enable efficient optimization:

- **Standard approach**: ~372s per scenario evaluation, ~363s for linearization
- **JAX-accelerated**: ~0.45s per scenario evaluation, ~0.48s for linearization
- **Speedup**: ~830x for evaluation, ~750x for linearization

This enables:
- Multi-scenario robust optimization
- Extensive sensitivity analysis
- Rapid iteration during model development

See [Introduction and Methods](intro_methods.md) for details on numerical methodology.
