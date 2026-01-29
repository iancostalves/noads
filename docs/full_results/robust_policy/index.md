# Robust Policy Results

Results from scenario-robust optimization across multiple background scenarios (SSP2-1.9, SSP2-2.6, SSP2-3.4).

## Overview

Robust scenarios optimize a **single policy** that performs well across multiple future scenarios, rather than optimizing for a single expected future.

**Background Scenarios**:
- SSP2-1.9: Ambitious mitigation (~1.5°C)
- SSP2-2.6: Moderate mitigation (~2°C) - target scenario
- SSP2-3.4: Limited mitigation (~3°C)

**Key Differences Across Scenarios**:
- Electricity availability
- Grid carbon intensity
- Radiative forcing target

## Variants

### Robust Trend
- **Objective**: Minimize mean cumulative CO2 across three scenarios
- **Resource Allocation**: 5% of global biomass and electricity
- **Demand**: Trend growth (no caps)
- **Technology**: Mid only

### Robust Low-Demand
- **Objective**: Minimize mean burden of demand management across three scenarios
- **Resource Allocation**: 5% of global biomass and electricity
- **Demand**: Variable with caps to meet carbon budget in all scenarios
- **Technology**: Mid only

## Robust Trend Results

### Policy Strategy

The robust policy sizes alternative aircraft deployment for the **worst-case scenario (SSP2-3.4)**:
- Less alternative aircraft than SSP2-2.6 single-scenario optimum
- Uses surplus low-carbon electricity in better scenarios for electrofuels
- Maintains flexibility for variability

### Performance by Scenario

**SSP2-1.9** (best case - low carbon grid):
- Higher electrofuel deployment (grid is clean)
- Higher FT biofuel shares (can afford higher biomass)
- Higher overall biofuel in Jet-A
- Lower emissions than single-scenario optimum

**SSP2-2.6** (target):
- **Higher emissions than single-scenario optimum** (trade-off for robustness)
- Moderate alternative aircraft deployment
- Balanced SAF and alternative aircraft
- Still achieves deep decarbonization

**SSP2-3.4** (worst case - high carbon grid):
- Less alternative aircraft (grid is dirty)
- More reliance on biofuels
- Higher fossil consumption
- Demonstrates policy resilience

### Key Metrics

| Scenario | 2070 Emissions vs 2019 | Cumulative CO2 | Fossil Phase-out |
|----------|----------------------|----------------|------------------|
| SSP2-1.9 | -45% | Within budget | Yes, by 2065 |
| SSP2-2.6 | -15% | Close to budget | Partial |
| SSP2-3.4 | +5% | Exceeds budget | No |
| **Mean** | **-18%** | **~Budget** | **Varied** |

Compare to **SSP2-2.6 single-scenario optimum**: -25% emissions in 2070

**Trade-off**: Sacrifice 10% in target scenario to gain flexibility across range

## Robust Low-Demand Results

### Policy Strategy

Combines demand management with technology flexibility:
- Demand caps sized for worst-case resource availability
- Alternative aircraft deployment hedged against grid uncertainty
- SAF production balanced across scenarios

### Performance by Scenario

All three scenarios meet their respective carbon budgets through combination of:
- Moderate demand caps (~15-25% below trend)
- Alternative aircraft deployment
- SAF production matching availability

### Key Metrics

| Scenario | 2070 Traffic vs 2019 | 2070 Emissions vs 2019 | Budget Status |
|----------|---------------------|----------------------|---------------|
| SSP2-1.9 | +130% | -55% | Met |
| SSP2-2.6 | +120% | -45% | Met |
| SSP2-3.4 | +100% | -30% | Met |
| **Mean** | **+117%** | **-43%** | **All met** |

**Demand burden** is relatively balanced across scenarios due to optimization

## Comparison: Single-Scenario vs Robust

### Emissions Trade-off

For SSP2-2.6 (target scenario):

| Policy | 2070 Emissions vs 2019 | Cumulative CO2 |
|--------|----------------------|----------------|
| Single-scenario trend | -25% | Close to budget |
| Robust trend | -15% | Close to budget |
| **Difference** | **+10%** | **~0%** |

Cost of robustness: ~10% higher emissions in expected scenario
Benefit of robustness: Much better performance if SSP2-1.9 or SSP2-3.4 occurs

### Fleet Composition Differences

**Single-scenario optimum (SSP2-2.6)**:
- Sized for moderate grid carbon intensity
- Aggressive alternative aircraft deployment
- Moderate electrofuel production

**Robust optimum**:
- Sized for high grid carbon intensity (worst case)
- Conservative alternative aircraft deployment
- Higher electrofuel production in good scenarios
- More flexibility in energy mix

## Insights on Robustness

### When Robustness Matters

Robust optimization valuable when:
1. **High scenario uncertainty**: Don't know which SSP will occur
2. **Irreversible decisions**: Aircraft and infrastructure are long-lived
3. **Regret aversion**: Want to avoid worst-case outcomes
4. **Flexibility value**: Can adapt energy mix to realized scenario

### Robustness Mechanisms

The policy achieves robustness through:

1. **Conservative aircraft sizing**: For worst-case electricity constraints
2. **Flexible energy mix**: Can substitute electrofuels for hydrogen
3. **Demand hedging** (low-demand variant): Can adjust across scenarios
4. **Technology portfolio**: Multiple pathways provide options

### Cost of Robustness

Trade-offs include:
- Slightly higher emissions in target scenario (~10%)
- Underutilization of clean electricity if SSP2-1.9 occurs
- May forgo some beneficial alternative aircraft deployment

### Value of Robustness

Benefits include:
- Resilience to scenario uncertainty
- Avoids worst-case failure modes
- Maintains performance across wide range
- Provides adaptation pathways

## Detailed Results

Result files available in:
- `docs/examples/optimization/single_robust_policy/robust_policy/robust-SSP2-lowdemand-LowDemand-midTech/`

Contains:
- Individual scenario results (SSP2-19, SSP2-26, SSP2-34 subdirectories)
- Mean results across scenarios
- Ensemble comparison plots

## Policy Implications

### For Policymakers

1. **Scenario uncertainty is real**: Don't optimize for single scenario
2. **Flexibility has value**: Build in adaptation mechanisms
3. **Conservative sizing**: Size for worst case, adapt for best case
4. **Monitor and adapt**: Plan for review points to adjust policy

### For Industry

1. **Portfolio approach**: Develop multiple technology options
2. **Infrastructure flexibility**: Don't lock into single pathway
3. **Staged deployment**: Adapt aircraft orders to realized scenario
4. **Supply chain resilience**: Multiple SAF pathways reduce risk

### For Researchers

1. **Robust optimization methods**: Important for long-term planning
2. **Scenario ensembles**: Better than single best-guess
3. **Real options**: Maintain flexibility for learning
4. **Adaptive management**: Design for adjustment over time

## Related Results

- [Breakthrough Trend (SSP2-2.6)](../single_policy/breakthrough.md) - Single-scenario comparison
- [Scenario Trends - Main Results](../../scenario_trends/main_results.md) - Overall context

## References

For methodology:
- [Introduction and Methods](../../scenario_trends/intro_methods.md)
- [Discussion - Robust Optimization](../../scenario_trends/discussion.md)
