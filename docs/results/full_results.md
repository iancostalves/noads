# Full Results

This section provides comprehensive results for all simulated scenarios, organized similar to the Supplementary Information.

## Organization

Results are organized by scenario type and technology assumption:

### Baseline Scenarios (No Mitigation)

**SSP1-1.9 Baseline** (Fossil only)
- [SSP1-19-Fossil-lowTech](../examples/optimization/single_policy/single_policy/SSP1-19-Fossil-lowTech/) - Lower technology
- [SSP1-19-Fossil-midTech](../examples/optimization/single_policy/single_policy/SSP1-19-Fossil-midTech/) - Mid technology
- [SSP1-19-Fossil-upTech](../examples/optimization/single_policy/single_policy/SSP1-19-Fossil-upTech/) - Upper technology

**SSP2-2.6 Baseline** (Fossil only)
- [SSP2-26-Fossil-lowTech](../examples/optimization/single_policy/single_policy/SSP2-26-Fossil-lowTech/) - Lower technology
- [SSP2-26-Fossil-midTech](../examples/optimization/single_policy/single_policy/SSP2-26-Fossil-midTech/) - Mid technology
- [SSP2-26-Fossil-upTech](../examples/optimization/single_policy/single_policy/SSP2-26-Fossil-upTech/) - Upper technology

**SSP5-4.5 Baseline** (Fossil only)
- [SSP5-45-Fossil-lowTech](../examples/optimization/single_policy/single_policy/SSP5-45-Fossil-lowTech/) - Lower technology
- [SSP5-45-Fossil-midTech](../examples/optimization/single_policy/single_policy/SSP5-45-Fossil-midTech/) - Mid technology
- [SSP5-45-Fossil-upTech](../examples/optimization/single_policy/single_policy/SSP5-45-Fossil-upTech/) - Upper technology

### Drop-in Mitigation (SAF, 5% allocation)

**Trend Variant** (minimize cumulative CO2)
- [SSP2-26-DropIn-lowTech](../examples/optimization/single_policy/single_policy/SSP2-26-DropIn-lowTech/)
- [SSP2-26-DropIn-midTech](../examples/optimization/single_policy/single_policy/SSP2-26-DropIn-midTech/)
- [SSP2-26-DropIn-upTech](../examples/optimization/single_policy/single_policy/SSP2-26-DropIn-upTech/)

**Availability Variant** (8.6% allocation)
- [SSP2-26-DropIn-Availability-lowTech](../examples/optimization/single_policy/single_policy/SSP2-26-DropIn-Availability-lowTech/)
- [SSP2-26-DropIn-Availability-midTech](../examples/optimization/single_policy/single_policy/SSP2-26-DropIn-Availability-midTech/)
- [SSP2-26-DropIn-Availability-upTech](../examples/optimization/single_policy/single_policy/SSP2-26-DropIn-Availability-upTech/)

**Low-Demand Variant** (demand management)
- [SSP2-26-DropIn-LowDemand-lowTech](../examples/optimization/single_policy/single_policy/SSP2-26-DropIn-LowDemand-lowTech/)
- [SSP2-26-DropIn-LowDemand-midTech](../examples/optimization/single_policy/single_policy/SSP2-26-DropIn-LowDemand-midTech/)
- [SSP2-26-DropIn-LowDemand-upTech](../examples/optimization/single_policy/single_policy/SSP2-26-DropIn-LowDemand-upTech/)

### Breakthrough Mitigation (SAF + Alternative Aircraft, 5% allocation)

**Trend Variant** (minimize cumulative CO2)
- [SSP2-26-lowTech](../examples/optimization/single_policy/single_policy/SSP2-26-lowTech/)
- [SSP2-26-midTech](../examples/optimization/single_policy/single_policy/SSP2-26-midTech/)
- [SSP2-26-upTech](../examples/optimization/single_policy/single_policy/SSP2-26-upTech/)

**Availability Variant** (8.6% allocation)
- [SSP2-26-Availability-lowTech](../examples/optimization/single_policy/single_policy/SSP2-26-Availability-lowTech/)
- [SSP2-26-Availability-midTech](../examples/optimization/single_policy/single_policy/SSP2-26-Availability-midTech/)
- [SSP2-26-Availability-upTech](../examples/optimization/single_policy/single_policy/SSP2-26-Availability-upTech/)

**Low-Demand Variant** (demand management)
- [SSP2-26-LowDemand-lowTech](../examples/optimization/single_policy/single_policy/SSP2-26-LowDemand-lowTech/)
- [SSP2-26-LowDemand-midTech](../examples/optimization/single_policy/single_policy/SSP2-26-LowDemand-midTech/)
- [SSP2-26-LowDemand-upTech](../examples/optimization/single_policy/single_policy/SSP2-26-LowDemand-upTech/)

### Scenario-Robust Policies

Optimized across SSP2-1.9, 2.6, and 3.4 (Mid technology only):

**Trend Variant**
- [robust-SSP2-midTech](../examples/optimization/single_robust_policy/robust_policy/robust-SSP2-midTech/)

**Availability Variant**
- [robust-SSP2-availability-Availability-midTech](../examples/optimization/single_robust_policy/robust_policy/robust-SSP2-availability-Availability-midTech/)

**Low-Demand Variant**
- [robust-SSP2-lowdemand-LowDemand-midTech](../examples/optimization/single_robust_policy/robust_policy/robust-SSP2-lowdemand-LowDemand-midTech/)

**Multi-Scenario Comparison Figures:**
- [Multi-ensemble trends](../examples/optimization/single_robust_policy/robust_policy/multi_ensemble_trends.pdf)
- [Multi-ensemble jet fuel](../examples/optimization/single_robust_policy/robust_policy/multi_ensemble_jet_fuel.pdf)
- [Multi-ensemble fleet carriers](../examples/optimization/single_robust_policy/robust_policy/multi_ensemble_fleet_carriers.pdf)

## File Structure

Each scenario folder typically contains:
- **`fleet_*.pdf`** - Fleet composition by market segment showing:
  - Current aircraft retirement
  - New aircraft deployment (conventional and alternative)
  - Carbon intensity by market
  
- **`energy_*.pdf`** - Energy production by pathway showing:
  - Primary energy sources (oil, biomass, electricity)
  - Intermediate production (biofuels, hydrogen)
  - Final energy carriers (Jet-A, LH2, batteries)
  
- **`conso_*.pdf`** - Energy consumption by carrier showing:
  - Direct consumption by aircraft
  - Total energy consumption over time
  - Consumption breakdown by market
  
- **`demand_*.pdf`** (low-demand variants only) - Demand evolution showing:
  - Trend demand vs. actual demand
  - Supply reduction ratio by market
  - Traffic avoidance patterns

## Summary Comparison Figures

High-level comparison across scenarios:
- [baseline.pdf](../examples/optimization/single_policy/single_policy/baseline.pdf) - All baseline scenarios (SSP1, SSP2, SSP5)
- [dropin.pdf](../examples/optimization/single_policy/single_policy/dropin.pdf) - Drop-in SAF scenarios
- [breakthrough.pdf](../examples/optimization/single_policy/single_policy/breakthrough.pdf) - Breakthrough scenarios with alternative aircraft

## Key Metrics Summary

Approximate values for Mid technology scenarios:

| Scenario | 2070 Traffic vs 2019 | 2070 Emissions vs 2019 | Fossil Phase-out | Carbon Budget |
|----------|---------------------|----------------------|------------------|---------------|
| Baseline SSP1 | +50% | +45% | No | Exceeded 2x |
| Baseline SSP2 | +100% | +80% | No | Exceeded 2.5x |
| Baseline SSP5 | +150% | +120% | No | Exceeded 3x |
| Drop-in trend | +100% | ~0% | No | Exceeded |
| Drop-in avail | +100% | -30% | Partial | Exceeded |
| Drop-in low-demand | +50% | -50% | Yes | Met |
| Breakthrough trend | +100% | -20% | Partial | Close |
| Breakthrough avail | +100% | -55% | Yes | Close |
| Breakthrough low-demand | +140% | -50% | Yes | Met |
| Robust trend | +100% | -18% (mean) | Varied | ~Budget |
| Robust low-demand | +117% (mean) | -43% (mean) | Varied | All met |

## Interpretation Guidance

When viewing results:
1. **Technology sensitivity** - Compare Lower/Mid/Upper for same scenario
2. **Policy effectiveness** - Compare trend vs. availability vs. low-demand
3. **Carrier choice** - Check fleet composition for alternative aircraft deployment
4. **Resource use** - Check energy production for biomass/electricity consumption
5. **Temporal dynamics** - Note timing of transitions and peaks

For detailed interpretation, see [Main Results](main_results.md) and [Discussion](../discussion/index.md).
