# SSP2-2.6 Breakthrough Scenarios

Breakthrough mitigation scenarios for SSP2-2.6 with alternative aircraft (hydrogen and electric) in addition to SAF.

## Scenario Description

- **SSP**: SSP2 (Middle of the Road)
- **RCP**: 2.6 (~2°C warming target)
- **Policy**: Breakthrough mitigation (SAF + alternative aircraft)
- **Energy Carriers**: Jet-A (fossil+SAF), Liquid H2, Batteries
- **Resource Allocation**: 5.0% (trend) or 8.6% (availability) of global biomass and electricity
- **Variants**: Trend, Availability, Low-Demand
- **Technology Scenarios**: Lower, Mid, Upper

## Key Characteristics

**Breakthrough Mitigation**:
- Sustainable Aviation Fuels (biofuels and electrofuels)
- Battery-electric aircraft (shortest ranges)
- Hydrogen fuel cell aircraft (short to medium ranges)
- Hydrogen gas turbine aircraft (all ranges)
- Entry-into-service timing optimized per market

**Three Variants**:

1. **Trend**: Minimize cumulative CO2 with 5% resource allocation, no demand caps
2. **Availability**: Minimize cumulative CO2 with 8.6% resource allocation, no demand caps
3. **Low-Demand**: Minimize demand burden with 5% allocation, meet Paris carbon budget

## Results Summary

### Trend Variant (5% allocation)

**Lower Technology**:
- 2070 Traffic: +100% vs 2019
- 2070 Emissions: +10% vs 2019
- Peak Emissions: ~2038
- Cumulative: Exceeds budget
- Fossil phase-out: Partial by 2070
- Alternative aircraft: Limited deployment, mainly commuter/regional

**Mid Technology**:
- 2070 Traffic: +100% vs 2019
- 2070 Emissions: -20% vs 2019
- Peak Emissions: ~2036
- Cumulative: Close to budget
- Fossil phase-out: Nearly complete by 2070
- Alternative aircraft: Moderate deployment across markets

**Upper Technology**:
- 2070 Traffic: +100% vs 2019
- 2070 Emissions: -50% vs 2019
- Peak Emissions: ~2034
- Cumulative: Within budget range
- Fossil phase-out: Complete by 2065
- Alternative aircraft: Wide deployment, hydrogen dominates medium+ ranges

### Availability Variant (8.6% allocation)

**Lower Technology**:
- 2070 Traffic: +100% vs 2019
- 2070 Emissions: -30% vs 2019
- Fossil phase-out: Yes, by 2070
- Extra resources enable more SAF production
- Alternative aircraft similar to trend variant

**Mid Technology**:
- 2070 Traffic: +100% vs 2019
- 2070 Emissions: -55% vs 2019
- Fossil phase-out: Yes, by 2065
- Can achieve near-zero post-2070
- Balanced SAF and alternative aircraft

**Upper Technology**:
- 2070 Traffic: +100% vs 2019
- 2070 Emissions: -70% vs 2019
- Fossil phase-out: Yes, by 2060
- Near-zero trajectory
- Alternative aircraft dominant in all markets

### Low-Demand Variant (5% allocation + demand caps)

**Lower Technology**:
- 2070 Traffic: ~+100% vs 2019 (less than trend)
- 2070 Emissions: -40% vs 2019
- Cumulative: Meets budget
- Demand caps required but moderate
- Combines all mitigation measures

**Mid Technology**:
- 2070 Traffic: ~+120% vs 2019
- 2070 Emissions: -50% vs 2019
- Cumulative: Meets budget
- Minimal demand management needed
- Alternative aircraft crucial

**Upper Technology**:
- 2070 Traffic: ~+140% vs 2019
- 2070 Emissions: -60% vs 2019
- Cumulative: Meets budget with headroom
- Almost no demand caps needed
- Demonstrates sufficiency of technology

## Fleet Composition

### Market-Specific Patterns

**General Aviation** (shortest range, 3% supply):
- First to adopt alternative aircraft
- Battery-electric becomes dominant with mid/upper tech
- Hydrogen fuel cell with lower tech
- Complete transition by 2050

**Commuter** (1500 km, 22% supply):
- Second wave of adoption (2035-2050)
- Mix of battery-electric and hydrogen fuel cell
- Hydrogen gas turbine with upper tech
- Major contributor to system-wide decarbonization

**Regional** (4500 km, 39% supply):
- Critical market (largest share)
- Primarily hydrogen fuel cell and gas turbine
- Battery-electric not feasible at this range
- Full transition by 2055-2060

**Short-Medium** (8000 km, 21% supply):
- Later transition (2045+)
- Hydrogen gas turbine primary option
- Battery-electric not feasible
- Drop-in SAF bridge until hydrogen ready

**Long Range** (15000 km, 15% supply):
- Last to transition
- Hydrogen gas turbine only alternative
- Requires mature cryogenic tank technology
- May remain on SAF blend longer

### Technology Sensitivity

Fleet composition is highly sensitive to technology assumptions:

- **Lower tech**: Conservative deployment, later EIS, less alternative aircraft share
- **Mid tech**: Balanced deployment across all markets
- **Upper tech**: Aggressive deployment, earlier EIS, high alternative aircraft share

## Energy System

### Energy Carrier Mix

**Jet-A Evolution**:
- Starts 100% fossil
- Biofuel blend increases from 2030
- Electrofuel grows as grid decarbonizes
- Fossil phases out by 2060-2070 (upper tech)

**Liquid H2**:
- Production starts 2035-2040
- Rapid growth 2040-2060
- Becomes dominant carrier with upper tech
- Competes with electrofuels for electricity

**Batteries**:
- Limited to shortest ranges
- Requires high specific energy (>600 Wh/kg)
- Direct use of low-carbon electricity
- Highest efficiency per MJ

### Production Pathway Selection

**Biofuels**:
- HEFA preferred initially (less biomass per MJ)
- FT increases as decarbonization progresses (lower emissions per MJ)
- Limited by biomass availability constraint

**Electrofuels**:
- Depends on grid carbon intensity
- Competes with direct hydrogen use and electrolysis
- Trade-off: use electricity for e-fuels or alternative aircraft?
- Breakthrough scenarios favor alternative aircraft

**Hydrogen**:
- Electrolysis from low-carbon electricity
- Liquefaction for aviation use
- Enables alternative aircraft deployment
- Key to deep decarbonization

## Comparison with Drop-in Scenarios

Breakthrough scenarios offer several advantages:

1. **Lower Emissions**: 20-50% lower than Drop-in with same resource allocation
2. **Fossil Phase-out**: Complete fossil phase-out possible by 2060-2070
3. **Resource Efficiency**: Better use of limited low-carbon energy
4. **Flexibility**: Multiple pathways for different market segments
5. **Long-term**: Benefits increase over time as fleet turns over

Trade-offs:

1. **Technology Risk**: Depends on immature technologies
2. **Infrastructure**: Requires hydrogen infrastructure at airports
3. **Timing**: Benefits delayed until 2040+ (fleet turnover lag)
4. **Complexity**: More technologies to develop and certify

## Detailed Results

Result plots available in:
- `docs/examples/optimization/single_policy/single_policy/SSP2-26-*Tech/`
- `docs/examples/optimization/single_policy/single_policy/SSP2-26-Availability-*Tech/`
- `docs/examples/optimization/single_policy/single_policy/SSP2-26-LowDemand-*Tech/`

Files include:
- `fleet_*.pdf` - Fleet composition by market
- `energy_*.pdf` - Energy production by pathway
- `conso_*.pdf` - Energy consumption by carrier
- `demand_*.pdf` - Demand evolution (low-demand variant only)

## Key Insights

1. **Alternative Aircraft are Critical**: For deep decarbonization beyond what SAF alone can achieve

2. **Technology Maturation Matters**: Factor of 2-3x difference in outcomes between lower and upper tech

3. **Market Segmentation**: Different solutions for different range categories

4. **Resource Competition**: Electricity and hydrogen preferentially allocated to alternative aircraft rather than e-fuels

5. **Timing**: Benefits delayed but compound over time with full fleet replacement

6. **Complementarity**: Alternative aircraft, SAF, and demand management work together

7. **Flexibility Value**: Multiple technology options provide hedging against uncertainty

## Related Scenarios

- [Drop-in Scenarios](dropin.md) - Mitigation without alternative aircraft
- [Robust Breakthrough](../robust_policy/robust_trend.md) - Robust across SSP2 variants
- [SSP2 Baseline](ssp2_fossil.md) - No mitigation comparison

## References

For interpretation:
- [Scenario Trends - Main Results](../../scenario_trends/main_results.md)
- [Scenario Trends - Discussion](../../scenario_trends/discussion.md)

For methodology:
- [Model Documentation](../../model_documentation/index.md)
