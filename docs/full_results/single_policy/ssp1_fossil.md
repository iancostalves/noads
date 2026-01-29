# SSP1-1.9 Fossil Baseline

Baseline scenario for SSP1-1.9 with no mitigation policies (fossil kerosene only).

## Scenario Description

- **SSP**: SSP1 (Sustainability - Taking the Green Road)
- **RCP**: 1.9 (~1.5°C warming target)
- **Policy**: No mitigation (fossil kerosene only)
- **Energy Carriers**: Jet-A (100% fossil)
- **Technology Scenarios**: Lower, Mid, Upper

## Key Characteristics

**SSP1 Storyline**:
- Low challenges to mitigation and adaptation
- Gradual shift toward sustainable path
- Demand 10% below global trend due to sustainability emphasis
- Population stabilizes earlier than other SSPs
- Income growth moderate

**No Mitigation**:
- Only conventional kerosene turbofan aircraft
- Two generations of new aircraft optimized for EIS timing
- No alternative fuels or aircraft
- Optimization minimizes cumulative CO2 by choosing deployment timing

## Results Summary

### Lower Technology

**Key Metrics**:
- 2070 Traffic: ~50% above 2019
- 2070 Emissions: ~45% above 2019
- Peak Emissions: ~2040 at ~1250 MtCO2/year
- Cumulative 2020-2070: ~52 GtCO2
- Carbon Budget: Exceeded by ~2.0x

### Mid Technology

**Key Metrics**:
- 2070 Traffic: ~50% above 2019
- 2070 Emissions: ~42% above 2019
- Peak Emissions: ~2038 at ~1200 MtCO2/year
- Cumulative 2020-2070: ~50 GtCO2
- Carbon Budget: Exceeded by ~1.9x

### Upper Technology

**Key Metrics**:
- 2070 Traffic: ~50% above 2019
- 2070 Emissions: ~38% above 2019
- Peak Emissions: ~2036 at ~1150 MtCO2/year
- Cumulative 2020-2070: ~48 GtCO2
- Carbon Budget: Exceeded by ~1.8x

## Detailed Results

Result plots are available in: `docs/examples/optimization/single_policy/single_policy/SSP1-19-Fossil-*Tech/`

## Key Observations

1. **Demand Saturation Effect**: SSP1's lower demand growth leads to earlier emissions stabilization
2. **Technology Sensitivity**: Technology assumptions create ~10% variation in 2070 emissions
3. **Fleet Renewal Timing**: Optimal deployment is immediate (2030-2032) for first generation
4. **Carbon Budget**: Even SSP1 baseline exceeds carbon budget by factor of ~2x

## Related Scenarios

- [SSP2-2.6 Fossil Baseline](ssp2_fossil.md)
- [SSP5-4.5 Fossil Baseline](ssp5_fossil.md)
