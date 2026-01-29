# Calibration Methodology

This page describes how models are calibrated to historical data and validated.

## Overview

Model calibration ensures that projections are grounded in observed historical trends. The main calibration efforts focus on:

1. **Demand calibration**: Fitting logistic function to historical RPK and income data
2. **Fleet calibration**: Matching current aircraft performance to 2019 flight data
3. **Lifetime calibration**: Matching fleet replacement rates to retirement statistics
4. **Energy pathway calibration**: Using literature values for production efficiencies

## Demand Calibration

### Data Sources

- **ICAO**: Global RPK data (1980-2019)
- **World Bank**: Population and GDP per capita data
- **Bergero et al.**: Additional RPK validation data [@bergero_pathways_2023]

### Calibration Period

- **Training**: 1980-2019 (excluding 2020-2023 due to COVID-19)
- **COVID adjustment**: Shift inflection point to account for 2024 recovery to 2019 levels

### Parameter Estimation

The logistic function parameters $\theta = (L, R, \iota, B, C, \nu)$ are calibrated to minimize the 2-norm of the error:

$$\min_{\theta} \left\| RPKpc_{model}(\theta) - RPKpc_{data} \right\|_2$$

**Results**:
- Error-to-data ratio: 10.6%
- $R^2$ on per-capita data: 0.954
- $R^2$ on globally aggregated data: 0.967

### Calibrated Parameter Values

For global demand calibration (exact values depend on normalization):
- Left asymptote $L$: ~100 km/capita/year
- Right asymptote $R$: ~2700-4300 km/capita/year (scenario-dependent)
- Inflection income $\iota$: ~$15,000-20,000 per capita
- Growth rate $B$: calibrated to match transition speed
- Parameters $C$ and $\nu$: adjusted for best fit

### Regional Variations

Regional calibrations show significant variation:
- **Developed regions** (USA, EU, UK): Near saturation with high per-capita RPK
- **Emerging economies** (China, Brazil): Rapid growth phase
- **Developing regions** (India, Bolivia): Early growth phase

However, the global model uses a single calibration for simplicity.

### Load Factor Projection

Load factor grows from 82.4% (2019) to 92% (2075) following a quadratic curve calibrated to match:
- Historical load factor trends
- Industry projections for improved seat utilization

## Current Fleet Calibration

### Data Sources

- **AeroSCOPE**: Flight-by-flight data for 2019 [@salgas_aeroscope]
  - Passengers flown
  - Seating capacity
  - Fuel consumption
  - CO2 emissions
  - Route distance

### Market Segmentation

Flights are aggregated into 5 market segments based on distance:

| Market | Distance Range (km) | Share of ASK | Share of CO2 |
|--------|---------------------|--------------|--------------|
| General | 0-1000 | 3% | 2% |
| Commuter | 1000-2500 | 22% | 19% |
| Regional | 2500-6500 | 39% | 36% |
| Short-medium | 6500-11000 | 21% | 24% |
| Long range | >11000 | 15% | 19% |

### Energy Consumption

For each market, energy consumption per seat-km is estimated from the distribution of actual flights:

- **Lower tech**: Lower quartile (less efficient aircraft)
- **Mid tech**: Median (typical aircraft)
- **Upper tech**: Upper quartile (most efficient aircraft)

This accounts for operational variability and technology heterogeneity within each market.

### Fleet Lifetime Calibration

**Data Source**: Planespotters database [@planespotters]
- Aircraft retirement ages
- Aggregated by number of seats (as proxy for aircraft class)

**Statistics Extracted**:

| Seats | Mean Age | Median Age | Lower Quartile | Upper Quartile |
|-------|----------|------------|----------------|----------------|
| <50 | 28.5 | 22.2 | 15.8 | 33.8 |
| 50-100 | 24.2 | 21.9 | 16.9 | 26.5 |
| 100-150 | 18.7 | 15.3 | 12.1 | 20.5 |
| 150-200 | 29.1 | 24.8 | 19.6 | 33.8 |
| >200 | 26.5 | 23.6 | 19.2 | 29.5 |

**Technology Scenario Mapping**:
- **Lower tech**: Upper quartile (longer lifetimes, slower renewal)
- **Mid tech**: Intermediate value between median and upper quartile
- **Upper tech**: Median (faster renewal)

Note: Lower quartile not used because median is already below mean, indicating aggressive renewal assumption.

### Comparison with Literature

Lifetimes used in comparable studies:
- Grewe et al. [@grewe_evaluating_2021]: 15 years
- Kar et al. [@kar_dynamics_2009]: 20 years
- Delbecq et al. [@delbecq-fleet]: 25 years

Our calibration yields 15.3-33.8 years depending on market and technology scenario.

## Aircraft Technology Parameters

### Data Sources

Multiple sources for technology parameter evolution:

**Academic Literature**:
- Wallington et al. [@wallington_green_2024]: Green transportation technologies
- Adler et al. [@adler_energy_2025]: Energy carrier assessment
- ICCT studies [@icct_hydrogen; @icct_electric; @icct_fuelcell]: Aircraft design analysis

**Industry Roadmaps**:
- IATA [@iata_tech_2050]: Technology aspirations to 2050
- ATI [@ati_aerodynamic; @ati_cryogenic; @ati_electrical; @ati_fuelcell]: UK technology roadmaps

**Government Studies**:
- NASA [@felder_nasa_2015; @papathakis_nasa_2017; @bradley_subsonic_2015]: Electric propulsion studies
- EASA [@easa234]: Type certificate data

### Parameter Ranges

For each technology parameter, lower and upper bounds are extracted from literature based on expected EIS year:

**Example: Battery Specific Energy**

| Source | 2025 | 2040 | 2050+ |
|--------|------|------|-------|
| IATA | 250 | 600 | 1000 |
| ICCT | 200 | 400 | 700 |
| NASA | 300 | 800 | 1500 |
| **Range Used** | **200** | **350-800** | **600-1500** |

### Interpolation

Technology parameters are interpolated using cubic splines between key years (2020, 2040, 2060).

For aircraft with EIS between key years, parameters are interpolated to the specific EIS year.

### Conservative Approach

Overly optimistic projections are excluded to avoid over-reliance on immature technology, especially given aviation's track record of missing environmental targets.

**Example: Fuel Cell Systems**
- Lower bound uses conservative assumptions on specific power (1-3 kW/kg)
- Upper bound uses optimistic but plausible values (3-6 kW/kg by 2060)
- Excluded some literature values >8 kW/kg as too aggressive

## Energy Production Pathways

### Consumption Factors

Energy consumption per unit output from literature:

**Fossil Kerosene** [@jing_understanding_2022]:
- Oil consumption: 1.16 MJ oil / MJ kerosene
- Direct emissions: 88.7 g CO2 / MJ

**Biofuels** [@NEULING201854]:

| Pathway | Biomass (MJ/MJ) | Direct CO2 (g/MJ) |
|---------|----------------|-------------------|
| HEFA | 1.95 | 62.73 |
| ATJ | 3.33 | 51.55 |
| FT | 5.0 | 35.3 |

**Hydrogen and Electrofuels** [@wallington_green_2024]:

| Pathway | 2025 | 2050 | Direct CO2 (g/MJ) |
|---------|------|------|-------------------|
| Electrolysis (MJ elec/MJ H2) | 1.41 | 1.33 | 0 |
| Power-to-Liquid (MJ elec/MJ fuel) | 0.65 | 0.56 | 0 |
| Power-to-Liquid (MJ H2/MJ fuel) | 1.89 | 1.68 | 0 |
| Liquefaction (MJ elec/MJ LH2) | 0.22 | 0.16 | 0 |

### Electricity Grid Carbon Intensity

Scenario-dependent values from AR6 Database [@ar6_database]:
- Time-dependent grid emission factors
- Different pathways for different SSP scenarios
- Used for calculating indirect emissions of electricity-based pathways

## Scenario Background Data

### Socioeconomic Drivers

From AR6 Scenario Database [@ar6_database]:
- Population projections by SSP
- GDP per capita projections by SSP
- Storyline-specific growth patterns

### Energy System

From AR6 Scenario Database [@ar6_database]:
- Global biomass production
- Global electricity production
- Grid carbon intensity evolution
- Pathway-specific to SSP and RCP combinations

### Fair Share Calculation

Aviation's fair share of global energy:
1. Calculate 2019 aviation share of biomass/electricity
2. Apply same percentage to future global production
3. Results: ~5% for trend scenarios, ~8.6% for high-availability scenarios

## Model Validation

### Historical Validation (1980-2019)

- Demand model reproduces historical RPK trends with R² > 0.95
- Fleet energy consumption matches 2019 AeroSCOPE data
- Emissions match ICAO reported values

### Cross-Model Validation

Comparison with AeroMAPS [@planes_aeromaps_2023]:
- Similar demand projections for SSP2 baseline
- Comparable fleet efficiency improvements
- Agreement on SAF production pathway efficiencies

### Sensitivity Analysis

Key parameters tested:
- Demand saturation level: ±20%
- Technology maturation rates: ±5 years
- Fleet lifetimes: ±20%
- Energy pathway efficiencies: ±10%

Results show:
- Demand saturation has largest impact on cumulative emissions
- Technology timing affects optimal deployment strategies
- Energy efficiency has moderate impact on pathway choice

## Uncertainty and Limitations

### Calibration Uncertainties

1. **Extrapolation risk**: Logistic function calibrated on 1980-2019 data, used for projections to 2070
2. **COVID impact**: Uncertainty in recovery trajectory and long-term behavioral changes
3. **Regional aggregation**: Global calibration masks regional variations
4. **Technology forecasting**: High uncertainty in long-term technology performance

### Data Limitations

1. **Incomplete coverage**: AeroSCOPE covers commercial flights, not private aviation
2. **Aggregation**: Market segments are coarse approximations of diverse operations
3. **Historical bias**: Past trends may not continue (peak oil, climate policy shifts)

### Validation Constraints

1. **No future data**: Cannot validate projections beyond 2019
2. **Model structure**: Simplified representation of complex socio-technical system
3. **Exogenous assumptions**: Background scenarios not optimized, taken as given

## References

For detailed data sources and calibration results, see:
- Supplementary Information, Section 1 (Models)
- Figures showing calibration fits and residuals
