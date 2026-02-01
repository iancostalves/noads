# Demand Model

## Overview

The demand model estimates future air traffic based on socioeconomic drivers (population and income per capita) using logistic functions calibrated on historical data.

## Model Formulation

### Trend Demand

Per-capita demand as a function of per-capita income:

$$RPKpc_{trend}(t) = L + (R-L) \left( \frac{C}{C+\exp(-B(I(t)-\iota))} \right)^{1/\nu}$$

**Parameters:**
- $L$: Left asymptote (propensity to travel at zero income)
- $R$: Right asymptote (propensity to travel at infinite income)
- $\iota$: Income per capita at inflection point
- $B$: Logistic growth rate
- $C$: Transition duration parameter
- $\nu$: Asymptote preference parameter

### Storyline Multiplier

SSP-specific adjustment to trend demand:

$$\frac{RPKpc_{story}(t)}{RPKpc_{trend}(t)} = 1 + \frac{F_{SSP}}{1+\exp\left(- \frac{I(t)-2I_{2024}}{2I_{2024}}\right)}$$

where $F_{SSP}$ is storyline-dependent:
- SSP1: $F_{SSP} = -0.1$ (10% below trend)
- SSP2: $F_{SSP} = 0$ (follows trend)
- SSP5: $F_{SSP} = 0.5$ (50% above trend)

### Total Demand

$$RPK_{trend}(t) = Pop(t) \times RPKpc_{story}(t)$$

### Load Factor

Converts passenger-km to seat-km:

$$LF(t) = \frac{RPK_{trend}(t)}{ASK_{trend}(t)}$$

Load factor grows quadratically from 82.4% (2019) to 92% (2075).

## Calibration

### Data Sources

- **ICAO**: Global RPK data (1980-2019)
- **World Bank**: Population and GDP per capita
- **Bergero et al.**: Additional RPK validation [@bergero_pathways_2023]

### Calibration Period

- **Training**: 1980-2019 (excluding 2020-2023 due to COVID-19)
- **COVID adjustment**: Shift inflection point to account for 2024 recovery

### Results

- Error-to-data ratio: 10.6%
- $R^2$ on per-capita data: 0.954
- $R^2$ on globally aggregated data: 0.967

## Assumptions

**Key Assumptions:**
- Historical relationships between income and air travel remain unchanged
- Demand saturates at high income levels (logistic function)
- Load factor continues improving to 2075
- Global aggregation (no regional detail)

**Limitations:**
- Extrapolation beyond calibrated income range
- Saturation levels uncertain for developing countries
- No network effects or route-specific factors
- Price elasticity not coupled with energy costs

## Visualization

For interactive calibration plots and scenario projections, see:
[Demand Examples](../../examples/models/demand/)

Available plots:
- `ar6_data.pdf` - Comparison with AR6 scenario database
- `plot_demand_calibration.py` - Calibration to historical data
- `plot_demand_scenarios.py` - SSP scenario projections
