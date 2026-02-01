# Aircraft Models

## Overview

Aircraft models cover both current fleet characteristics and prospective aircraft design for alternative propulsion architectures.

## Current Aircraft Fleet

### Market Segmentation

Fleet divided into 5 market segments based on design range:

| Market | Range (km) | Seats | Consumption (MJ/seat-km) | Lifetime (years) |
|--------|-----------|-------|-------------------------|------------------|
| General | 500 | 19 | 2.74-1.47 | 33.8-22.2 |
| Commuter | 1500 | 50 | 1.25-0.88 | 26.5-21.9 |
| Regional | 4500 | 80 | 0.87-0.73 | 20.5-15.3 |
| Short-medium | 8000 | 120 | 1.00-0.82 | 33.8-24.8 |
| Long range | 15000 | 250 | 1.03-0.83 | 29.5-23.6 |

### Technology Scenarios

Three technology scenarios affect current fleet:
- **Lower tech**: Lower quartile efficiency, upper quartile lifetimes (slower renewal)
- **Mid tech**: Median efficiency, intermediate lifetimes
- **Upper tech**: Upper quartile efficiency, median lifetimes (faster renewal)

### Data Sources

- **AeroSCOPE**: Flight-by-flight data for 2019 [@salgas_aeroscope]
  - Passengers flown, seating capacity, fuel consumption
  - Emissions per route
- **Planespotters**: Aircraft retirement ages [@planespotters]
  - Aggregated by number of seats

## Prospective Aircraft Design

### Propulsion Architectures

Four architectures considered:
1. **Conventional kerosene turbofan**
2. **Liquid-hydrogen turbofan**
3. **Liquid-hydrogen fuel-cell with electric motors**
4. **Battery-electric**

### Top-Level Aircraft Requirements (TLAR)

**By Architecture:**

| Architecture | Speed (Mach) | Altitude (1000 ft) |
|--------------|--------------|-------------------|
| Jet-A Gas Turbine | 0.75 | 27 |
| LH2 Gas Turbine | 0.75 | 27 |
| Battery E-motor | 0.5 | 20 |
| LH2 Fuel-cell E-motor | 0.5 | 20 |

**By Market:** Range and seats determined by market segment (see table above).

### Technology Evolution

Component-level parameters evolve over time:

| Technology Parameter | Unit | 2020 | 2040 | 2060 |
|---------------------|------|------|------|------|
| Battery Specific Energy | Wh/kg | 200 | 350-800 | 600-1500 |
| E-motor Specific Power | kW/kg | 2 | 10-25 | 15-28 |
| Electronics Specific Power | kW/kg | 2 | 15-25 | 20-32 |
| Fuel cell Specific Power | kW/kg | 1 | 2-3 | 3-6 |
| Fuel cell Efficiency | % | 40 | 45-55 | 50-65 |
| LH2 tank Gravimetric Index | % | 20 | 30-65 | 35-80 |
| Structural weight reduction | % | 0 | 15-30 | 20-40 |

**Interpolation:** Values interpolated using cubic splines between key years.

### Design Tool

**Generic Airplane Model (GAM)** [@kambiri_energy_2024]: Conceptual design tool using:
- Regression of historical airplane data
- Propulsion system mass calculation
- Performance estimation for each architecture

## Assumptions

**Key Assumptions:**
- TLAR fixed within markets (size, speed, altitude not optimized)
- Market segmentation by distance remains constant
- Aircraft designed specifically for market range
- Technology parameters follow predefined evolution paths

**Limitations:**
- Cryogenic tank efficiency fixed across markets
- Gas turbine weight/consumption assumed identical for kerosene and hydrogen
- Airlines may operate aircraft at non-optimal distances
- No network effects or operational constraints

## Data Sources

Technology parameter ranges from:
- **Academic literature**: [@wallington_green_2024; @adler_energy_2025]
- **ICCT studies**: [@icct_hydrogen; @icct_electric; @icct_fuelcell]
- **Industry roadmaps**: [@iata_tech_2050; @ati_cryogenic; @ati_electrical; @ati_fuelcell]
- **NASA studies**: [@felder_nasa_2015; @papathakis_nasa_2017; @bradley_subsonic_2015]
- **EASA**: Type certificate data [@easa234]

## Visualization

For interactive plots of aircraft performance and technology evolution, see:
[Aircraft Examples](../../examples/models/aircraft/)

Available plots:
- `current_aircraft_energy.pdf` - Current fleet energy consumption distribution
- `current_aircraft_lifetime.pdf` - Fleet lifetime distribution
- `aircraft_prospective_energy.pdf` - Prospective aircraft energy consumption
- `aircraft_prospective_mass.pdf` - Prospective aircraft mass breakdown
- `aircraft_technology.pdf` - Technology parameter evolution over time
- `plot_prospective_aircraft.py` - Interactive script for aircraft design
