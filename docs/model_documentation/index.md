# Model Documentation

This section provides detailed explanations of model assumptions, equations, and calibration methodology used in the aviation decarbonization scenarios.

## Overview

The models implemented in this package cover:

- **Air Traffic Demand**: Estimation of future passenger demand based on socioeconomic drivers
- **Aircraft Fleet**: Current and future aircraft with technology evolution
- **Energy System**: Production pathways for various energy carriers
- **Emissions**: Well-to-wake lifecycle emissions accounting

## Contents

- [Model Assumptions](assumptions.md) - Key assumptions and limitations for each model discipline
- [Model Equations](equations.md) - Mathematical formulation of the models
- [Calibration Methodology](calibration.md) - How models are calibrated to historical data

## Model Disciplines

The main model disciplines are:

1. **Policy Controls**: Time-dependent policy variables (fleet replacement, energy mix, demand caps)
2. **Demand**: Traffic demand evolution based on population and income
3. **Current Fleet**: Existing aircraft with historical performance
4. **New Aircraft**: Prospective designs with evolving technology
5. **Fleet Replacement**: Gradual turnover of aircraft
6. **Energy Mix**: Production of energy carriers from various pathways
7. **Constraints**: Resource availability and feasibility limits
8. **Objectives**: Cumulative emissions or demand management burden

## Data Sources

Models are calibrated using data from:

- ICAO (International Civil Aviation Organization)
- World Bank socioeconomic indicators
- AeroSCOPE flight database [@salgas_aeroscope]
- Planespotters aircraft database [@planespotters]
- AR6 Scenario Database [@ar6_database]
- Aircraft design studies [@icct_hydrogen; @icct_electric; @icct_fuelcell]
- Technology roadmaps [@iata_tech_2050; @ati_cryogenic; @ati_electrical; @ati_fuelcell]

## Model Validation

Models have been validated against:

- Historical trends (1980-2019 for demand)
- Current fleet performance (2019 AeroSCOPE data)
- Technology assessments from literature
- Expert judgment on technology maturation timelines
