# Energy Mix Model

## Overview

The energy mix model tracks production pathways for various energy carriers, including lifecycle (well-to-wake) emissions.

## Energy Carriers

**Final Energy Carriers:**
- Jet-A (fossil + SAF blend)
- Liquid hydrogen
- Batteries (electricity)

**Intermediate Energies:**
- Gaseous hydrogen
- Electrofuels (synthetic kerosene)

**Primary Inputs:**
- Oil (fossil kerosene)
- Biomass (biofuels)
- Electricity (grid)

## Production Pathways

| Energy Carrier | Pathway | Energy Consumption (MJ/MJ) | Direct Emissions (g CO2/MJ) |
|---------------|---------|---------------------------|---------------------------|
| Fossil Jet-A | Refinery | 1.16 oil | 88.7 |
| Biofuel | HEFA | 1.95 biomass | 62.73 |
| Biofuel | ATJ | 3.33 biomass | 51.55 |
| Biofuel | FT | 5.0 biomass | 35.3 |
| Gas H2 | Electrolysis | 1.41-1.33 electricity | 0 |
| Electrofuel | Power-to-liquid | 0.65-0.56 elec, 1.89-1.68 H2 | 0 |
| Liquid H2 | Liquefaction | 0.22-0.16 elec, 1.0 H2 | 0 |

**Note:** Ranges indicate technology maturation from 2025 to 2050.

## Model Formulation

### Impact Factor Calculation

Impact per unit production (intensive property):

$$IF_{pathway} = IF^{direct}_{pathway} + \sum_{i \in inputs} CF_{pathway,i} \times IF_i$$

where:
- $IF^{direct}$ is direct impact of process
- $CF_{pathway,i}$ is consumption of input $i$ per unit output
- $IF_i$ is impact factor of input $i$

### Mixed Energy Impact

Weighted average by pathway shares:

$$IF_{energy} = \sum_{p \in pathways} S_p \times IF_p$$

### Production Calculation

Production of each energy type (extensive property):

$$P_{energy} = C^{direct}_{energy} + \sum_{o \in outputs} CF_{o,energy} \times P_o$$

Computed from final-to-primary (backward).

### Pathway Production

$$P_{pathway} = S_{pathway} \times P_{energy}$$

### Input Consumption

$$C_{energy} = \sum_{p \in pathways} CF_{p,energy} \times P_p$$

## Modular Implementation

**Intensive properties** (impact factors): Computed primary-to-final
- Impacts of inputs must be known before accounting indirect impacts of outputs

**Extensive properties** (production/consumption): Computed final-to-primary
- Total consumption of final energies determines required input consumption

This separation avoids coupling and enables direct computation without iteration.

## Assumptions

**Key Assumptions:**
- Well-to-wake lifecycle emissions considered
- Constant carbon intensities for fossil and biofuels
- Electricity-based pathways use scenario-dependent grid intensity
- Fair share constraint on biomass and electricity (5.0% or 8.6%)

**Limitations:**
- No differentiation between biomass types
- PtL assumes only Direct Air Capture (no concentrated CO2 sources)
- No regional variation in resources
- Simplified competition between pathways

## Data Sources

Production pathway data from:
- [@jing_understanding_2022] - Fossil kerosene
- [@NEULING201854] - Biofuels
- [@wallington_green_2024] - Hydrogen and electrofuels
- [@ar6_database] - Grid electricity carbon intensity

## Related Documentation

For energy mix coupling diagrams, see:
[User Guide - Energy Mix](../user_guide/index.md#models)
