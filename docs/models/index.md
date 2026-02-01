# Model Documentation

This section provides detailed documentation for each model discipline used in the aviation decarbonization scenarios.

## Model Overview

The models are organized into several disciplines:

1. **[Demand Model](demand.md)** - Air traffic demand evolution based on socioeconomic drivers
2. **[Aircraft Models](aircraft.md)** - Current fleet and prospective aircraft design
3. **[Fleet Model](fleet.md)** - Fleet composition and replacement dynamics
4. **[Energy Mix Model](energy.md)** - Energy production pathways and lifecycle emissions

## General Model Assumptions

For comprehensive assumptions and limitations across all models, see the model assumptions table:

| Model Discipline | Assumptions | Limitations |
|-----------------|-------------|-------------|
| **Policy controls** | Time-delayed share-based control | Instability under rapid volume changes |
| **Air traffic demand** | Logistic demand functions calibrated on historical data | Extrapolation outside calibrated range |
| **Current aircraft** | Fixed composition from historical data | Potential mismatch between efficiency and lifetimes |
| **New aircraft** | GAM conceptual design with 4 architectures | TLAR fixed within markets |
| **Fleet replacement** | Gradual replacement with market-specific lifetimes | Airlines may operate at non-optimal distances |
| **Energy mix** | Multiple carriers and pathways with WTW emissions | No differentiation between biomass types |
| **Constraints** | Feasibility and fair share constraints | Grandfathering approach for fair shares |
| **Objectives** | Minimize emissions or demand burden | Simplified price elasticity modeling |

## Data Sources

Models are calibrated using data from:
- **ICAO**: Historical RPK data
- **World Bank**: Socioeconomic indicators
- **AeroSCOPE**: Flight-level performance [@salgas_aeroscope]
- **Planespotters**: Aircraft lifetimes [@planespotters]
- **AR6 Database**: Background scenarios [@ar6_database]
- **Literature**: Technology parameters and production pathways

## Interactive Examples

Interactive examples demonstrating each model are available in the [Examples](../../examples/) section:
- [Demand calibration and scenarios](../../examples/models/demand/)
- [Aircraft performance and technology](../../examples/models/aircraft/)
