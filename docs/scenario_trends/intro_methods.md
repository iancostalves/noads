# Introduction and Methods

This page provides the context, methodology, and numerical methods used in the optimization of aviation decarbonization scenarios.

## Context

Aviation is often considered a hard-to-abate sector because carbon-free commercial aircraft are not readily available, and the production of alternative energy carriers on a global scale requires significant amounts of biomass and low-carbon electricity [@becken_implications_2023]. However, the sector will play a critical role in long-term climate mitigation of transportation [@girod_global_2012].

### Energy Carrier Options

Several key energy carriers have emerged as potential substitutes for conventional kerosene:
- Synthetic kerosene (biomass-to-liquid and power-to-liquid pathways)
- Liquid hydrogen
- Ammonia
- Liquid natural gas
- Ethanol and methanol
- Batteries

Among these, only synthetic kerosene can be used in today's fleet without aircraft redesign. Current engines are limited by certification to 50% mixing with conventional kerosene, though advances toward 100% Sustainable Aviation Fuels (SAF) are ongoing.

### Aircraft Design Challenges

Compared to ground and marine transportation, aircraft are more weight-sensitive because they must generate lift to remain airborne. Increasing mass leads to:
- Increased induced drag
- Higher structural mass to support extra loads
- Further drag increase (snowball effect)

This means estimating energy consumption is a fundamentally coupled problem. Alternative aircraft designs may display:
- Higher energy consumption (batteries due to low specific energy)
- Limited payload and range (liquid hydrogen due to cryogenic tank weight)

## Research Positioning

Global Integrated Assessment Models (IAMs) lack technology and fleet detail concerning aviation emissions, while industry roadmaps rely on strong assumptions of abundant renewable energy. This work bridges these approaches using **optimization to endogenously choose** appropriate timing and market penetration of conventional and alternative aircraft concepts.

### Key Contributions

1. **Link sectoral mitigation with background SSP scenarios**
2. **Incorporate fleet and technology detail** with market segmentation
3. **Apply efficient numerical methods** for high-dimensional nonlinear optimization
4. **Demonstrate scenario-robust policy optimization** across multiple futures

## Methodology Overview

The optimization approach follows a structured process:

**Conceptual Data Flow**: For visualization of model couplings and data flow, see the [OOP Architecture diagrams](../user_guide_extended/oop_architecture.md) which include:
- General system overview
- Fleet assembly structure  
- All model couplings
- Energy mix interactions

### Before Optimization Loop

The chosen global scenario determines evolution of:
- Socioeconomic drivers (population and economy)
- Energy system (global production of biomass and electricity)
- Grid electricity emission factor

### During Optimization Loop

Iteratively, a set of policy variables is chosen:
- When and how much to deploy new aircraft
- How much to produce of each energy pathway
- How much trend traffic to avoid (in low-demand formulation)

The simulation models are evaluated to estimate:
- Objective function value
- Constraint satisfaction

### Model Disciplines

The main model disciplines and their interactions:

1. **Policy Controls**: Time-dependent variables with realistic delays
2. **Demand**: Traffic evolution from population and income
3. **Current Fleet**: Existing aircraft with calibrated performance
4. **New Aircraft**: Conceptual design with GAM (Generic Airplane Model)
5. **Fleet Replacement**: Gradual turnover based on lifetimes
6. **Energy Mix**: Production pathways with lifecycle emissions
7. **Constraints**: Resource limits and feasibility requirements
8. **Objectives**: Cumulative emissions or demand management burden

For detailed model assumptions and equations, see the [Model Documentation](../model_documentation/index.md) section.

## Numerical Methods

The numerical methods employed were crucial to enable fast multi-scenario analysis:

- **Automatic model coupling**: Dependencies resolved automatically
- **Gradient-based optimization**: Efficient for high-dimensional problems
- **Automatic differentiation**: Via JAX for gradient computation
- **Vectorization and JIT compilation**: For computational efficiency

### Computational Gains

Comparison of execution times:

| Method | Computation (s) | Linearization (s) | Speedup |
|--------|----------------|-------------------|---------|
| Standard (FD) | 372.7 | 362.6 | 1x |
| JAX version | 0.45 | 0.48 | ~830x / ~750x |

**Full optimization speedup**: From 1.5 hours to ~1 minute on a conventional laptop

The use of differential programming (JAX) reduces:
- **Implementation burden**: Automatic differentiation eliminates manual derivative coding
- **Execution time**: JIT compilation and vectorization provide orders of magnitude speedup
- **Memory requirements**: Efficient gradient computation via automatic differentiation

### Software Stack

- **GEMSEO**: Multidisciplinary optimization framework [@gallard_gems_2018]
- **JAX**: Automatic differentiation and JIT compilation [@jax2018github]
- **GEMSEO-JAX**: Plugin bridging JAX into GEMSEO processes [@gemseo_jax]
- **Generic Airplane Model (GAM)**: Aircraft design tool [@kambiri_energy_2024]

## Background Scenarios

### Shared Socioeconomic Pathways (SSP)

The SSPs were introduced between IPCC AR5 and AR6 to bridge socioeconomic dimensions into climate assessments [@riahi_shared_2017]:

**SSP1 - Sustainability (Taking the Green Road)**:
- Low challenges to mitigation and adaptation
- Gradual shift toward sustainable path
- Inclusive development respecting environmental boundaries
- Consumption oriented toward low material growth

**SSP2 - Middle of the Road**:
- Medium challenges to mitigation and adaptation
- Trends don't shift markedly from historical patterns
- Slow progress toward sustainable development goals

**SSP5 - Fossil-fueled Development (Taking the Highway)**:
- High challenges to mitigation, low for adaptation
- Faith in markets, innovation, and human capital
- Exploitation of abundant fossil fuels
- Resource and energy-intensive lifestyles

### Mitigation Scenarios

From each baseline SSP, mitigation scenarios add climate policies to match Representative Concentration Pathways (RCP):
- RCP 1.9: ~1.5°C warming
- RCP 2.6: ~2°C warming
- RCP 4.5: ~3°C warming

Used in this work:
- **SSP1-1.9**: Ambitious mitigation with sustainable development
- **SSP2-2.6**: Moderate mitigation, middle-of-the-road development
- **SSP2-1.9**: Moderate development with ambitious mitigation
- **SSP2-3.4**: Moderate development with limited mitigation
- **SSP5-4.5**: Fossil-intensive development with moderate mitigation

### AR6 Scenario Database

Background data from AR6 Scenario Database [@ar6_database]:
- Population and GDP projections
- Energy system evolution (biomass, electricity production)
- Grid carbon intensity pathways
- Multi-model ensemble for validation

## Optimization Formulations

### Trend Formulation

**Objective**: Minimize cumulative CO2 emissions (2020-2070)

**Variables**:
- Entry-into-service year and maximum share per aircraft per market
- Energy production pathway shares
- (New aircraft performance is designed, not optimized)

**Constraints**:
- Aircraft design feasibility
- Fleet retirement non-negativity
- Pathway shares sum to 1
- Biomass and electricity consumption ≤ fair share of global production

### Low-Demand Formulation

**Objective**: Minimize present value of demand reduction burden

**Additional Variables**:
- Supply reduction ratio per market (demand caps)

**Additional Constraints**:
- Cumulative CO2 emissions ≤ carbon budget (Paris-aligned)

**Burden Calculation**: Simplified as relative price increase from constant price elasticity

### Scenario-Robust Formulation

**Objective**: Minimize mean cumulative emissions across multiple background scenarios

**Approach**:
- Single policy applied to all scenarios (SSP2-1.9, 2.6, 3.4)
- Optimizes for flexibility rather than single-scenario optimum
- Results in higher emissions in target scenario but better performance across range

## Technology Scenarios

Three technology maturation scenarios explore uncertainty in technology development:

**Lower Technology**:
- Conservative technology assumptions
- Longer aircraft lifetimes (slower renewal)
- Less efficient new aircraft
- Favors earlier deployment to capture long-term benefits

**Mid Technology**:
- Moderate assumptions
- Intermediate renewal rates
- Balanced efficiency improvements

**Upper Technology**:
- Optimistic assumptions
- Shorter lifetimes (faster renewal)
- More efficient new aircraft
- Can afford to delay deployment

Applied to:
- Battery specific energy (200-1500 Wh/kg by 2060)
- Fuel cell specific power (1-6 kW/kg by 2060)
- Fuel cell efficiency (40-65% by 2060)
- LH2 tank gravimetric index (20-80% by 2060)
- Structural weight reduction (0-40% by 2060)
- Current fleet efficiency and lifetimes

## Scenario Definitions

See the [Scenario Trends overview](index.md) for the complete scenario summary table.

### Baseline Scenarios
- No mitigation policies
- Fossil kerosene only
- Optimize aircraft deployment timing to minimize emissions
- Three SSP narratives (1, 2, 5)

### Drop-in Scenarios
- Add SAF (biofuels and electrofuels)
- Constrain biomass and electricity use
- Three variants: trend, availability, low-demand

### Breakthrough Scenarios  
- Add alternative aircraft (LH2, batteries)
- Constrain biomass and electricity use
- Three variants: trend, availability, low-demand

### Robust Scenarios
- Optimize across multiple background scenarios
- Two variants: trend, low-demand
- Mid technology only

## Model Validation

Models validated through:
- **Historical calibration**: Demand model R² > 0.95 on 1980-2019 data
- **Cross-model comparison**: Agreement with AeroMAPS [@planes_aeromaps_2023]
- **AR6 ensemble comparison**: Results fall within AR6 scenario range
- **Sensitivity analysis**: Key parameters tested for robustness

## Limitations

Key simplifications:
- **Non-CO2 effects**: Contrails and NOx not included
- **Global aggregation**: Regional variations not captured
- **Fixed network**: Route structure assumed constant
- **Exogenous scenarios**: Background SSP not optimized
- **Technology uncertainty**: Fixed maturation pathways
- **Simplified economics**: Constant price elasticities

For detailed discussion of limitations, see [Model Assumptions](../model_documentation/assumptions.md).

## References

For mathematical details, see:
- [Model Equations](../model_documentation/equations.md)
- [Calibration Methodology](../model_documentation/calibration.md)

For code and data, see:
- GitHub repository: [iancostalves/noads](https://github.com/iancostalves/noads)
