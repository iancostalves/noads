# Methods Overview

An overview of the data-flow between model disciplines is shown in the conceptual diagram. The optimization approach follows a structured process.

## Optimization Process

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

## Model Disciplines

The main model disciplines and their interactions:

1. **Policy Controls**: Time-dependent variables with realistic delays
2. **Demand**: Traffic evolution from population and income
3. **Current Fleet**: Existing aircraft with calibrated performance
4. **New Aircraft**: Conceptual design with GAM (Generic Airplane Model)
5. **Fleet Replacement**: Gradual turnover based on lifetimes
6. **Energy Mix**: Production pathways with lifecycle emissions
7. **Constraints**: Resource limits and feasibility requirements
8. **Objectives**: Cumulative emissions or demand management burden

For detailed model documentation, see the [Model Documentation](../models/index.md) section.

## Detailed Methods

For detailed information on:
- Optimization formulation
- Numerical methods and automatic differentiation
- Computational gains

See: [Detailed Methods](formulation.md)
