# Core Concepts

This page describes the fundamental concepts and design patterns used throughout the `noads` package.

## Model Abstraction

At the core of the package is the concept of a `Model`, which calculates a set of outputs from a set of inputs.

### Base Model Class

All models inherit from a base `Model` class that provides:

- Input/output specification
- Automatic differentiation support via JAX
- Integration with GEMSEO optimization framework

### Model Types

**JAXModel**: Direct JAX implementation with manual gradient specification

**AutoModel**: Automatic differentiation of JAX functions

The choice between them depends on:
- Complexity of the model
- Need for custom gradient implementation
- Performance requirements

## Streams and Flows

The package uses a stream-based abstraction for material and energy flows:

### Stream Types

**Stream**: Generic flow with name and unit

**Energy**: Energy flow (electricity, hydrogen, etc.)

**MaterialStream**: Material flow with density

**EnergyCarrier**: Combined energy and material (fuel, battery storage)

**Impact**: Environmental impact (CO2 emissions, land use, etc.)

### Flow Modeling

Streams connect production pathways and consumption points:

1. **Input Streams**: Materials/energy consumed in production
2. **Output Streams**: Products from production pathways
3. **Impact Streams**: Environmental effects of production/consumption

## Production Pathways

Production pathways model how energy carriers are produced:

### Pathway Structure

Each pathway specifies:
- **Input consumption**: Amount of each input per unit output
- **Direct impacts**: Emissions/impacts from the process itself
- **Indirect impacts**: Impacts from consuming inputs

### Modular Composition

Pathways can be chained:
- Electricity → Hydrogen (electrolysis)
- Hydrogen → Liquid H2 (liquefaction)
- Hydrogen + CO2 → Synthetic fuel (power-to-liquid)

The system automatically:
- Tracks indirect impacts through the chain
- Calculates total resource consumption
- Handles circular dependencies

## Technology Evolution

Aircraft technology parameters evolve over time to represent improving technology:

### AircraftTechParameter

Parameters like battery energy density, fuel cell efficiency, and structural weight change with entry-into-service year:

- **2020 baseline**: Current technology
- **2040 targets**: Near-term improvements
- **2060 targets**: Long-term aspirations

Values are interpolated smoothly between these points using splines.

### Impact on Design

Technology evolution affects:
- Aircraft energy consumption
- Maximum range and payload
- Economic viability
- Optimal entry-into-service timing

## Time-Dependent Controls

Policy variables are modeled as time-dependent controls with realistic dynamics:

### Control Types

**Direct Controls**: Explicit time series of values

**Interpolated Controls**: Values at key points, interpolated between

**Delayed Controls**: First-order delay response to setpoints

### Delay Modeling

First-order delays model realistic policy response:

```
dx/dt = (u - x) / τ
```

where:
- `u` is the control input
- `x` is the actual state
- `τ` is the time constant (delay time)

This represents:
- Fleet replacement inertia
- Technology deployment lag
- Market adaptation time

## Optimization Formulations

The package supports multiple optimization formulations:

### Trend Formulation

Minimize cumulative emissions while meeting trend demand:
- **Objective**: Minimize total CO2 from 2020-2070
- **Variables**: Aircraft deployment timing, energy production mix
- **Constraints**: Resource availability, physical feasibility

### Low-Demand Formulation

Add demand management as a variable:
- **Objective**: Minimize burden of demand reduction (price increase)
- **Variables**: Also includes demand caps per market segment
- **Constraints**: Also includes emission budget limits

### Robust Formulation

Optimize across multiple background scenarios:
- **Objective**: Minimize mean cumulative emissions across scenarios
- **Variables**: Policy that applies to all scenarios
- **Constraints**: Feasibility in all scenarios

## Coupling and Dependencies

Models are coupled through shared variables:

### Automatic Coupling

The framework:
1. Analyzes model dependencies from inputs/outputs
2. Determines execution order (topological sort)
3. Identifies feedback loops
4. Applies appropriate coupling algorithms

### Derivative Computation

For gradient-based optimization, coupled derivatives are computed:
- **Direct mode**: Forward propagation of perturbations
- **Adjoint mode**: Backward propagation of sensitivities
- **Automatic differentiation**: JAX handles the details

## Design Philosophy

The architecture follows several key principles:

1. **Composition over inheritance**: Build complex behaviors from simple components
2. **Explicit is better than implicit**: Clear dependencies and data flow
3. **Optimize late**: Keep code flexible, optimize only when needed
4. **Type safety**: Use type hints to catch errors early
5. **Testability**: Design for easy unit testing
