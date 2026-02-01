# Formulation, Numerical Methods, and Computational Gains

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

## Numerical Methods

The numerical methods employed were crucial to enable fast multi-scenario analysis:

- **Automatic model coupling**: Dependencies resolved automatically
- **Gradient-based optimization**: Efficient for high-dimensional problems
- **Automatic differentiation**: Via JAX for gradient computation
- **Vectorization and JIT compilation**: For computational efficiency

### Software Stack

- **GEMSEO**: Multidisciplinary optimization framework [@gallard_gems_2018]
- **JAX**: Automatic differentiation and JIT compilation [@jax2018github]
- **GEMSEO-JAX**: Plugin bridging JAX into GEMSEO processes [@gemseo_jax]
- **Generic Airplane Model (GAM)**: Aircraft design tool [@kambiri_energy_2024]

## Computational Gains

### Benchmark Results

Comparison of execution times over single scenario computation and linearization (mean of 50 repetitions on Intel Core i7-10850H at 2.70 GHz):

| Method | Computation (s) | Linearization (s) | Speedup |
|--------|----------------|-------------------|---------|
| Standard (FD) | 372.7 | 362.6 | 1x |
| JAX version | 0.45 | 0.48 | ~830x / ~750x |

**Full optimization speedup**: From 1.5 hours to ~1 minute on a conventional laptop

### Benefits of Differential Programming

The use of JAX reduces:
- **Implementation burden**: Automatic differentiation eliminates manual derivative coding
- **Execution time**: JIT compilation and vectorization provide orders of magnitude speedup
- **Memory requirements**: Efficient gradient computation via automatic differentiation

### Algorithm Comparison

Gradient-based algorithms:
- Require far fewer iterations for convergence
- Achieve higher objective precision
- Confirm advantages for high-dimensional problems

Using an uncompiled model with standard Finite Differences (FD) was prohibitive due to memory demands. JIT compilation makes FD feasible yet still less efficient—both in iteration count and runtime—than forward-mode automatic differentiation. Overall, using JAX yields speedups of up to two orders of magnitude.

### Compilation Overhead

JAX introduces an initial overhead of roughly 38 seconds due to compilation:
- This is excluded from single computation comparisons
- But included in full-optimization timings
- Once compiled, JAX achieves speedups of up to three orders of magnitude

### Comparison with AeroMAPS

For context, a single scenario simulation using a restricted subset of AeroMAPS models [@planes_aeromaps_2023] (default bottom-up setup without offsetting, non-CO₂, or cost models) takes 1.29 seconds, about three times slower than the present approach. This comparison should be interpreted cautiously, as the modeling scopes differ substantially:
- **AeroMAPS includes**: freight, non-CO₂ effects, and cost models
- **AeroMAPS omits**: aircraft design routines and links between traffic growth and socioeconomic drivers

## Scenario Overview

Several policy scenarios are simulated using the numerical optimization methods presented:

### Baseline Scenarios (SSP1, 2, and 5)
- Two new generations of conventional aircraft
- Entry-into-service and deployment optimized to minimize cumulative emissions
- Only fossil kerosene consumed
- Three technology scenarios affecting energy consumption and fleet replacement lifetimes

### Drop-in Mitigation Scenarios (SSP2 baseline)
- **Trend**: Biofuel and electrofuel (SAF) in Jet-A blend, 5.0% of global electricity/biomass
- **Availability**: Same but 8.6% of global electricity/biomass
- **Low-demand**: Trend consumption with traffic avoidance to meet emission constraints
- Each explored with 3 technology scenarios

### Breakthrough Mitigation Scenarios (SSP2 baseline)
- Same as Drop-in plus alternative aircraft (Battery-Electric, LH2 Fuel-Cell, LH2 Gas Turbine)
- **Trend**: 5.0% of global production
- **Availability**: 8.6% of global production
- **Low-demand**: 5.0% with traffic avoidance
- Each explored with 3 technology scenarios

### Scenario-Robust Mitigation
- Optimizes mean emissions across SSP2-1.9, 2.6, and 3.4
- **Trend** and **Low-demand** variants
- Mid technology only

| Scenario | Background | Demand | Traffic Aversion | Objective | Energy Carriers | % Production | Tech |
|----------|-----------|--------|------------------|-----------|-----------------|--------------|------|
| Baseline SSP1 | SSP1-1.9 | Trend -10% | - | Min CO2 | Jet-A (fossil) | - | L,M,U |
| Baseline SSP2 | SSP2-2.6 | Trend | - | Min CO2 | Jet-A (fossil) | - | L,M,U |
| Baseline SSP5 | SSP5-4.5 | Trend +50% | - | Min CO2 | Jet-A (fossil) | - | L,M,U |
| Drop-in trend | SSP2-2.6 | Trend | - | Min CO2 | Jet-A (fossil+SAF) | 5.0 | L,M,U |
| Drop-in availability | SSP2-2.6 | Trend | - | Min CO2 | Jet-A (fossil+SAF) | 8.6 | L,M,U |
| Drop-in low-demand | SSP2-2.6 | Trend | ✓ | Min price increase | Jet-A (fossil+SAF) | 5.0 | L,M,U |
| Breakthrough trend | SSP2-2.6 | Trend | - | Min CO2 | Jet-A, LH2, Battery | 5.0 | L,M,U |
| Breakthrough availability | SSP2-2.6 | Trend | - | Min CO2 | Jet-A, LH2, Battery | 8.6 | L,M,U |
| Breakthrough low-demand | SSP2-2.6 | Trend | ✓ | Min price increase | Jet-A, LH2, Battery | 5.0 | L,M,U |
| Robust trend | SSP2 (1.9,2.6,3.4) | Trend | - | Min mean CO2 | Jet-A, LH2, Battery | 5.0 | M |
| Robust low-demand | SSP2 (1.9,2.6,3.4) | Trend | ✓ | Min mean price | Jet-A, LH2, Battery | 5.0 | M |

**Note**: L,M,U = Lower, Mid, Upper technology scenarios
