<!--
 Copyright 2025 ISAE-SUPAERO, https://www.isae-supaero.fr/en/
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

(sec-paper-numerics)=

# Numerical methods

## Numerical Methodology

The optimization of mitigation scenarios requires libraries that allow for: assembly
and integration of numerous heterogeneous models; scaling with high-dimensional
variables, due primarily to the disagregation among periods, technologies, and
markets; minimal extra implementation when incorporating or modifying models.

Multidisciplinary Optimization (MDO) frameworks are classically used to optimize or
improve a given design under various constraints {cite:p}`mdobook`, it offers
practical methods to handle model integration and coupling. Efficient optimization
under high-dimension can also be achieved by using gradient-based algorithms, but
their use often implies an extra burden due to the manual implementation of the
derivatives of objective and constraints with regards to optimization variables.
GEMSEO is an open-source Python software to automate multidisciplinary processes
{cite:p}`gallard_gems_2018`, which provided the following features that were used in
the present study: automatic handling of coupled derivatives, automatic assembly of
complex MDO processes based on dependency graphs, interfaces to optimization
algorithms, results visualization, data storage.

Differential Programming, on the other hand, is commonly used for machine learning and
scientific computing research. This programming paradigm allows for the Automatic
Differentiation (AD) of lines of code. The use of AD for MDO can significantly reduce
implementation burden associated to efficient high-dimension optimization. JAX is a
library for array-oriented numerical computation, with capabilities for AD and
just-in-time (JIT) compilation {cite:p}`jax2018github`, enabling high-performance
scientific computing in multiple hardware configurations (CPU, GPU, and TPU).

GEMSEO-JAX {cite:p}`gemseo_jax` is an open-source plug-in that was developped by the
authors to bridge JAX programs into a GEMSEO process.

(subsec-gains)=

## Computational gains

```{list-table} Benchmark over a single scenario computation and linearization. Performed using a mean of 50 repetitions in a single Intel(R) Core(TM) i7-10850H CPU core with 2.70 GHz.
:name: tab-execution-linearization
:header-rows: 1

* - Method
  - Computation (s)
  - Linearization (s)
* - Standard (FD)
  - 372.7
  - 362.6
* - JAX version
  - 0.45
  - 0.48
* - Speedup
  - 830
  - 749
```

To evaluate the simulation gains offered by a Differential Programming paradigm, we
first compare execution times over full scenario optimizations, then over a single
computation and linearization.

```{list-table} Benchmark over a Drop-in policy optimization with trend formulation. Performed using a single Intel(R) Core(TM) i7-10850H CPU core with 2.70 GHz, as the mean of 3 repetitions, varying technology assumptions. The objective (cumulative CO2 emissions) is normalized by the 3% of the 2°C carbon budget.
:name: tab-optimization
:header-rows: 1

* - Method
  - Algorithm
  - Gradient-based
  - Iterations
  - Total time (s)
  - Speedup
  - Objective value
* - Standard (FD)
  - SLSQP
  - Yes
  - X
  - X
  - X
  - X
* - FD + JIT
  - SLSQP
  - Yes
  - 91
  - 69.3
  - 87
  - 1.72
* - JAX version
  - SLSQP
  - Yes
  - 86
  - 53.2
  - 114
  - 1.72
* - Standard
  - COBYLA
  - No
  - 1852
  - 6052
  -
  - 1.75
* - JAX version
  - COBYLA
  - No
  - 1898
  - 51
  - 119
  - 1.75
```

In {numref}`tab-optimization`, entire scenario optimization is performed with varying
optimization algorithm, differentiation strategies, and compilation. Gradient-based
algorithms not only require far fewer iterations for convergence but also achieve
higher objective precision, confirming their advantages for high-dimensional problems
{cite:p}`perez_pyopt_2012`. Using an uncompiled model with standard Finite Differences
(FD) was prohibitive due to memory demands; JIT compilation makes FD feasible yet
still less efficient—both in iteration count and runtime—than forward-mode AD.
Overall, using JAX yields speedups of up to two orders of magnitude for both
algorithms tested.

In {numref}`tab-execution-linearization` the computation compares compares how long
one single scenario computation takes to complete with (JAX version) and without
(standard) JIT compilation, and the time required to linearize objectives and
constraints. JAX introduces an initial overhead of roughly 38 s due to compilation;
this is excluded from single computation comparisons but included in full-optimization
timings. Once compiled, JAX achieves speedups of up to three orders of magnitude.

For context, a single scenario simulation using a restricted subset of AeroMAPS models
{cite:p}`planes_aeromaps_2023` (default bottom-up setup without offsetting, non-CO2,
or cost models) takes 1.29 s, about three times slower than the present approach. This
comparison should be interpreted cautiously, as the modeling scopes differ
substantially: AeroMAPS includes freight, non-CO2 effects, and cost models, but omits
aircraft design routines and links between traffic growth and socioeconomic drivers.
