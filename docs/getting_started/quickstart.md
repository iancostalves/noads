<!--
 Copyright 2025 ISAE-SUPAERO, https://www.isae-supaero.fr/en/
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# NOADS in 10 minutes

This guide walks you through a complete optimization — from setting up an aviation
decarbonization scenario to interpreting the results.

:::{admonition} Prerequisites
:class: tip
Make sure you have [installed NOADS](installation.md) and are familiar with the
[core concepts](concepts.md).
:::
## Step 1: Run a single-policy optimization

The fastest way to get results is through the high-level API. This sets up a complete
scenario (fleet, energy system, traffic demand, constraints) and runs the optimizer:

```python
from noads.application.examples import single_policy_scenario_optimization

output = single_policy_scenario_optimization(
    global_scenario_name="SSP2-26",   # (1)!
    technology_index=1,                # (2)!
    carbon_budget_percent=3.0,         # (3)!
    drop_in_only=False,                # (4)!
    fossil_kerosene_only=False,        # (5)!
    plot_optimum=True,                 # (6)!
    save_optimum=True,                 # (7)!
)
```

1. **Climate scenario**: SSP2 pathway with 2.6 W/m² forcing (≈2°C target).
2. **Technology level**: `0` = pessimistic, `1` = mid, `2` = optimistic assumptions for
   battery density, motor power, structural efficiency, etc.
3. **Carbon budget share**: aviation gets 3% of the global remaining CO₂ budget.
4. **Drop-in only**: if `True`, restricts to kerosene-compatible fuels (no hydrogen,
   no batteries).
5. **Fossil kerosene only**: if `True`, disables all alternative fuels (baseline
   scenario).
6. **Plot**: automatically generates result plots.
7. **Save**: writes the optimal solution to disk for later reuse.

The optimizer (NLOPT SLSQP with JAX-computed gradients) typically converges in a few
minutes. You will see an optimization history and result plots showing emissions, fleet
composition, and energy mix trajectories from 2025 to 2075.

## Step 2: Understand the scenario setup

Behind the scenes, `single_policy_scenario_optimization` calls `single_scenario_setup`,
which you can also use directly for more control:

```python
from noads.application.scenario_setup import single_scenario_setup

scenario, design_space, constraints, energy_mix, fleet = single_scenario_setup(
    name="my_scenario",
    background_scenario_name="SSP2-26",
    start_year=2025,
    end_year=2075,
    technology_index=1,
)
```

This returns five objects:

| Object | Type | What it contains |
|--------|------|-----------------|
| `scenario` | `TemporalScenario` | The assembled time-dependent model (fleet + energy + traffic) |
| `design_space` | GEMSEO `DesignSpace` | Decision variables: fuel blend shares, aircraft entry dates, market shares |
| `constraints` | `dict` | Inequality constraints: carbon budget, resource limits, feasibility bounds |
| `energy_mix` | `EnergyMix` | The energy production system (pathways, carriers, impacts) |
| `fleet` | `Fleet` | The aircraft fleet (current + new aircraft, market segments) |

## Step 3: Run the optimizer with GEMSEO

With these objects, you have full control over the optimization:

```python
from gemseo import create_scenario

gemseo_scenario = create_scenario(
    disciplines=[scenario.discipline],
    formulation_name="DisciplinaryOpt",
    objective_name="cumulative.CO2",
    design_space=design_space,
)

# Add constraints
for name, (value, positive) in constraints.items():
    gemseo_scenario.add_constraint(name, "ineq", value=value, positive=positive)

# Run
gemseo_scenario.execute(
    algo_name="NLOPT_SLSQP",
    max_iter=2000,
    ftol_rel=1e-15,
    ineq_tolerance=1e-4,
)

# Retrieve optimal outputs
x_opt = gemseo_scenario.optimization_result.x_opt_as_dict
output = scenario.discipline.execute(x_opt)
```

## Step 4: Visualize results

```python
from noads.application.visualization import plot_single_scenario_result

plot_single_scenario_result(
    scenario_name="SSP2-26",
    output_optimal={**x_opt, **output},
    energy_mix=energy_mix,
    fleet=fleet,
    save_figs=True,
    directory_path="results/my_scenario",
)
```

This generates plots for:

- **Emissions**: CO₂ trajectory vs. the allocated carbon budget
- **Fleet composition**: market shares of conventional and new aircraft over time
- **Energy mix**: evolution of fuel types (fossil kerosene, biofuel, e-fuel, hydrogen)
- **Traffic**: RPK demand trajectory with demand avoidance effects

## Step 5: Compare scenarios

To explore how results change across climate futures or technology assumptions, loop over
configurations:

```python
import itertools

scenarios = ["SSP1-19", "SSP2-26", "SSP5-45"]
tech_levels = [0, 1, 2]  # lower, mid, upper

for scenario_name, tech in itertools.product(scenarios, tech_levels):
    output = single_policy_scenario_optimization(
        global_scenario_name=scenario_name,
        technology_index=tech,
        save_optimum=True,
        plot_optimum=False,  # plot later in batch
    )
```

For **robust optimization** across multiple SSP pathways simultaneously, use the
multi-scenario API:

```python
from noads.application.scenario_setup import multi_scenario_setup

multi_scenario, design_space, constraints, energy_mix, fleet = multi_scenario_setup(
    name="robust",
    background_scenario_names=["SSP1-19", "SSP2-26", "SSP5-45"],
    start_year=2025,
    end_year=2075,
    technology_index=1,
)
```

This finds a single policy that performs well across all three climate futures.

## Step 6: Load and replot saved results

Saved results can be reloaded without re-running the optimization:

```python
output = single_policy_scenario_optimization(
    global_scenario_name="SSP2-26",
    technology_index=1,
    load_optimum=True,   # load from disk
    plot_optimum=True,   # regenerate plots
)
```

---

## What's next?

- **[Extended paper](../paper/index.md)**: the companion paper with the full models,
  formulations, and results.
- **[Extending the analysis](../extending/index.md)**: add aircraft, energy pathways,
  or resources, and quantify uncertainty.
- **[Examples](../gallery/models/index.rst)**: runnable scripts covering aircraft
  design, energy pathways, demand calibration, and the
  [optimized scenario families](../gallery/optimization/index.rst).
- **[API reference](../reference/index.md)**: full class and function documentation.
