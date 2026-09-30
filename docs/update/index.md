<!--
 Copyright 2025 ISAE-SUPAERO, https://www.isae-supaero.fr/en/
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

(sec-update)=

# Update paper

This section documents the updates to the models of the [main paper](../paper/index.md).
The main paper and its results stay reproducible: every update is opt-in, and the
paper configuration remains the default.

:::{note}
This section is work in progress. Updated optimization results will be added once
all the model updates are in place.
:::

## What changes compared with the main paper

| Update | Status | Page |
|---|---|---|
| Aircraft design with GAM V3.0 | available | [Aircraft design](aircraft_design.md) |
| Powertrain scale effects and maximum power per propulsor | available | [Aircraft design](aircraft_design.md) |
| LH2 tank gravimetric index depending on tank size and technology | available | [Aircraft design](aircraft_design.md) |
| Geological hydrogen | planned | [Geological hydrogen](geological_hydrogen.md) |
| Updated optimization results | planned | |

## Selecting the update models

The aircraft design model is selected with the `aircraft_model` argument, `"paper"`
(default) or `"update"`, of
{func}`~noads.application.base_objects.initialize_base_objects`,
{func}`~noads.application.scenario_setup.single_scenario_setup`,
{func}`~noads.application.scenario_setup.multi_scenario_setup` and of the example
drivers of {mod}`noads.application.examples`:

```python
from noads.application.scenario_setup import single_scenario_setup

scenario, design_space, constraints, energy_mix, fleet = single_scenario_setup(
    "SSP2-26-update",
    "SSP2-26",
    technology_index=1,
    aircraft_model="update",
)
```

The optimization results of the update model are stored in an `update` subfolder of
the results directory, so that they never overwrite the pre-computed results of the
main paper.

```{toctree}
:hidden:

aircraft_design
geological_hydrogen
```
