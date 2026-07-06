<!--
 Copyright 2025 ISAE-SUPAERO, https://www.isae-supaero.fr/en/
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Extending the analysis

The scenarios of the [extended paper](../paper/index.md) are one instantiation of the
framework. Everything that defines them, the market segmentation, the aircraft
concepts, the energy production system, and the background scenario ensemble, is
declared in the application layer
({mod}`noads.application.base_objects` and {mod}`noads.application.scenario_setup`)
and can be changed without touching the core models. This section shows where to
plug in new analyses:

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`rows` Partitioning the fleet
:link: fleet_partitioning
:link-type: doc

Change the market segmentation: distance bands, seats, ranges, shares of supply, and
fleet lifetimes.
:::

:::{grid-item-card} {octicon}`paper-airplane` Adding an aircraft concept
:link: new_aircraft
:link-type: doc

Introduce a new propulsion architecture, such as a liquid-methane gas turbine, and
let the optimizer decide its deployment.
:::

:::{grid-item-card} {octicon}`flame` Energy pathways and resources
:link: energy_system
:link-type: doc

Add production pathways or primary energy sources, such as geological hydrogen, to
the energy mix.
:::

:::{grid-item-card} {octicon}`graph` Uncertainty quantification
:link: uncertainty
:link-type: doc

Sweep technology scenarios, sample uncertain parameters, and exploit the JAX speedups
for ensemble studies.
:::
::::

```{toctree}
:hidden:

fleet_partitioning
new_aircraft
energy_system
uncertainty
```
