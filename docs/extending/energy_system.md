<!--
 Copyright 2025 ISAE-SUPAERO, https://www.isae-supaero.fr/en/
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Energy pathways and resources

The energy production system is a graph of
{class}`~noads.core.models.energy.energy.Energy` nodes (primary resources),
{class}`~noads.core.models.energy.energy.ProducedEnergy` nodes (intermediate
energies), {class}`~noads.core.models.energy.energy.ProducedEnergyCarrier` nodes
(final carriers embarked in aircraft), and
{class}`~noads.core.models.energy.production_pathway.ProductionPathway` edges. The
graph is declared in `initialize_base_objects`
({mod}`noads.application.base_objects`), and the numeric coefficients are scenario
inputs set in `single_scenario_setup`
({mod}`noads.application.scenario_setup`). The
{class}`~noads.core.models.energy.energy_mix.EnergyMix` assembles the graph into
models for intensive impacts (primary-to-final), extensive production and consumption
(final-to-primary), and resource constraints; see the
[energy-mix model](../paper/models/energy.md) of the paper for the formulation.

## Adding a production pathway

To add a pathway to an existing energy, declare it and list it among the pathways of
its output energy. For instance, an additional biofuel route:

```python
bio_new = ProductionPathway(
    "NewRoute",
    impacts=[co2],            # impacts generated directly at production
    input_streams=[biomass],  # what the process consumes
)
biofuel = ProducedEnergy("BIOFUEL", pathways=[bio_ft, bio_atj, bio_hefa, bio_new])
```

Then provide its coefficients in the `constants` dictionary of
`single_scenario_setup` (or in `interpolated_2025_2035_2050` for values that mature
in time, as (2025, 2035, 2050) triplets):

```python
constants.update({
    "NewRoute.direct.CO2_index": 40.0,   # g CO2 per MJ produced
    "NewRoute.BIOMASS.efficiency": 0.4,  # MJ produced per MJ of biomass
})
```

Every produced energy with more than one pathway automatically gets time-dependent
**pathway share controls** as optimization variables (one per pathway except the
last, which takes the remainder), together with the constraint that shares stay in
[0, 1]. Nothing else is required: the optimizer decides how much of the new route to
use, and the [energy sankey diagrams](../paper/results/index.md) include it
automatically.

## Adding a primary resource: geological hydrogen

A new primary resource is an {class}`~noads.core.models.energy.energy.Energy` node
plus a pathway feeding an existing energy. Natural (geological) hydrogen, for
example, would compete with electrolysis and gas reforming to supply gaseous
hydrogen:

```python
geo_h2 = Energy("GEO-H2")

geological = ProductionPathway(
    "Geological_extraction",
    impacts=[co2],
    input_streams=[geo_h2],
)
gh2 = ProducedEnergy("GAS-H2", pathways=[electrolysis, gas, geological])
```

with its coefficients:

```python
constants.update({
    "GEO-H2.CO2_index": 0.0,
    "Geological_extraction.direct.CO2_index": 5.0,   # extraction and purification
    "Geological_extraction.GEO-H2.efficiency": 0.9,
})
```

### Limiting its availability

Biomass and electricity consumption are limited to a fair share of a global
production trajectory. The same mechanism applies to any primary resource: list it in
the energy-mix constructor,

```python
energy_mix = EnergyMix(energies, inputs_to_constrain=[electricity, biomass, geo_h2])
```

and provide the two inputs the constraint model expects: a scalar
`"GEO-H2.fair_share"` (in the `constants` dictionary) and a time series
`"GEO-H2.global_production"` (in J/year, added to the scenario inputs). For biomass
and electricity these trajectories come from the AR6 scenario database
({mod}`noads.application.background_scenario_data`, keys
`"<RESOURCE>.global_production"`); for a resource absent from AR6, interpolate your
own trajectory onto `temporal_scenario.time_vector` the same way
`single_scenario_setup` does for the AR6 series.

:::{note}
Estimates of extractable geological hydrogen are still highly uncertain; treating
the global production trajectory as an uncertain input is a natural use case for the
[uncertainty quantification](uncertainty.md) workflows.
:::
