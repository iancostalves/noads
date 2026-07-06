<!--
 Copyright 2025 ISAE-SUPAERO, https://www.isae-supaero.fr/en/
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Partitioning the fleet differently

The global fleet is segmented into distance bands, each treated as an independent
market in which aircraft compete for supply. The segmentation lives entirely in
`noads/application/base_objects.py` and `noads/application/scenario_setup.py`; the
core models are agnostic to the number and definition of markets.

## What defines a market segment

Four dictionaries in {mod}`noads.application.base_objects`, keyed by market name:

| Dictionary | Content | Where the values come from |
|---|---|---|
| `categories_mission` | number of seats (`npax`) and design range (`range`, m) | Top-Level Aircraft Requirements per market |
| `category_conso` | current-fleet energy consumption (MJ/seat-km) as (lower, mid, upper quartile) | AeroSCOPE 2019 flights within each distance band |
| `category_lifetime` | fleet-replacement lifetimes in years, per technology scenario | planespotters age-at-retirement statistics |
| `propulsion_mission` | cruise speed and altitude per architecture | fixed per propulsion architecture, not per market |

In addition, each market's constant **share of trend supply** is set in
`single_scenario_setup` (`"<market>.share"` entries of the `constants` dict, summing
to 1 with the `general` market taking the remainder).

## Steps to re-partition

1. **Choose the new bands** and recompute the per-band statistics. The
   [current aircraft example](../gallery/models/aircraft/plot_current_aircraft.rst)
   computes the consumption quartiles and retirement ages per band; adapt its distance
   boundaries and re-run it against the AeroSCOPE per-route dataset to obtain the new
   `category_conso` and `category_lifetime` values.
2. **Update the four dictionaries** in `base_objects.py` with the new market names,
   TLARs, consumptions, and lifetimes.
3. **Update the supply shares** (`"<market>.share"`) in `scenario_setup.py` with the
   ASK share of each new band (also obtainable from the AeroSCOPE histograms).
4. Nothing else is needed: `initialize_base_objects` builds one
   {class}`~noads.core.models.fleet.fleet.Fleet` per entry of `categories_mission`,
   with one reference (current) aircraft and one
   {class}`~noads.core.models.fleet.aircraft_design.AircraftDesign` per compatible
   propulsion architecture, and `single_scenario_setup` creates the corresponding
   optimization variables and per-market retirement constraints automatically.

:::{warning}
The finer the partition, the more optimization variables (each market adds an
entry-into-service and a maximum market share per aircraft concept). The gradient
computation scales well thanks to forward-mode automatic differentiation, but SLSQP
convergence may need more iterations; check the optimization history when increasing
the number of markets.
:::
