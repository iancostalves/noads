<!--
 Copyright 2025 ISAE-SUPAERO, https://www.isae-supaero.fr/en/
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Adding an aircraft concept

A new propulsion architecture requires three declarations in
`noads/application/base_objects.py`, plus its energy supply chain (see
[energy pathways](energy_system.md)) if it embarks a carrier that does not exist yet.
The liquid-methane gas turbine below is used as the running example; the hooks for it
are already present as commented-out code in the repository.

## 1. Declare the architecture

Add an entry to `propulsion_architectures` (the power system passed to the Generic
Airplane Model) and to `propulsion_mission` (the cruise conditions):

```python
propulsion_architectures = {
    # ...
    "lCH4-GasTurbine": {
        "engine_count": 2,
        "engine_type": "turbofan",
        "thruster_type": "fan",
        "energy_type": "liquid_ch4",
        "bpr": 12.0,
    },
}

propulsion_mission = {
    # ...
    "lCH4-GasTurbine": {"speed": 0.8 * 340, "altitude": 28000 * 0.3048},
}
```

The `energy_type` must be one supported by the Generic Airplane Model
({class}`~noads.gam_jax.models.generic_airplane_model.GAM`), which handles the
propulsion-system weight (tanks, fuel cells, motors, batteries) as a function of the
[aircraft technology parameters](../paper/models/fleet.md).

## 2. Declare the energy carrier and its propulsion system

Inside `initialize_base_objects`, create the final energy carrier with its physical
properties and production pathways, and map the architecture to a
{class}`~noads.core.models.fleet.aircraft_operation.PropulsionSystem` consuming it:

```python
# Gas methane, produced by three competing pathways
fossil_ch4 = ProductionPathway("GM_fossil", impacts=[], input_streams=[natural_gas])
ptg = ProductionPathway("GM_methanation", impacts=[], input_streams=[gh2])
biogas = ProductionPathway("GM_biogas", impacts=[co2], input_streams=[biomass])
gch4 = ProducedEnergy("GAS-CH4", pathways=[ptg, biogas, fossil_ch4])

# Liquefaction into the final carrier embarked by the aircraft
gch4_liquefaction = ProductionPathway(
    "LM_liquefaction", impacts=[], input_streams=[electricity, gch4]
)
lch4 = ProducedEnergyCarrier(
    "LIQUID-CH4",
    pathways=[gch4_liquefaction],
    density=22.2 / 53.6,      # kg/L over MJ/kg -> used for tank sizing
    specific_energy=53.6,     # MJ/kg
)

energies.extend([gch4, lch4])
prop_systems.update({
    "lCH4-GasTurbine": PropulsionSystem("lch4_burn", {lch4: 1.0}),
})
```

`initialize_base_objects` then automatically creates one
{class}`~noads.core.models.fleet.aircraft_design.AircraftDesign` per market for the
new architecture, together with its entry-into-service and maximum market share
optimization variables.

## 3. Provide the pathway coefficients

The numeric efficiencies and direct emission indices of the new pathways are plain
scenario inputs, set in the `constants` dictionary of
`noads/application/scenario_setup.py` with the naming convention
`"<pathway>.<input>.efficiency"` and `"<pathway>.direct.CO2_index"`:

```python
constants.update({
    "NATURAL_GAS.CO2_index": 67.6,
    "GM_methanation.direct.CO2_index": 0.0,
    "GM_biogas.direct.CO2_index": 14.3,
    "GM_fossil.direct.CO2_index": 0.0,
    "LM_liquefaction.direct.CO2_index": 0.0,
    "GM_fossil.NATURAL_GAS.efficiency": 1.0,
    "GM_methanation.GAS-H2.efficiency": 0.89,
    # ...
})
```

Efficiencies maturing in time are declared in `interpolated_2025_2035_2050` instead,
as (2025, 2035, 2050) triplets that are linearly interpolated.

## 4. Run and inspect

Re-run any scenario, for example with
{func}`~noads.application.examples.single_policy_scenario_optimization` and
`load_optimum=False`. The optimizer will now trade the new architecture off against
the others in every market where its design is feasible. The fleet plots of
{mod}`noads.application.visualization` pick up new aircraft automatically from the
fleet assembly.

:::{tip}
If the new carrier competes for a constrained resource (biomass, electricity), no
extra work is needed: the resource constraints are generated from
`EnergyMix(..., inputs_to_constrain=[electricity, biomass])`. To constrain a new
primary resource, see [energy pathways and resources](energy_system.md).
:::
