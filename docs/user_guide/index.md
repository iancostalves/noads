<!--
 Copyright 2025 ISAE-SUPAERO, https://www.isae-supaero.fr/en/
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# User guide
This repository is intended as a support to the reproduction of results from the paper "Numerical Optimization of
Aviation Decarbonization Scenarios", it is not actively maintained, but many of the models and functionalities
demonstrated here will be incorporated into [AeroMAPS](https://aeromaps.isae-supaero.fr/).

## Core

The repository is centered around the notion of a [Model][noads.core.model] which calculates a set of outputs from a set
of inputs, based on the [gemseo-jax](https://gitlab.com/gemseo/dev/gemseo-jax) plug-in.

[Core UML diagram](core.html)

## Scenarios

Scenarios are made from a collection of models. [TemporalScenario][noads.core.scenarios.temporalscenario] is the direct
application time-dependent models, where a selective vectorization is applied depending on their inputs (for example,
only models that use time-dependent inputs are vectorized, while others are not).
[MultiScenario][noads.core.scenarios.multiscenario] is a further vectorization of
[TemporalScenario][noads.core.scenarios.temporalscenario]'s to evaluate multiple input values simultaneously (some
inputs can be kept fixed, while others are modified) and perform some operations across these multiple instances (a mean
of outputs for example).

[Scenarios UML diagram](scenarios.html)

## Models

Finally, some more classes are made to increase modularity and flexibility with respect to the Energy production and
Fleet operation.

We start with the notion of a [Stream][noads.core.models.energy.streams], a generic flow of material, energy, or impact.
A [ProductionPathway][noads.core.models.energy.production_pathway] consumes input streams to produce a given type of
energy, and can have direct impacts (made in the production process) and indirect impacts (due to the consumption of
inputs, each of them with a given impact index). As some energy types are consumed in the production of others, the
[EnergyMix][noads.core.models.energy.energy_mix] has to assemble all energy types and production pathways to estimate
total impacts and total input consumption in order to produce the final energies consumed by a fleet.

An [EnergyCarrier][noads.core.models.energy.energy] can be seen as a material and energy flow, and is used in an
[AircraftOperation][noads.core.models.fleet.aircraft_operation], through its propulsion system with a given energy
consumption. [AircraftDesign][noads.core.models.fleet.aircraft_design] is responsible for designing a new aircraft and
estimating its energy consumption using the [Generic Airplane Model](https://gitlab.com/m6029/genericairplanemodel),
subject to a set of [AircraftTechParameter][noads.core.models.fleet.aircraft_tech_parameter]'s that evolve in time.
[Fleet][noads.core.models.fleet.fleet] assembles the operation of several aircraft in a same market category, which can
be further aggregated to make up a collection of markets.

[Models UML diagram](models.html)
