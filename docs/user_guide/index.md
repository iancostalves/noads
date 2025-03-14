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

``` mermaid
classDiagram
    class AutoModel {
      discipline : AutoJAXDiscipline
    }
    class JAXModel {
    }
    class Model {
      discipline : JAXDiscipline
    }
    AutoModel --|> Model
    JAXModel --|> Model
```

## Scenarios

Scenarios are made from a collection of models. [TemporalScenario][noads.core.scenarios.temporalscenario] is the direct
application time-dependent models, where a selective vectorization is applied depending on their inputs (for example,
only models that use time-dependent inputs are vectorized, while others are not).
[MultiScenario][noads.core.scenarios.multiscenario] is a further vectorization of
[TemporalScenario][noads.core.scenarios.temporalscenario]'s to evaluate multiple input values simultaneously (some
inputs can be kept fixed, while others are modified) and perform some operations across these multiple instances (a mean
of outputs for example).

``` mermaid

classDiagram
    class AdjointMethod {
      name
    }
    class MultiScenario {
      fixed_inputs : list[str]
      mean_outputs : list[str]
      scenario_inputs : list[str]
      scenario_names : list[str]
      scenario_outputs : list[str]
      temporal_scenario
    }
    class TemporalScenario {
      constant_inputs : list[str]
      constrained_control_groups : Mapping[str:Sequence[str]]
      control_delay_times : Mapping[str, float]
      cubic_interpolation : bool
      custom_controls : Sequence[str]
      interpolated_inputs : list[str]
      interpolation_prefix
      models : list[Model]
      non_modified_inputs : list[str]
      time_integrated_outputs : list[str]
      time_interpolation
      time_vector
      unvectorized_chain : JAXChain
      vectorized_chain : JAXChain
    }
    TemporalScenario --* MultiScenario : temporal_scenario

```

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


``` mermaid

classDiagram
    class GAM {
    }
    class Fleet {
      consumed_carriers : list[EnergyCarrier]
      models : list[Model]
      name : str
      operating_aircraft : list[AircraftOperation | AircraftDesign]
      ask_model()
      consumption_model()
      demand_avoidance_model()
      last_share_model()
      mean_consumption_impacts_model()
    }
    class FleetAssembly {
      design_aircraft : bool
      fleets : list[Fleet]
      models
      ask_model()
      consumption_model()
      demand_avoidance_model()
      last_share_model()
      mean_consumption_impacts_model()
    }
    class AircraftDesign {
      mission : Mapping[str, str | float]
      models
      power_system : Mapping[str, str | float]
      reference_aircraft
      technology_evolution : list[AircraftTechParameter]
      consumption_model()
      control_model()
      design_model()
    }
    class AircraftOperation {
      energy_per_ask : float
      lifetime : float
      models : list[Model]
      name : str
      propulsion
      recent : bool
      consumption_model()
      share_model()
    }
    class AircraftTechParameter {
      name : str
      value_2020 : float
      value_2040 : float
      value_2060 : float
      value_at_entry_into_service(entry_into_service)
    }
    class Energy {
      unit : str
    }
    class EnergyCarrier {
      specific_energy : float
      unit : str
      energy_to_mass(energy)
      energy_to_volume(energy)
      mass_to_energy(mass)
      volume_to_energy(volume)
    }
    class EnergyMix {
      constrained_inputs : list[Stream]
      final_energies : list[ProducedEnergyCarrier]
      impacts : list[Impact]
      input_streams : list[Stream]
      models
      produced_energies : set[ProducedEnergy]
      secondary_energies : list[ProducedEnergy]
      secondary_energy : list
      constraint_model()
      input_streams_model()
      plot_pretty_couplings()
      total_impacts_model()
    }
    class Impact {
      budget : float
    }
    class InterpolatedUnivariateSpline {
      k : int
      antiderivative(xs)
      derivative(x, n)
      tree_flatten()
      tree_unflatten(aux_data, children)
    }
    class MaterialStream {
      density : float
      unit : str
      mass_to_volume(mass)
      volume_to_mass(volume)
    }
    class ProducedEnergy {
      impacts : list[Impact]
      input_streams : list[Stream]
      models
      output_streams : list[Stream]
      pathways : list[ProductionPathway]
      add_output_stream(output_stream: Stream)
      consumption_model()
      impact_index_model()
      production_model()
      set_output_streams(output_streams: Sequence[Stream])
    }
    class ProducedEnergyCarrier {
      specific_energy
      unit : str
      consumption_model()
    }
    class ProductionPathway {
      impacts : list[Impact]
      input_streams : list[Stream]
      models
      name : str
      consumption_model()
      impact_index_model()
    }
    class PropulsionSystem {
      energy_carrier_mix : Mapping[EnergyCarrier:float]
      name : str
    }
    class Stream {
      name : str
      unit : str
    }
    AircraftOperation --* AircraftDesign : reference_aircraft
    PropulsionSystem --* AircraftOperation : propulsion
    PropulsionSystem --o EnergyCarrier : energy_carrier_mix
    ProductionPathway --o Stream : input_streams
    ProductionPathway --o Impact : impacts
    ProducedEnergy --o ProductionPathway : pathways
    Fleet --o AircraftOperation : operating_aircraft
    AircraftDesign --o AircraftTechParameter : technology_evolution
    AircraftDesign --o GAM
    AircraftTechParameter --o InterpolatedUnivariateSpline
    EnergyMix --o ProducedEnergyCarrier : final_energies
    EnergyMix --o ProducedEnergy : secondary_energies
    EnergyMix --o Impact : impacts
    EnergyMix --o Stream : input_streams
    Energy --|> Stream
    EnergyCarrier --|> MaterialStream
    EnergyCarrier --|> Energy
    ProducedEnergy --|> Energy
    ProducedEnergyCarrier --|> EnergyCarrier
    ProducedEnergyCarrier --|> ProducedEnergy
    Impact --|> Stream
    MaterialStream --|> Stream
    AircraftDesign --|> AircraftOperation
    FleetAssembly --|> Fleet

```
