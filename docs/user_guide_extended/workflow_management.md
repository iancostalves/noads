# Workflow Management

The `noads` package provides powerful tools for managing modular and dynamic optimization workflows.

## Temporal Scenarios

`TemporalScenario` is the foundation for time-dependent model evaluation. It applies selective vectorization depending on model inputs:

- Models that use time-dependent inputs are vectorized
- Models with constant inputs are not vectorized, improving efficiency

``` mermaid
classDiagram
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
```

### Key Features

1. **Selective Vectorization**: Only vectorizes models that need it, reducing computational overhead
2. **Control Interpolation**: Smoothly interpolates control variables over time
3. **Delay Modeling**: Implements first-order delays for realistic policy response
4. **Time Integration**: Automatically integrates outputs over time when needed

## Multi-Scenario Analysis

`MultiScenario` extends temporal scenarios to evaluate multiple input values simultaneously:

``` mermaid
classDiagram
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
      models : list[Model]
      time_vector
    }
    TemporalScenario --* MultiScenario : temporal_scenario
```

### Use Cases

- **Robustness Analysis**: Evaluate policies across multiple background scenarios (SSP1, SSP2, SSP5)
- **Sensitivity Analysis**: Test how results change with different parameter values
- **Uncertainty Quantification**: Assess performance under various uncertain conditions

### Operations Across Scenarios

- **Fixed Inputs**: Some inputs remain constant across all scenarios
- **Scenario-Specific Inputs**: Other inputs vary by scenario
- **Aggregation**: Compute mean, min, max, or other statistics across scenarios

## Dynamic Workflow Assembly

Workflows are assembled dynamically based on model dependencies:

```python
# Models are automatically coupled based on their inputs/outputs
models = [
    demand_model,
    fleet_model,
    energy_model,
    emissions_model
]

scenario = TemporalScenario(
    models=models,
    time_vector=time_vector,
    # Dependencies are resolved automatically
)
```

### Automatic Coupling

The framework automatically:

1. Analyzes model input/output dependencies
2. Determines execution order
3. Handles iterative coupling when needed
4. Computes coupled derivatives for optimization

## Fleet System Architecture

The fleet system demonstrates dynamic composition:

``` mermaid
classDiagram
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
    Fleet --o AircraftOperation : operating_aircraft
    AircraftDesign --|> AircraftOperation
    FleetAssembly --|> Fleet
```

### Fleet Composition

- **Current Fleet**: Existing aircraft with historical performance data
- **New Aircraft**: Designed aircraft with evolving technology parameters
- **Market Segmentation**: Different aircraft for different distance ranges
- **Dynamic Replacement**: Models gradual fleet turnover

## Benefits of Dynamic Workflows

1. **Flexibility**: Easy to add, remove, or modify models
2. **Scalability**: Handles increasing complexity without manual rewiring
3. **Maintainability**: Dependencies are explicit and automatically managed
4. **Efficiency**: Selective vectorization and smart coupling reduce computational cost
5. **Robustness**: Automatic handling of coupled systems reduces implementation errors
