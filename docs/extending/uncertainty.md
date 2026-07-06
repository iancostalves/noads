<!--
 Copyright 2025 ISAE-SUPAERO, https://www.isae-supaero.fr/en/
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Uncertainty quantification

A single scenario computation takes about half a second once compiled, and its
linearization about the same (see the
[computational gains](../paper/numerical_methods.md)). This makes ensemble and
sampling studies practical on a laptop, where they would be prohibitive with
finite-difference or non-compiled models. Several complementary workflows are
available, from the ones used in the paper to full sampling-based UQ.

## Discrete sweeps (as in the paper)

The paper treats deep uncertainty with discrete alternatives, re-optimizing under
each:

- **Aircraft technology**: `technology_index` (0/1/2) selects the Lower/Mid/Upper
  values of every component technology parameter, of the current-fleet efficiency
  quartiles, and of the fleet lifetimes.
- **Background scenario**: `global_scenario_name` selects the SSP-RCP trajectory for
  population, GDP, grid emission factor, and resource availability.
- **Resource allocation**: `preferential_energy` switches the fair share between
  5.0 % and 8.6 %.

The scripts of the [optimization examples](../gallery/optimization/index.rst) sweep
these switches; new sweeps are one loop over
{func}`~noads.application.examples.single_policy_scenario_optimization`.

## Robust optimization across an ensemble

{class}`~noads.core.scenarios.multiscenario.MultiScenario` vectorizes a
{class}`~noads.core.scenarios.temporalscenario.TemporalScenario` across background
scenarios with `jax.vmap`, evaluating the whole ensemble in one compiled call. The
optimization variables are split between a set shared across the ensemble (the
aircraft mix) and per-scenario sets (the energy mix), and the objective is the
ensemble mean; see the
[robustness formulation](../paper/models/optimization.md) and the
[robust policy results](../paper/results/robust.md). To make a policy robust to a
different uncertainty source, change the list of `global_scenario_names` in
{func}`~noads.application.examples.single_policy_robust_scenario_optimization`, or
adjust which variables are fixed versus scenario-specific in
`multi_scenario_setup`.

## Sampling uncertain parameters

Most numeric assumptions are inputs of the assembled GEMSEO discipline: pathway
efficiencies and emission indices, price elasticity, discount rate, and fair shares.
Uncertainty propagation is therefore a standard GEMSEO DOE over the scenario
discipline (the aircraft technology parameters are a special case, treated
[below](#exposing-aircraft-technology-parameters)). Sketch:

```python
from gemseo import create_design_space
from gemseo import create_scenario

from noads.application.scenario_setup import single_scenario_setup

scenario, _, _, _, _ = single_scenario_setup(
    name="UQ",
    background_scenario_name="SSP2-26",
    start_year=2025.0,
    end_year=2075.0,
    technology_index=1,
)

uncertain_space = create_design_space()
uncertain_space.add_variable(
    "HEFA.direct.CO2_index", lower_bound=50.0, upper_bound=75.0, value=62.75
)
uncertain_space.add_variable(
    "constant.price_elasticity", lower_bound=-0.4, upper_bound=-0.15, value=-0.256
)

doe = create_scenario(
    scenario.discipline,
    formulation_name="DisciplinaryOpt",
    objective_name="cumulative.CO2",
    design_space=uncertain_space,
    scenario_type="DOE",
)
doe.execute(algo_name="OT_MONTE_CARLO", n_samples=200)
dataset = doe.to_dataset()  # samples for statistics or surrogate fitting
```

:::{note}
Mind the input naming conventions of the assembled discipline: scalar constants are
prefixed with `constant.`, time-dependent controls with `control.`, and interpolated
inputs with `interpolate.`; inspect `scenario.discipline.input_grammar.names` for
the exact spelling.
:::

From the same dataset, GEMSEO's uncertainty package (`gemseo.uncertainty`) provides
distributions, statistics, and sensitivity analyses (Morris, Sobol'), which can rank
which assumptions drive cumulative emissions or the demand-aversion burden.

(exposing-aircraft-technology-parameters)=

## Exposing aircraft technology parameters

The aircraft technology parameters (battery specific energy, fuel-cell specific power,
LH2 tank gravimetric index, ...) are handled differently from the other assumptions.
Their (2020, 2040, 2060) anchor values are stored on the
{class}`~noads.core.models.fleet.aircraft_tech_parameter.AircraftTechParameter`
objects at construction time
(from `tech_params_lower_mid_upper_2020_2040_2060[technology_index]` in
{mod}`noads.application.base_objects`), and
{class}`~noads.core.models.fleet.aircraft_design.AircraftDesign` closes over them: the
design model reads only the aircraft's entry-into-service year as a discipline input
and interpolates the anchor values internally. They are therefore **not** inputs of
the assembled discipline and cannot be sampled the way the sketch above samples
`HEFA.direct.CO2_index`. There are two ways to bring them into a UQ study.

**Outer rebuild loop (no code change).** Treat the technology assumptions as an
outer sampling loop: for each sample, patch the anchor values (or interpolate a
continuous "technology index" between the Lower/Mid/Upper triplets) in
`tech_params_lower_mid_upper_2020_2040_2060`, rebuild the scenario with
{func}`~noads.application.examples.single_policy_scenario_optimization` or
{func}`~noads.application.scenario_setup.single_scenario_setup`, and record the
outcome. This is the discrete technology sweep of the paper generalized to a
continuous sample; it is simple but pays a JIT recompilation per distinct structure,
so it suits a few hundred samples rather than large ensembles.

**Refactor into discipline inputs (one compile).** To sample them as cheaply as the
other parameters, expose the anchor values as named inputs of the design model.
In {class}`~noads.core.models.fleet.aircraft_design.AircraftDesign`, add
`{aircraft}.{param}.value_2020/2040/2060` to the design model's `default_inputs` and
`input_names`, and have `_gam_design` build each
{class}`~noads.core.models.fleet.aircraft_tech_parameter.AircraftTechParameter` (or
call its spline) from those inputs instead of from the closed-over attributes. Once
they flow through the assembled discipline they behave like any other input: they can
be added to the `uncertain_space` above, sampled in a single compiled `vmap` DOE, and
differentiated in forward mode for the local sensitivities below. This is the same
refactor that would let the optimizer trade technology maturity against
entry-into-service, rather than fixing it per scenario.

## Local sensitivities for free

Because objective and constraints are differentiated with forward-mode automatic
differentiation, the gradient of any output with respect to any input is available at
roughly the cost of one extra simulation
(`scenario.discipline.linearize(...)`). Around an optimized policy, these gradients
are first-order sensitivities of the outcome to every assumption, a cheap screening
before committing to a sampling study.
