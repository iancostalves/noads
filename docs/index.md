<!--
 Copyright 2025 ISAE-SUPAERO, https://www.isae-supaero.fr/en/
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# NOADS

**Numerical Optimization of Aviation Decarbonization Scenarios with GEMSEO-JAX**

---

NOADS is a Python framework for finding optimal decarbonization pathways for aviation.
It combines multidisciplinary optimization ([GEMSEO](https://gemseo.readthedocs.io/))
with automatic differentiation ([JAX](https://docs.jax.dev/)) to explore
trade-offs between fleet modernization, alternative fuels, and demand-side policies,
all within the carbon budgets defined by the IPCC's 6th Assessment Report.

NOADS serves as a **proof of concept** for advanced numerical methods (automatic
differentiation, JIT compilation, vectorized multi-scenario optimization) applied to
aviation decarbonization scenario analysis. The models build on the
[AeroMAPS](https://aeromaps.isae-supaero.fr/) platform developed at ISAE-SUPAERO,
and the numerical methods developed here are intended to be re-incorporated into
AeroMAPS. For the full scientific context, see the companion paper, available in this
documentation as an [online extended paper](paper/index.md):

> Costa-Alves I., Gourdain N., Gallard F., Gazaix A., Kambiri Y.-A., Druot T. (2026).
> *Numerical optimization of aviation decarbonization scenarios: balancing traffic and
> emissions with maturing energy carriers and aircraft technology.*
> Applied Energy, 412, 127631.
> [https://doi.org/10.1016/j.apenergy.2026.127631](https://doi.org/10.1016/j.apenergy.2026.127631)

:::{dropdown} BibTeX
```bibtex
@article{costaalves2026,
    title = {Numerical optimization of aviation decarbonization scenarios: balancing traffic and emissions with maturing energy carriers and aircraft technology},
    journal = {Applied Energy},
    volume = {412},
    pages = {127631},
    year = {2026},
    issn = {0306-2619},
    doi = {https://doi.org/10.1016/j.apenergy.2026.127631},
    url = {https://www.sciencedirect.com/science/article/pii/S0306261926002837},
    author = {Ian Costa-Alves and Nicolas Gourdain and François Gallard and Anne Gazaix and Yri Amandine Kambiri and Thierry Druot},
    keywords = {Multidisciplinary optimization, Low-carbon fuels, Aircraft design, Integrated assessment models, Shared socioeconomic pathways},
}
```
:::

:::{admonition} About this documentation
:class: note
This documentation site, and the reorganization of the repository around it, were
prepared with the assistance of Anthropic's Claude (Opus 4.8): structuring the pages,
wiring the example galleries, drafting the getting-started, extending, and API
material, and migrating the build from MkDocs to Sphinx. The [extended paper](paper/index.md)
reproduces the authors' own manuscript and supplementary information verbatim, and the
scientific models and results are entirely the authors' work.
:::

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`rocket` Getting started
:link: getting_started/index
:link-type: doc

Install NOADS, learn the core concepts, and run your first scenario optimization in
10 minutes.
:::

:::{grid-item-card} {octicon}`book` Extended paper
:link: paper/index
:link-type: doc

The companion paper in web form: models, results, supplementary material, and the
runnable examples that produce every figure.
:::

:::{grid-item-card} {octicon}`tools` Extending the analysis
:link: extending/index
:link-type: doc

Partition the fleet differently, add aircraft, pathways, or energy resources, and
quantify uncertainty.
:::

:::{grid-item-card} {octicon}`code` API reference
:link: reference/index
:link-type: doc

The complete API documentation generated from the source docstrings.
:::
::::

## Key capabilities

- **Fleet and aircraft design**: model current and future aircraft across market
  segments (commuter to long-range); new aircraft are sized with the Generic Airplane
  Model (GAM) under evolving technology assumptions.
- **Energy production system**: trace fuels from primary resources (biomass,
  electricity, crude oil) through production pathways (HEFA, Fischer–Tropsch,
  electrolysis) to final energy carriers, with full CO₂ and cost accounting.
- **Temporal scenarios**: optimize policy trajectories from 2025 to 2075 with
  time-dependent controls, ODE dynamics for technology diffusion, and cumulative
  constraint budgets.
- **Climate scenario integration**: drive scenarios with IPCC AR6 data: SSP pathways
  for GDP and population, carbon budgets for 1.5°C to 2.5°C targets, and
  biomass/electricity availability bounds.
- **Fast gradient-based optimization**: JAX automatic differentiation provides exact
  gradients, enabling speedups of two orders of magnitude over finite-difference
  approaches.
- **Robust and multi-objective policies**: find policies that are robust across
  multiple climate futures, or explore Pareto trade-offs between emissions and cost.

## Quick example

```python
from noads.application.examples import single_policy_scenario_optimization

results = single_policy_scenario_optimization(
    global_scenario_name="SSP2-26",  # SSP2 pathway, 2°C target
    technology_index=1,               # mid-tech assumptions
    carbon_budget_percent=3,          # aviation's share of global CO₂ budget
    plot_optimum=True,
)
```

## Credits

The [Generic Airplane Model (GAM)](https://gitlab.com/m6029/genericairplanemodel) was
re-written in JAX together with Yri Amandine Kambiri. See
[Kambiri et al., *Energy consumption of Aircraft with new propulsion systems and storage
media*, SciTech Forum, Orlando, January 2024](https://doi.org/10.2514/6.2024-1707).

Open-access data sources used for calibration:

- **World Bank**: Population, GDP, air transport departures (World Development
  Indicators).
- **ICAO**: Revenue Passenger Kilometers (via Airlines for America).
- **IPCC AR6**: Primary energy, electricity, emissions, GDP, and population projections
  ([AR6 Scenario Database](https://data.ece.iiasa.ac.at/ar6/), Byers et al., 2022).

## License

- Source code: [GNU LGPL v3](https://www.gnu.org/licenses/lgpl-3.0.en.html) (GPL v3
  for AeroMAPS-derived equations)
- Examples: BSD 0-Clause
- Documentation: CC BY-SA 4.0

```{toctree}
:hidden:
:maxdepth: 2

getting_started/index
paper/index
extending/index
reference/index
credits
licenses
```
