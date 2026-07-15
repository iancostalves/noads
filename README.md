<!--
Copyright 2025 ISAE-SUPAERO, https://www.isae-supaero.fr/en/
Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Numerical Optimization of Aviation Decarbonization Scenarios (NOADS)

[![PyPI](https://img.shields.io/pypi/v/noads.svg)](https://pypi.org/project/noads/)
[![Documentation](https://github.com/iancostalves/noads/actions/workflows/docs.yml/badge.svg)](https://iancostalves.github.io/noads/)
[![License: LGPL v3](https://img.shields.io/badge/license-LGPL--3.0-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0.en.html)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)

## Overview

Numerical Optimization of Aviation Decarbonization Scenarios with GEMSEO-JAX.

Despite being considered a hard-to-abate sector, aviation's emissions will play an
important role in long-term climate mitigation of transportation. The introduction of
low-carbon energy carriers and the deployment of new aircraft in the current fleet are
modeled as a technology-centered decarbonization policy, and supply constraints in
targeted market segments are modeled as demand-side policy.

Mitigation scenarios are formulated as optimization problems and three applications
are demonstrated: no-policy baselines, single-policy optimization, and scenario-robust
policies.

The usual burdens associated with nonlinear optimization with high-dimensional
variables are dealt with by jointly using libraries for Multidisciplinary Optimization
([GEMSEO](https://gemseo.readthedocs.io/)) and Automatic Differentiation
([JAX](https://docs.jax.dev/)), which resulted in speedups of two orders of magnitude
at the optimization level, while reducing associated implementation efforts.

NOADS accompanies the paper *Numerical optimization of aviation decarbonization
scenarios: balancing traffic and emissions with maturing energy carriers and aircraft
technology* (Costa-Alves et al., Applied Energy, 2026,
[doi:10.1016/j.apenergy.2026.127631](https://doi.org/10.1016/j.apenergy.2026.127631)).
The documentation includes the paper as a browsable "extended paper" with the full
supplementary information.

## Citation

If you use NOADS in your research, please cite the paper:

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

## Installation

Install the latest version from PyPI (Python 3.10-3.12):

```bash
pip install noads
```

JAX runs on CPU by default, which is sufficient to reproduce all the scenarios.

## Quick start

```python
from noads.application.examples import single_policy_scenario_optimization

results = single_policy_scenario_optimization(
    global_scenario_name="SSP2-26",  # SSP2 pathway, 2°C target
    technology_index=1,               # mid-tech assumptions
    carbon_budget_percent=3,          # aviation's share of the global CO2 budget
    plot_optimum=True,
)
```

The documentation covers a
[10-minute quickstart](docs/getting_started/quickstart.md), the core concepts, a user
guide, runnable example galleries for the elementary models and every scenario family
of the paper, and the API reference. Pre-computed optima are shipped in
`docs/examples/optimization/results/` so the comparison examples run in seconds; the
`run_*.py` scripts reproduce each optimization from scratch.

## Documentation

The built documentation, including the ["extended paper"](docs/paper/index.md) and
the API reference, is published at
**[iancostalves.github.io/noads](https://iancostalves.github.io/noads/)**. The `docs`
GitHub Actions workflow runs the test suite first (unit tests plus the
documentation-deployability checks in `tests/test_docs.py`) and only builds and
publishes the site, on pushes to `main`, once the tests pass.

To build it locally:

```bash
git clone https://github.com/iancostalves/noads.git
cd noads
pip install -e ".[doc]"
sphinx-build -b html docs docs/_build/html
```

Then open `docs/_build/html/index.html` in a browser. Equivalently, with
[tox](https://tox.wiki):

```bash
tox -e doc          # one-off build into docs/_build/html
tox -e doc-serve     # live-reloading local server (sphinx-autobuild)
```

The first build re-executes the `plot_*.py` example scripts (a few minutes, loading
the pre-computed optima shipped in the repo); `-W --keep-going` is used in CI to fail
on any warning. `run_*.py` scripts (full scenario optimizations, hours of compute) are
rendered without execution and are not part of the build.

## Development

Clone the repository and install in editable mode with the development extras:

```bash
git clone https://github.com/iancostalves/noads.git
cd noads
pip install -e ".[test,doc]"
```

Common tasks (automated with [tox](https://tox.wiki), using
[tox-uv](https://github.com/tox-dev/tox-uv)):

```bash
tox -e test         # run the test suite (or: pytest tests/)
tox -e check        # lint and format via pre-commit (ruff)
tox -e doc          # build the documentation with Sphinx into docs/_build/html
tox -e doc-serve    # serve the documentation with live reload
tox -e dist         # build and check the PyPI distribution
tox -e update-deps  # regenerate the frozen requirements files
```

The documentation is built with Sphinx (MyST markdown, sphinx-book-theme,
sphinx-gallery). Example scripts under `docs/examples/` follow the sphinx-gallery
format: only `plot_*.py` scripts are executed during the build; `run_*.py` scripts
(full optimizations) are rendered without execution.

## Credits

The [Generic Airplane Model (GAM)](https://gitlab.com/m6029/genericairplanemodel) code
was re-written in JAX together with Yri Amandine KAMBIRI. For more information on the
GAM model please check [Kambiri et al., *Energy consumption of Aircraft with new
propulsion systems and storage media*, Scitech Forum, Orlando, January
2024](https://doi.org/10.2514/6.2024-1707).

Furthermore, open-access data are also used for calibration of air traffic demand
(World Bank, ICAO) and for linking the background with a scenario from the IPCC's 6th
Assessment Report (AR6 scenario database):

- Population, total, SP.POP.TOTL; GDP (current US$), NY.GDP.MKTP.CD; Air transport,
  registered carrier departures worldwide, IS.AIR.DPRT. World Development Indicators.
  World Bank Group Archives, Washington, D.C., United States.

- RPK. World Airlines Traffic and Capacity, Airlines for America. Source: ICAO. url:
  <https://www.airlines.org/dataset/world-airlines-traffic-and-capacity/>

- Primary Energy (Biomass); Final Energy (Electricity); Emissions Energy Supply
  (Electricity); GDP; Population. Variables are taken for a select scenario range, C1
  (limit warming to 1.5°C with no or limited overshoot) to C6 (limit warming to 3°C),
  and can be vizualized
  [here](https://data.ece.iiasa.ac.at/ar6//#/workspaces/share/3b43eae5-2f6f-494d-8376-146fd252c11d).
  Source: Byers et al, 2022. AR6 Scenarios Database hosted by IIASA. International
  Institute for Applied Systems Analysis, 2022. doi: 10.5281/zenodo.5886911 | url:
  data.ece.iiasa.ac.at/ar6/

## Licences

Most of the repository is under the GNU Lesser General Public License v3 (LGPL3), same
as [GEMSEO](https://github.com/gemseo/gemseo) and
[GAM](https://gitlab.com/m6029/genericairplanemodel).

However, some equations, such as sigmoid functions instead of custom time-dependent
controls for aircraft penetration [noads.core.models.fleet.aircraft_operation] and
quadratic load factor [noads.core.models.traffic] are a direct JAX re-implementation
of [AeroMAPS](https://github.com/AeroMAPS/AeroMAPS) models, therefore they are left
with GNU General Public License v3 (GPL3). The model instantiation
[noads.application.base_objects], scenario setup
[noads.application.scenario_setup.single_scenario_setup], scenario optimization
examples [noads.application.examples] and the documentation that depend on these are
also GPL3.

The examples are distributed under the BSD 0-Clause license and the documentation
under CC BY-SA 4.0.

## Bugs and questions

Please use the [GitHub issue tracker](https://github.com/iancostalves/noads/issues)
to submit bugs or questions.

## Contributing

See the
[contributing section of GEMSEO](https://gemseo.readthedocs.io/en/stable/software/developing.html#dev).

## Contributors

- Ian COSTA ALVES
- François GALLARD
- Yri Amandine KAMBIRI
