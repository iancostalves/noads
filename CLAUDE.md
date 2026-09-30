<!--
 Copyright 2025 ISAE-SUPAERO, https://www.isae-supaero.fr/en/
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# CLAUDE.md — Project Context for AI Assistants

## Project Overview

**NOADS** (Numerical Optimization of Aviation Decarbonization Scenarios) is a Python
framework for optimizing aviation decarbonization pathways using multidisciplinary
optimization (MDO) with automatic differentiation via GEMSEO-JAX.

It models the interplay between fleet modernization, alternative energy carriers
(biofuels, synthetic fuels, hydrogen, batteries), air traffic demand, and climate
constraints (IPCC AR6 carbon budgets) over time horizons from 2025 to 2075.

## Repository Layout

```
noads/
├── src/noads/              # Main source code
│   ├── core/
│   │   ├── model.py        # Base classes: Model, JAXModel, AutoModel (GEMSEO-JAX wrappers)
│   │   ├── models/
│   │   │   ├── fleet/      # Aircraft fleet: operations, design, tech parameters, market dynamics
│   │   │   ├── energy/     # Energy carriers, production pathways, impacts, energy mix
│   │   │   ├── traffic.py  # Air traffic demand (generalized logistic from GDP per capita)
│   │   │   └── interpolation.py
│   │   └── scenarios/
│   │       ├── temporalscenario.py  # Time-dependent scenario with ODE integration (diffrax)
│   │       └── multiscenario.py     # Vectorization across SSP climate scenarios
│   ├── application/        # High-level setup: base objects, scenario config, AR6 data, viz
│   ├── demand_calibration/ # Demand forecasting calibration (RPK, regional departures)
│   └── gam_jax/            # Generic Airplane Model — aircraft sizing & performance in JAX
│                           #   generic_airplane_model.py: V2.0 port (paper, frozen)
│                           #   gam_v3.py: V3.0 port + powertrain scaling + LH2 tank size law
├── docs/                   # Documentation (Sphinx + MyST + sphinx-book-theme)
│   ├── conf.py             # Sphinx configuration (gallery, bibtex, autodoc)
│   ├── getting_started/    # Onboarding: installation, concepts, quickstart
│   ├── paper/              # "Main paper": manuscript + SI converted to MyST
│   │   ├── models/         #   Models section (paper Methods + SI detail, verbatim prose)
│   │   ├── results/        #   Scenario results (+ SI full results gallery)
│   │   ├── figures/        #   Committed PNG figure assets
│   │   └── latex_src/      #   Original LaTeX sources + vector figures
│   ├── update/             # "Update paper": opt-in model updates (aircraft_model="update")
│   ├── extending/          # How to extend: fleet, aircraft, pathways, resources, UQ
│   ├── examples/           # sphinx-gallery sources; optimization/results/ holds
│   │                       #   pre-computed optima (NOADS_RESULTS_DIR)
│   └── reference/          # API reference entry point (autosummary)
├── tests/                  # pytest tests
├── requirements/           # Frozen dependency files per Python version
├── pyproject.toml          # Package metadata, dependencies, entry points
└── tox.ini                 # Test/build/doc automation
```

## Tech Stack

- **Python 3.9–3.12**
- **GEMSEO-JAX** (`>=2.0.2, <3`) — MDO framework with JAX backend
- **JAX** — Automatic differentiation, vectorization (`vmap`), functional transforms
- **diffrax** (`>=0.7,<0.8`) — ODE solvers (Dopri5) for temporal scenario integration
- **optimistix** — Root-finding for aircraft sizing (Newton's method)
- **xlrd** — Reading AR6 scenario data from Excel files
- **Sphinx** (MyST + sphinx-book-theme + sphinx-gallery) — Documentation generation

## Key Concepts

- **AutoModel / JAXModel / Model**: Wrappers around GEMSEO-JAX disciplines. AutoModel auto-detects inputs/outputs from function signatures.
- **TemporalScenario**: Assembles models into a time-dependent optimization problem with ODE integration, control interpolation, and constraint evaluation.
- **MultiScenario**: Vectorizes a TemporalScenario across multiple AR6/SSP background scenarios for robust optimization.
- **Fleet**: Combines multiple AircraftOperation and AircraftDesign objects competing for market share via sigmoid adoption curves.
- **EnergyMix / ProductionPathway**: Models energy supply chains with CO2, cost, and resource impacts.
- **GAM (Generic Airplane Model)**: Physics-based aircraft sizing using empirical regressions for preliminary design.
- **Aircraft models**: `aircraft_model="paper"` (default, GAM V2.0, must reproduce the paper — guarded by `tests/test_gam_paper_baseline.py`) or `"update"` (GAM V3.0, powertrain scale effects with max power per propulsor, size-dependent LH2 tank GI). Never change the paper model's numbers.

## Development Commands

```bash
# Install in dev mode
pip install -e ".[test,doc]"

# Run tests
tox -e test          # or: pytest tests/

# Build docs
tox -e doc           # sphinx-build into docs/_build/html (doc-serve for live reload)

# Lint
tox -e check         # ruff via pre-commit
```

## Conventions

- Code style enforced by **ruff** (`.ruff.toml`)
- Pre-commit hooks configured (`.pre-commit-config.yaml`)
- License: **LGPL-3.0** for core code, **GPL-3.0** for equations/application layer (AeroMAPS-derived)
- Examples licensed under **BSD-0-Clause**, docs under **CC-BY-SA-4.0**
- CI runs on GitLab (`.gitlab-ci.yml`)

## Domain Context

- **SSP**: Shared Socioeconomic Pathways (IPCC scenarios — SSP1 to SSP5)
- **AR6**: IPCC 6th Assessment Report — provides carbon budgets and socioeconomic projections
- **RPK**: Revenue Passenger Kilometers — standard air traffic demand metric
- **SAF**: Sustainable Aviation Fuel (biofuels, e-fuels)
- **GAM**: Generic Airplane Model — statistical aircraft design tool
- **AeroMAPS**: Aviation Environmental Assessment Model (upstream project whose equations are partially reused)
- Tech parameters evolve across three horizons: **2020** (current), **2040** (near-term), **2060** (long-term)

## Important Files for Common Tasks

| Task | Key Files |
|------|-----------|
| Add a new aircraft type | `application/base_objects.py`, `core/models/fleet/aircraft_operation.py` |
| Add an energy pathway | `core/models/energy/production_pathway.py`, `application/scenario_setup.py` |
| Modify optimization setup | `application/scenario_setup.py`, `application/examples.py` |
| Change AR6 scenario data | `application/background_scenario_data.py` |
| Modify aircraft physics | `gam_jax/models/gam_v3.py` (update model; `generic_airplane_model.py` is the frozen paper model) |
| Update aircraft technology | `application/base_objects.py` (update params), `application/lh2_tank_calibration.py`, `application/aircraft_tech_data/` |
| Add a new constraint | `core/scenarios/temporalscenario.py` |
| Visualization | `application/visualization.py` |
| Tests | `tests/test_model.py`, `tests/test_scenarios.py`, etc. |
