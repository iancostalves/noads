<!--
Copyright 2025 ISAE-SUPAERO, https://www.isae-supaero.fr/en/
Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Numerical Optimization of Aviation Decarbonization Scenarios (NOADS)

[![PyPI - License](https://img.shields.io/pypi/l/numerical-optimization-of-aviation-decarbonization-scenarios)](https://www.gnu.org/licenses/lgpl-3.0.en.html)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/numerical-optimization-of-aviation-decarbonization-scenarios)](https://pypi.org/project/numerical-optimization-of-aviation-decarbonization-scenarios/)
[![PyPI](https://img.shields.io/pypi/v/numerical-optimization-of-aviation-decarbonization-scenarios)](https://pypi.org/project/numerical-optimization-of-aviation-decarbonization-scenarios/)
[![Codecov branch](https://img.shields.io/codecov/c/gitlab/gemseo:dev/numerical-optimization-of-aviation-decarbonization-scenarios/develop)](https://app.codecov.io/gl/gemseo:dev/numerical-optimization-of-aviation-decarbonization-scenarios)

## Work in progress

This repository is still a Work in progress, source files are ready, but the installation and documentation are not deployed yet. To have a preliminary view of the functionalities check `docs/user_guide`, and `docs/examples` for some examples.


## Overview

Numerical Optimization of Aviation Decarbonization Scenarios with GEMSEO-JAX

Despite being considered a hard-to-abate sector, aviation's emissions will play an important role in long-term climate mitigation of transportation. The introduction of low-carbon energy carriers and the deployment of new aircraft in the current fleet are modeled as a technology-centered decarbonization policy, and supply constraints in targeted market segments are modeled as demand-side policy.

Mitigation scenarios are formulated as optimization problems and three applications are demonstrated: single-policy optimization, scenario-robust policy, and multi-objective policy trade-off.

The usual burdens associated with nonlinear optimization with high-dimensional variables are dealt with by jointly using libraries for Multidisciplinary Optimization (GEMSEO) and Automatic Differentiation (JAX), which resulted in speedups of two orders of magnitude at the optimization level, while reducing associated implementation efforts.


## Credits

The [Generic Airplane Model (GAM)](https://gitlab.com/m6029/genericairplanemodel) code was re-written in JAX together with Yri Amandine KAMBIRI. For more information on the GAM model please check [Kambiri et al., *Energy consumption of Aircraft with new propulsion systems and storage media*, Scitech Forum, Orlando, January 2024](https://doi.org/10.2514/6.2024-1707).

Furthermore, open-access data are also used for calibration of air traffic demand (World Bank, ICAO) and for linking the background with a scenario from the IPCC's 6th Assessment Report (AR6 scenario database):

- Population, total, SP.POP.TOTL; GDP (current US$), NY.GDP.MKTP.CD; Air transport, registered carrier departures worldwide, IS.AIR.DPRT. World Development Indicators. World Bank Group Archives, Washington, D.C., United States.

- RPK. World Airlines Traffic and Capacity, Airlines for America. Source: ICAO. url: https://www.airlines.org/dataset/world-airlines-traffic-and-capacity/

- Primary Energy (Biomass); Final Energy (Electricity); Emissions Energy Supply (Electricity); GDP; Population. Variables are taken for a select scenario range, C1 (limit warming to 1.5°C with no or limited overshoot) to C6 (limit warming to 3°C), and can be vizualized [here](https://data.ece.iiasa.ac.at/ar6//#/workspaces/share/3b43eae5-2f6f-494d-8376-146fd252c11d). Source: Byers et al, 2022. AR6 Scenarios Database hosted by IIASA. International Institute for Applied Systems Analysis, 2022. doi: 10.5281/zenodo.5886911 | url: data.ece.iiasa.ac.at/ar6/

## Installation

Install the latest version with `pip install numerical-optimization-of-aviation-decarbonization-scenarios`.

See [pip](https://pip.pypa.io/en/stable/getting-started/) for more information.

## Licences

Most of the repository is under the GNU Lesser General Public License v3 (LGPL3), same as [GEMSEO](https://github.com/gemseo/gemseo) and [GAM](https://gitlab.com/m6029/genericairplanemodel).

However, some equations, such as sigmoid functions instead of custom time-dependent controls for aircraft penetration [noads.core.models.fleet.aircraft_operation] and quadratic load factor [noads.core.models.traffic] are a direct JAX re-implementation of [AeroMAPS](https://github.com/AeroMAPS/AeroMAPS) models, therefore they are left with GNU General Public License v3 (GPL3). The model instantiation [noads.application.base_objects], scenario setup [noads.application.scenario_setup.single_scenario_setup], scenario optimization examples [noads.application.examples] and the documentation that depend on these are also GPL3.

See [pip](https://pip.pypa.io/en/stable/getting-started/) for more information.

## Bugs and questions

Please use the [gitlab issue tracker](https://gitlab.com/gemseo/dev/numerical-optimization-of-aviation-decarbonization-scenarios/-/issues) to submit bugs or questions.

## Contributing

See the [contributing section of GEMSEO](https://gemseo.readthedocs.io/en/stable/software/developing.html#dev).

## Contributors

- Ian COSTA ALVES
- François GALLARD
- Yri Amandine KAMBIRI
