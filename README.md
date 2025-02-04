
# Numerical Optimization of Aviation Decarbonization Scenarios (NOADS)

[![PyPI - License](https://img.shields.io/pypi/l/numerical-optimization-of-aviation-decarbonization-scenarios)](https://www.gnu.org/licenses/lgpl-3.0.en.html)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/numerical-optimization-of-aviation-decarbonization-scenarios)](https://pypi.org/project/numerical-optimization-of-aviation-decarbonization-scenarios/)
[![PyPI](https://img.shields.io/pypi/v/numerical-optimization-of-aviation-decarbonization-scenarios)](https://pypi.org/project/numerical-optimization-of-aviation-decarbonization-scenarios/)
[![Codecov branch](https://img.shields.io/codecov/c/gitlab/gemseo:dev/numerical-optimization-of-aviation-decarbonization-scenarios/develop)](https://app.codecov.io/gl/gemseo:dev/numerical-optimization-of-aviation-decarbonization-scenarios)

## Overview

Numerical Optimization of Aviation Decarbonization Scenarios with GEMSEO-JAX

Despite being considered a hard-to-abate sector, aviation's emissions will play an important role in long-term climate mitigation of transportation. The introduction of low-carbon energy carriers and the deployment of new aircraft in the current fleet are modeled as a technology-centered decarbonization policy, and supply constraints in targeted market segments are modeled as demand-side policy.

Mitigation scenarios are formulated as optimization problems and three applications are demonstrated: single-policy optimization, scenario-robust policy, and multi-objective policy trade-off.

The usual burdens associated with nonlinear optimization with high-dimensional variables are dealt with by jointly using libraries for Multidisciplinary Optimization (GEMSEO) and Automatic Differentiation (JAX), which resulted in speedups of two orders of magnitude at the optimization level, while reducing associated implementation efforts.


## Credits

The [Generic Airplane Model (GAM)](https://gitlab.com/m6029/genericairplanemodel) code was re-written in JAX together with Yri Amandine KAMBIRI. For more information on the GAM model please check [Kambiri et al., *Energy consumption of Aircraft with new propulsion systems and storage media*, Scitech Forum, Orlando, January 2024](https://doi.org/10.2514/6.2024-1707).

## Installation

Install the latest version with `pip install numerical-optimization-of-aviation-decarbonization-scenarios`.

See [pip](https://pip.pypa.io/en/stable/getting-started/) for more information.

## Bugs and questions

Please use the [gitlab issue tracker](https://gitlab.com/gemseo/dev/numerical-optimization-of-aviation-decarbonization-scenarios/-/issues)
to submit bugs or questions.

## Contributing

See the [contributing section of GEMSEO](https://gemseo.readthedocs.io/en/stable/software/developing.html#dev).

## Contributors

- Ian COSTA ALVES
- François GALLARD
- Yri Amandine KAMBIRI
- Thierry DRUOT
