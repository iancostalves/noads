<!--
 Copyright 2025 ISAE-SUPAERO, https://www.isae-supaero.fr/en/
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Installation

## Requirements

- **Python 3.9 to 3.12**
- A working [pip](https://pip.pypa.io/en/stable/getting-started/) installation

NOADS depends on [GEMSEO-JAX](https://gitlab.com/gemseo/dev/gemseo-jax) (multidisciplinary optimization with automatic differentiation),
[JAX](https://github.com/jax-ml/jax) (numerical computing and auto-diff), and
[diffrax](https://github.com/patrick-kidger/diffrax) (differential equation solvers).
These are installed automatically.

## Install from PyPI

```bash
pip install numerical-optimization-of-aviation-decarbonization-scenarios
```

## Install from source (development)

Clone the repository and install in editable mode:

```bash
git clone https://gitlab.com/gemseo/dev/numerical-optimization-of-aviation-decarbonization-scenarios.git
cd numerical-optimization-of-aviation-decarbonization-scenarios
pip install -e ".[test,doc]"
```

This installs NOADS with all development dependencies (testing, documentation).

## Verify the installation

```python
import noads
from noads.application.scenario_setup import single_scenario_setup
print("NOADS is ready.")
```

## Optional: JAX GPU support

By default, JAX runs on CPU. For GPU acceleration (useful for large-scale optimizations),
follow the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html)
to install the CUDA-enabled version of JAX **before** installing NOADS.

## Development tools

NOADS uses [tox](https://tox.wiki/) for automation. After installing from source:

```bash
# Run the test suite
tox -e test

# Build the documentation locally (or tox -e doc-serve for live reload)
tox -e doc

# Run the linter (ruff, via pre-commit)
tox -e check
```

Pre-commit hooks are configured for code quality. To activate them:

```bash
pip install pre-commit
pre-commit install
```
