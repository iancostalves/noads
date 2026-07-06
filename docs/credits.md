<!--
 Copyright 2025 ISAE-SUPAERO, https://www.isae-supaero.fr/en/
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Credits

## People and projects

- The models build on the [AeroMAPS](https://aeromaps.isae-supaero.fr/) platform
  developed at ISAE-SUPAERO; some equations of the application layer are derived from
  it (GPL v3).
- The [Generic Airplane Model (GAM)](https://gitlab.com/m6029/genericairplanemodel),
  provided by the CADO team (ENAC), was re-written in JAX together with
  Yri Amandine Kambiri
  ([Kambiri et al., 2024](https://doi.org/10.2514/6.2024-1707)).
- Matthias De Lozzo and Antoine Dechaume are gratefully acknowledged for repository
  maintenance and thorough code reviews.

## Software dependencies

NOADS stands on the shoulders of:

- [GEMSEO](https://gemseo.readthedocs.io/) and
  [GEMSEO-JAX](https://gitlab.com/gemseo/dev/gemseo-jax): multidisciplinary
  design optimization with a JAX backend (LGPL v3).
- [JAX](https://docs.jax.dev/): automatic differentiation, JIT compilation, and
  vectorization (Apache 2.0).
- [diffrax](https://docs.kidger.site/diffrax/): differentiable ODE solvers
  (Apache 2.0).
- [optimistix](https://docs.kidger.site/optimistix/): root-finding for aircraft
  sizing (Apache 2.0).
- [NumPy](https://numpy.org/), [SciPy](https://scipy.org/),
  [matplotlib](https://matplotlib.org/), and
  [plotly](https://plotly.com/python/): scientific computing and visualization.

## Data sources

- **World Bank**: Population, GDP, and air transport departures (World Development
  Indicators).
- **ICAO**: Revenue Passenger Kilometers (via Airlines for America).
- **IPCC AR6**: Primary energy, electricity, emissions, GDP, and population
  projections ([AR6 Scenario Database](https://data.ece.iiasa.ac.at/ar6/),
  Byers et al., 2022).
- **AeroSCOPE**: 2019 aircraft fleet database used to initialize the current fleet.
