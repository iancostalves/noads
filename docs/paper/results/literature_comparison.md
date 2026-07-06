<!--
 Copyright 2025 ISAE-SUPAERO, https://www.isae-supaero.fr/en/
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Comparison with literature

```{figure} ../figures/figs/ar6_comparison.png
:name: fig-comparison
:width: 100%

Comparison of aviation final energy demand and direct CO2 emissions between the
scenarios developed in this work and the AR6 scenario ensemble {cite:p}`ar6_database`.
The AR6 shaded envelope reflects diverse assumptions across models regarding aviation
scope, demand growth, efficiency gains, and fuel transitions. The trajectories from
this study remain within the ensemble's range through mid-century, while the stronger
long-term decline results from explicitly modeling per-capita demand saturation and
full fleet renewal.
```

{numref}`fig-comparison` contrasts aviation's projected final energy use and emissions
from the scenarios generated in this work against the AR6 scenario ensemble. The AR6
ensemble exhibits substantial variation, reflecting divergent modeling assumptions
regarding aviation scope (international versus domestic, commercial operations,
freight inclusion), demand growth trajectories, efficiency improvements, and
alternative fuel adoption pathways. The scenarios developed in this study fall within
the inter-model spread of the AR6 ensemble for both energy consumption and CO2
emissions through approximately 2050, demonstrating broadly consistent near- to
mid-term dynamics.

Beyond 2050, the present work projects lower energy and emissions compared to much of
the AR6 ensemble. This divergence stems primarily from the incorporation of demand
saturation effects—as per-capita incomes rise according to SSP narratives, the
logistic demand model captures the stabilization of air travel propensity at high
income levels, rather than assuming continued growth. This comparison validates the
plausibility of the scenarios presented in this work, as they remain consistent with
the AR6 range while providing additional insight through explicit modeling of demand
saturation dynamics, technology-specific aircraft performance, and resource
constraints.

## Reproduce this comparison

The AR6 overlay figure is produced by the following script, which loads all the
pre-computed optima shipped with the repository:

```{eval-rst}
.. minigallery:: examples/optimization/ar6_comparison/*.py
```
