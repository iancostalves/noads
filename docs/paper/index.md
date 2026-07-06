<!--
 Copyright 2025 ISAE-SUPAERO, https://www.isae-supaero.fr/en/
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Extended paper

**Numerical optimization of aviation decarbonization scenarios: balancing traffic and
emissions with maturing energy carriers and aircraft technology**

Ian Costa-Alves{sup}`1,2`, Nicolas Gourdain{sup}`1`, François Gallard{sup}`2`,
Anne Gazaix{sup}`2`, Yri Amandine Kambiri{sup}`1,3`, Thierry Druot{sup}`3`

{sup}`1` Aerodynamics, Energetics and Propulsion Department, ISAE-SUPAERO, Toulouse,
France —
{sup}`2` Multidisciplinary Optimization Competence Center, IRT Saint Exupéry, Toulouse,
France —
{sup}`3` Conceptual Airplane Design and Operations, ENAC, Toulouse, France

:::{note}
This section is the web version of the companion paper, published as:
Costa-Alves I., Gourdain N., Gallard F., Gazaix A., Kambiri Y.-A., Druot T. (2026).
*Numerical optimization of aviation decarbonization scenarios: balancing traffic and
emissions with maturing energy carriers and aircraft technology.* Applied Energy, 412,
127631. [doi:10.1016/j.apenergy.2026.127631](https://doi.org/10.1016/j.apenergy.2026.127631).
The LaTeX sources are kept in the repository under `docs/paper/latex_src/`.
:::

## Abstract

Despite being considered a hard-to-abate sector, aviation's emissions will play an
important role in long-term climate mitigation of transportation. The introduction of
low-carbon energy carriers and the deployment of new aircraft in the current fleet are
modeled as technology-centered decarbonization policies, while supply constraints in
targeted market segments are modeled as demand-side policies. Shared Socioeconomic
Pathways (SSPs) are used to estimate trend-mitigation traffic demand and to limit the
sectoral consumption of electricity and biomass. Mitigation scenarios are formulated as
optimization problems, and three applications are demonstrated: no-policy baselines,
single-policy optimization, and scenario-robust policies. Results show that the choice
of energy carrier is highly dependent on assumptions regarding aircraft technology and
the background energy system. Across all SSP-based scenarios, emissions peak by around
2040, but achieving alignment with the Paris Agreement requires either targeted demand
management or additional low-carbon energy supply. The use of gradient-based
optimization within a multidisciplinary framework enables the efficient resolution of
these nonlinear, high-dimensional problems while reducing implementation effort.

## Highlights

- Optimization framework links SSP scenarios with detailed aircraft technology;
- Fleet replacement and demand saturation leads to emissions peak around 2040;
- Impact of alternative aircraft requires extending analysis beyond 2050;
- Respecting +2°C carbon budgets requires demand caps or extra energy availability;
- Policy optimization was prohibitive without speedups from numerical methodology.

**Keywords**: Multidisciplinary Optimization; Low-carbon fuels; Aircraft design;
Integrated Assessment Models; Shared Socioeconomic Pathways

```{figure} figures/figs/graphical_abstract.png
:name: fig-graphical-abstract
:width: 100%

Graphical abstract.
```

## Acronyms

AD
: Automatic Differentiation

ASK
: Available Seat Kilometers

ATJ
: Alcohol-to-Jet

BtL
: Biomass-to-Liquid

EIS
: Entry-Into-Service

FD
: Finite Differences

FT
: Fischer-Tropsch

GAM
: Generic Airplane Model

GDP
: Gross Domestic Product

HEFA
: Hydroprocessed Esters and Fatty Acids

IAM
: Integrated Assessment Model

IPCC
: Intergovernmental Panel on Climate Change

JIT
: Just-In-Time

LH2
: Liquid hydrogen

MDO
: Multidisciplinary Optimization

PtL
: Power-to-Liquid

RCP
: Representative Concentration Pathway

RPK
: Revenue Passenger Kilometers

SAF
: Sustainable Aviation Fuel

SSP
: Shared Socioeconomic Pathways

TLAR
: Top Level Aircraft Requirements

The mathematical symbols used throughout the models are collected in the
[nomenclature](nomenclature.md).

## Code availability

All the scripts and data required to reproduce the results from this work are openly
available in <https://github.com/iancostalves/noads>.

## Acknowledgements

Gratitude is extended to the Conceptual Airplane Design and Operations (CADO) team at
École Nationale de l'Aviation Civile (ENAC), to the Aviation, Climate, Environment
(ACE) group at ISAE-SUPAERO, and to the Institute for Sustainable Aviation (ISA) for
their support, assistance, and fruitful discussions. The Generic Aircraft Design Model
(GAM), provided by the CADO team, was essential for enabling modeling and analysis of
alternative aircraft designs. The AeroMAPS platform, developped by ISAE-SUPAERO and
ISA, was reponsible for laying the groundwork upon which this research was built.
Special thanks to Pascal Roches, Nicolas Monrolin, Thomas Planès, Scott Delbecq,
Antoine Salgas, Florian Simatos, Laurent Joly, and Xavier Carbonneau for their sharp
insights, support, and collaboration.

Also, the authors thank the Multidisciplinary Optimization Competence Center at IRT
Saint Exupéry, for their availability and support with the methodological developments
that preceded this research. Special thanks to Matthias De Lozzo, and Antoine Dechaume
for their aid with repository maintenance and thorough code reviews.

## CRediT authorship contribution statement

**Ian Costa-Alves**: Writing – original draft, Writing – review & editing,
Conceptualization, Data curation, Investigation, Methodology, Software, Validation,
Visualization.

**Nicolas Gourdain**: Writing – original draft, Writing – review & editing,
Conceptualization, Methodology, Funding acquisition, Project administration,
Supervision.

**François Gallard**: Writing – original draft, Writing – review & editing,
Conceptualization, Methodology, Software, Supervision.

**Anne Gazaix**: Writing – review & editing, Conceptualization, Funding acquisition,
Project administration, Supervision.

**Yri-Amandine Kambiri**: Writing – review & editing, Software, Validation.

**Thierry Druot**: Writing – review & editing, Conceptualization, Software,
Supervision, Validation.

## Funding sources

This work was supported by the Occitania region, ISAE-SUPAERO, and IRT Saint Exupéry.

```{toctree}
:hidden:

introduction
models/index
numerical_methods
results/index
discussion
conclusion
nomenclature
references
```
