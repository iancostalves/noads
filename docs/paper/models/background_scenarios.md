<!--
 Copyright 2025 ISAE-SUPAERO, https://www.isae-supaero.fr/en/
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

(sec-paper-background)=

# Background scenarios

The Shared Socioeconomic Pathways (SSP) were introduced between IPCC's 5th and 6th
Assessment Report (AR5 and AR6) as a way to bridge the socioeconomic dimension of
mitigation into climate assessments {cite:p}`riahi_shared_2017`. These scenarios are
formulated based on qualitative stories first, which define high-level assumptions
that steer IAM simulations into different possible futures regarding economic
development, demographics, energy production, and land-use. Each SSP is created from a
storyline to represent a consistent underlying logic to the depth of socioeconomic
changes that societies are expected to have:

- SSP1: Sustainability – Taking the Green Road (Low challenges to mitigation and
  adaptation): "The world shifts gradually, but pervasively, toward a more sustainable
  path, emphasizing more inclusive development that respects perceived environmental
  boundaries. Consumption is oriented toward low material growth and lower resource
  and energy intensity" {cite:p}`ssp1`.
- SSP2: Middle of the Road (Medium challenges to mitigation and adaptation): "The
  world follows a path in which social, economic, and technological trends do not
  shift markedly from historical patterns. Global and national institutions work
  toward but make slow progress in achieving sustainable development goals"
  {cite:p}`ssp2`.
- SSP5 Fossil-fueled Development – Taking the Highway (High challenges to mitigation,
  low challenges to adaptation): "This world places increasing faith in competitive
  markets, innovation and participatory societies to produce rapid technological
  progress and development of human capital as the path to sustainable development. At
  the same time, the push for economic and social development is coupled with the
  exploitation of abundant fossil fuel resources and the adoption of resource and
  energy-intensive lifestyles around the world" {cite:p}`ssp5`.

After the adoption of the Paris Agreement, studies focused on the investigation of
pathways limiting warming to 1.5°C {cite:p}`ipcc_wg3_an3` in the context of the
Special Report on Global Warming of 1.5°C (SR1.5). From then until AR6, significant
work has been done in improving the modeling of energy and transportation
technologies, and in multi-model studies covering scenarios from extrapolation current
policy trends and the implementation of NDCs, all of them are made available in the
AR6 scenario database {cite:p}`ar6_database`, which is used to provide for
scenario-dependent inputs.

From each baseline SSP, mitigation scenarios are introduced by incorporating varying
mitigation strategies and are named according to a target radiative forcing by the end
of the century, these are made to match scenario emissions to a Representative
Concentration Pathway (RCP), which are used in the analysis carried by the IPCC
Working Groups 1 and 2. These scenarios are then incorporated by main modeling groups,
each resposible for one IAM, forming an SSP-RCP-IAM matrix (see
{cite:p}`ipcc_wg3_an3`).

The scenario database from the 6th IPCC Assessment Report is used to provide for
inputs for future indicators of socioeconomic drivers, and energy system background
{cite:p}`ar6_database`. The choice of scenario determines: the background population
and GDP of the entire economy, affecting future traffic; the total energy production
for biomass and grid electricity (considered further as primary energy inputs to the
aviation energy system), as well as the associated electricity emission intensity
({numref}`fig-availability`).

```{figure} ../figures/figs_models/ar6_data.png
:name: fig-availability
:width: 70%

Scenario-dependent inputs from the AR6 database: population, income per capita,
emission factor of grid electricity, electricity production, and biomass production.
Source {cite:p}`ar6_database`.
```

The simulated results, are based on scenarios: SSP1 with RCP 1.9 from model
WITCH-GLOBIOM 3.1, SSP2 with RCP's 1.9, 2.6, and 3.4 from MESSAGE-GLOBIOM 1.0, and
SSP5 with RCP 4.5 from REMIND-MAgPIE 1.5. Then in the validation of the final results,
the aviation sector's final energy and emissions are compared with the entire scenario
ensemble from models IMAGE 3.2, REMIND-MAgPIE 2.1-4.3, REMIND-Transport 2.1, and
AIM/Hub-Global 2.0.

## Reproduce these figures

The background-scenario inputs ({numref}`fig-availability`) and the AR6 aviation output
ensemble used for validation are produced by the following example, which reads them
directly from the shipped AR6 database export:

```{eval-rst}
.. minigallery:: examples/models/scenarios/plot_*.py
```
