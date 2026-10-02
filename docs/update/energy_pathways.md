<!--
 Copyright 2025 ISAE-SUPAERO, https://www.isae-supaero.fr/en/
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

(sec-update-energy)=

# Energy pathways and primary energy

## Technology-dependent pathway efficiencies

In the main paper, the efficiencies of the energy pathways depend on time only. In the
update model, those of electrolysis, power-to-liquid and hydrogen liquefaction also
depend on the technology scenario: Mid keeps the values of the main paper
{cite:p}`u-wallington_green_2024`, and all scenarios share their 2025 value, their gap
to Mid widening until 2050 (constant afterwards). The ranges come from the literature
bundled in `noads/application/aircraft_tech_data/pathway_literature_data.csv`:

| Pathway | Literature | Lower 2025 / 2035 / 2050 | Mid | Upper |
|---|---|---|---|---|
| Electrolysis [MJ H2 / MJ electricity] | 53 to 55 kWh/kg H2 for 2025 systems (0.61 to 0.63), IRENA 2050 target below 42 kWh/kg (0.79), SOEC 0.85 to 0.90 | 0.71 / 0.715 / 0.72 | 0.71 / 0.73 / 0.75 | 0.71 / 0.76 / 0.80 |
| Liquefaction [MJ LH2 / MJ electricity] | 10 to 15 kWh/kg for existing plants, 6.4 kWh/kg for the IDEALHY design, 2.7 kWh/kg theoretical | 4.54 / 4.75 / 5.0 | 4.54 / 5.45 / 6.36 | 4.54 / 5.75 / 6.76 |
| Power-to-liquid [MJ e-fuel / MJ H2] | Fischer-Tropsch synthesis about 73 % from syngas, power-to-liquid 35 to 50 % overall | 0.53 / 0.54 / 0.55 | 0.53 / 0.56 / 0.59 | 0.53 / 0.59 / 0.65 |
| Power-to-liquid [MJ e-fuel / MJ electricity] | carbon capture and synthesis | 1.53 / 1.58 / 1.63 | 1.53 / 1.65 / 1.77 | 1.53 / 1.75 / 1.95 |

The 2025 values of the main paper are already at the level of the best large-scale
designs (47 kWh/kg H2 for electrolysis, 7.3 kWh/kg for liquefaction), above today's
plants. The biofuel pathways (HEFA, ATJ, FT), refining and battery charging keep the
constant efficiencies of the main paper: the ranges of their biomass-to-jet yields
remain to be documented. The HEFA efficiency of the code (0.59, i.e. 1.69 MJ of
biomass per MJ) differs from the table of the main paper (1.95 MJ/MJ).

## Primary energy per final energy

With all hydrogen produced by electrolysis, the primary energy (oil, biomass or grid
electricity, as in the main paper) consumed per MJ of final energy delivered to the
aircraft chains the pathway efficiencies
({mod}`noads.application.primary_energy`): e-fuel needs 3.3 MJ of electricity per MJ
in 2025 and 2.4 to 3.1 MJ in 2050, LH2 1.6 MJ and 1.4 to 1.6 MJ, batteries 1.02 MJ.

```{figure} figures/primary_energy_update.png
:name: fig-update-primary-energy
:width: 100%

Primary energy per final energy of the energy carriers, versus year. Filled between
the Upper and Lower scenarios, solid line for Lower and dotted line for Mid. The
pathways do not depend on the background (SSP) scenario.
```

## Primary energy of prospective aircraft

The primary energy per seat-km of an aircraft is its energy per seat-km (design
mission) times the primary energy per final energy of its carrier at its EIS, for the
same technology scenario. The architectures are compared per primary resource.

```{figure} figures/aircraft_update_primary_oil.png
:name: fig-update-primary-oil
:width: 100%

Oil per seat-km of the Jet-A aircraft, compared with the 2019 fleet (grey).
Turboprops, which are not in the optimized fleets, are shown by their Mid curve only.
```

```{figure} figures/aircraft_update_primary_biomass.png
:name: fig-update-primary-biomass
:width: 100%

Biomass per seat-km of the Jet-A aircraft on each biofuel pathway.
```

```{figure} figures/aircraft_update_primary_electricity.png
:name: fig-update-primary-electricity
:width: 100%

Electricity per seat-km of the Jet-A aircraft on e-fuel, of the battery-electric
aircraft and of the LH2 aircraft (gas turbine and fuel cell).
```

On the design missions, the new Jet-A turbofans of the update model are not more
efficient than the 2019 fleet on the regional and long range markets: GAM V3.0
penalizes small turbofans, the structure gains are revised downwards, and the
take-off and climb energy is charged at the powertrain efficiency. After 2040,
LH2 gas turbines need about a third less electricity than e-fuel aircraft, fuel cell
aircraft a third to two thirds less than LH2 gas turbines, and battery-electric
aircraft less still, where they exist.

## Reproduce these figures

```{eval-rst}
.. minigallery:: examples/models/energy/plot_primary_energy_update.py
```

## References

```{bibliography}
:keyprefix: u-
:labelprefix: U
:filter: docname in docnames
```
