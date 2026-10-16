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

## Energy pathways

The energy pathways of the update model are those of the main paper: the
efficiencies of electrolysis, power-to-liquid and hydrogen liquefaction improve until
2050 {cite:p}`u-wallington_green_2024`, and those of refining, the biofuel pathways
and battery charging are constant. The technology scenarios of the update model vary
aircraft technology only.

The biofuel efficiencies are the kerosene efficiencies of
{cite:t}`u-NEULING201854` (lower heating value, all inputs), averaged over two
feedstocks per route: 58 to 60 % for HEFA (jatropha and palm oil), 18 to 42 % for
alcohol-to-jet (wheat straw and grain) and 19 to 21 % for Fischer-Tropsch
biomass-to-liquid (wheat straw and willow). The overall efficiencies of these
processes, counting their diesel, naphtha and butane co-products, are higher (90 %,
30 to 68 % and 35 to 38 %). The HEFA efficiency of 0.59 (1.69 MJ of biomass per MJ)
is consistent with this source; the 1.95 MJ/MJ in the table of the main paper is a
typographical error.

## Primary energy per final energy

With all hydrogen produced by electrolysis, the primary energy (oil, biomass or grid
electricity, as in the main paper) consumed per MJ of final energy delivered to the
aircraft chains the pathway efficiencies
({mod}`noads.application.primary_energy`): e-fuel needs 3.3 MJ of electricity per MJ
in 2025 and 2.8 MJ from 2050, LH2 1.6 MJ and 1.5 MJ, batteries 1.02 MJ.

```{figure} figures/primary_energy_update.png
:name: fig-update-primary-energy
:width: 100%

Primary energy per final energy of the energy carriers, versus year. The pathways do
not depend on the technology or background (SSP) scenario.
```

## Primary energy of prospective aircraft

The primary energy per seat-km of an aircraft is its energy per seat-km (design
mission) times the primary energy per final energy of its carrier at its EIS; the
bands come from aircraft technology only. The architectures are compared per primary resource.

```{figure} figures/aircraft_update_primary_oil.png
:name: fig-update-primary-oil
:width: 100%

Oil per seat-km of the Jet-A aircraft, compared with the 2019 fleet (grey). Filled
between the Upper and Lower scenarios, solid line for Lower and dotted line for Mid.
Turboprops are shown for comparison, although they are not in the optimized fleets.
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

On the design missions, the new Jet-A turbofans of the update model reach the oil
consumption of the 2019 fleet average by 2040 in the Mid scenario, thanks to the
engine efficiency trend ({ref}`sec-update-aircraft`), but remain above it in 2025 on
the regional and long range markets, where the design range exceeds the average
stage length and turboprops dominate today's regional fleet. After 2040,
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
