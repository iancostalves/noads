<!--
 Copyright 2025 ISAE-SUPAERO, https://www.isae-supaero.fr/en/
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Mitigation policy optimization

The outcomes of Drop-in and Breakthrough mitigation are presented in
{numref}`fig-trends-dropin` and {numref}`fig-trends-breakthrough` and compared with
the no-policy baseline SSP2.

Overall, the Drop-in mitigation scenarios are less sensitive to aircraft technology
assumptions, as emission reductions is achieved either with SAF incorporation or
demand aversion. The trend variant reaches emission levels comparable to 2019, but
fails to phase out of consuming fossil fuels. The availability variant further reduces
emission levels, and allow for much lower fossil consumption. The low-demand variant,
traffic is nearly half of the trend by 2070, but still around 50 % higher than the
2019 reference, this variant is the only one that allows for keeping the cumulative
emissions in check with the Paris Agreement, phasing out of fossil consumption, and
lowering the emissions peak.

The Breakthrough mitigation displays much higher sensitivity to aircraft technology,
lower emission levels than Drop-in mitigation (when compared with similar technology),
and lower fossil consumption. The trend variant, for instance, still consumes fossil
fuels and can emit more than 2019 emissions with Lower technology, or less than half
of 2019 with Upper technology. The availability variant can reach near zero emissions
after 2070, phasing out of fossil can happen as soon as 2060, allowing for relaxing
the biomass consumption constraint, but cumulative emissions are still higher than the
sector's fair share. The low-demand variant respects the carbon budget constraint with
much less difficulty than the Drop-in low-demand, allowing for much higher demand
levels, which may reach up to 140 % more traffic than 2019, but is still lower than
the trend 200 % increase.

```{figure} ../figures/figs/dropin_jetfuel.png
:name: fig-jet-dropin
:width: 100%

Comparison of Jet-A fuel blend for the Drop-in scenarios. The sensibility to aircraft
technology is displayed with 3 scenarios of component-level performances: Lower
technology (continuous line), Mid technology (dotted line), and Upper technology
(shadowed region). These impact the performance of current and new aircraft designs,
as well as the speed of fleet renewal.
```

```{figure} ../figures/figs/breakthrough_jetfuel.png
:name: fig-jet-breakthrough
:width: 100%

Comparison of Jet-A fuel blend for the Breakthrough scenarios (legend as in
{numref}`fig-jet-dropin`).
```

{numref}`fig-jet-dropin` and {numref}`fig-jet-breakthrough` compare how the Jet-A fuel
is composed, blending SAF from BtL and PtL into fossil kerosene. The
{ref}`energy production sankey diagrams <fig-sankey>` detail further how energy resources
are split among production pathways in order to provide for final energy carriers, in
a subset of scenarios. Regarding biofuel production, while the HEFA pathway displays
significantly higher emissions compared to FT, it consumes significantly less biomass.
This yields that, in scenarios where fossil kerosene consumption is still high,
biomass is preferentially allocated to HEFA production. For electrofuel production,
Breaktrough scenarios display much lower shares of electrofuel in the Jet-A blend,
because hydrogen and electricity are preferentially allocated to alternative aircraft
rather than to make electrofuel, but this trade-off is highly dependent on the flight
distance considered.

```{figure} ../figures/figs/breakthrough_carriers.png
:name: fig-carrier
:width: 100%

Comparison of supply per final energy carrier for the Breakthrough mitigation
scenarios. The sensibility to aircraft technology is displayed with 3 scenarios of
component-level performances: Lower technology (continuous line), Mid technology
(dotted line), and Upper technology (shadowed region). These impact the performance of
current aircraft, new aircraft, and the speed of fleet renewal, driving the choice of
which architectures to deploy and when they are launched.
```

Regarding the aircraft fleet, {numref}`fig-carrier` shows the overall comparison of
supply according to final energy carrier, while {numref}`fig-fleet-trend-low` to
{numref}`fig-fleet-lowdemand-up` provide the bottom-up view of how aircraft
technologies are composed to make up the supply in each market segment. The general
and commuter markets are consistently the first to transition toward new aircraft,
whereas short-medium and long-range markets remain dependent on drop-in fuels for
longer periods. The Breakthrough scenarios demonstrate the importance of both
technological maturity and availability assumptions: Lower technology delay deployment
of hydrogen systems, but once extra energy availability is assumed alternative
aircraft are launched as soon as available. In contrast, the Upper technology case
allows for more aggressive displacement of conventional aircraft, and also drive the
adoption of Battery-Electric in the general market and of Hydrogen Fuel-Cells on
remaining markets.

```{figure} ../figures/figs/fleet_SSP2-26-lowTech.png
:name: fig-fleet-trend-low
:width: 90%

Aircraft fleet composition per market: Breakthrough trend, Lower technology.
```

```{figure} ../figures/figs/fleet_SSP2-26-Availability-lowTech.png
:name: fig-fleet-availability-low
:width: 90%

Aircraft fleet composition per market: Breakthrough availability, Lower technology.
```

```{figure} ../figures/figs/fleet_SSP2-26-LowDemand-lowTech.png
:name: fig-fleet-lowdemand-low
:width: 90%

Aircraft fleet composition per market: Breakthrough low-demand, Lower technology.
```

```{figure} ../figures/figs/fleet_SSP2-26-upTech.png
:name: fig-fleet-trend-up
:width: 90%

Aircraft fleet composition per market: Breakthrough trend, Upper technology.
```

```{figure} ../figures/figs/fleet_SSP2-26-Availability-upTech.png
:name: fig-fleet-availability-up
:width: 90%

Aircraft fleet composition per market: Breakthrough availability, Upper technology.
```

```{figure} ../figures/figs/fleet_SSP2-26-LowDemand-upTech.png
:name: fig-fleet-lowdemand-up
:width: 90%

Aircraft fleet composition per market: Breakthrough low-demand, Upper technology.
Assumptions on energy availability and aircraft technology significantly impacts the
performance of new aircraft designs, driving the choice of which architectures to
deploy and when they are launched.
```

## Full results: Drop-in

The complete energy-mix results for the Drop-in scenarios (supplementary information
of the published article):

**Drop-in trend**

::::{tab-set}

:::{tab-item} Low technology
![Energy-mix, Drop-in trend, low technology](../figures/figs_results/single_policy/SSP2-26-DropIn-lowTech/energy_SSP2-26-DropIn-lowTech.png)
:::

:::{tab-item} Mid technology
![Energy-mix, Drop-in trend, mid technology](../figures/figs_results/single_policy/SSP2-26-DropIn-midTech/energy_SSP2-26-DropIn-midTech.png)
:::

:::{tab-item} Upper technology
![Energy-mix, Drop-in trend, upper technology](../figures/figs_results/single_policy/SSP2-26-DropIn-upTech/energy_SSP2-26-DropIn-upTech.png)
:::

::::

**Drop-in availability**

::::{tab-set}

:::{tab-item} Low technology
![Energy-mix, Drop-in availability, low technology](../figures/figs_results/single_policy/SSP2-26-DropIn-Availability-lowTech/energy_SSP2-26-DropIn-Availability-lowTech.png)
:::

:::{tab-item} Mid technology
![Energy-mix, Drop-in availability, mid technology](../figures/figs_results/single_policy/SSP2-26-DropIn-Availability-midTech/energy_SSP2-26-DropIn-Availability-midTech.png)
:::

:::{tab-item} Upper technology
![Energy-mix, Drop-in availability, upper technology](../figures/figs_results/single_policy/SSP2-26-DropIn-Availability-upTech/energy_SSP2-26-DropIn-Availability-upTech.png)
:::

::::

**Drop-in low-demand** (Mid technology)

::::{tab-set}

:::{tab-item} Fleet composition
![Fleet composition, Drop-in low-demand, mid technology](../figures/figs_results/single_policy/SSP2-26-DropIn-LowDemand-midTech/fleet_SSP2-26-DropIn-LowDemand-midTech.png)
:::

:::{tab-item} Energy-mix
![Energy-mix, Drop-in low-demand, mid technology](../figures/figs_results/single_policy/SSP2-26-DropIn-LowDemand-midTech/energy_SSP2-26-DropIn-LowDemand-midTech.png)
:::

::::

The Drop-in optimizations and comparison figures are produced by these scripts:

```{eval-rst}
.. minigallery:: examples/optimization/dropin/*.py
```

## Full results: Breakthrough

The complete fleet and energy-mix results for the Breakthrough scenarios:

**Breakthrough trend**

::::{tab-set}

:::{tab-item} Fleet, low tech
![Fleet composition, Breakthrough trend, low technology](../figures/figs_results/single_policy/SSP2-26-lowTech/fleet_SSP2-26-lowTech.png)
:::

:::{tab-item} Fleet, mid tech
![Fleet composition, Breakthrough trend, mid technology](../figures/figs_results/single_policy/SSP2-26-midTech/fleet_SSP2-26-midTech.png)
:::

:::{tab-item} Fleet, upper tech
![Fleet composition, Breakthrough trend, upper technology](../figures/figs_results/single_policy/SSP2-26-upTech/fleet_SSP2-26-upTech.png)
:::

:::{tab-item} Energy, low tech
![Energy-mix, Breakthrough trend, low technology](../figures/figs_results/single_policy/SSP2-26-lowTech/energy_SSP2-26-lowTech.png)
:::

:::{tab-item} Energy, mid tech
![Energy-mix, Breakthrough trend, mid technology](../figures/figs_results/single_policy/SSP2-26-midTech/energy_SSP2-26-midTech.png)
:::

:::{tab-item} Energy, upper tech
![Energy-mix, Breakthrough trend, upper technology](../figures/figs_results/single_policy/SSP2-26-upTech/energy_SSP2-26-upTech.png)
:::

::::

**Breakthrough availability**

::::{tab-set}

:::{tab-item} Fleet, low tech
![Fleet composition, Breakthrough availability, low technology](../figures/figs_results/single_policy/SSP2-26-Availability-lowTech/fleet_SSP2-26-Availability-lowTech.png)
:::

:::{tab-item} Fleet, mid tech
![Fleet composition, Breakthrough availability, mid technology](../figures/figs_results/single_policy/SSP2-26-Availability-midTech/fleet_SSP2-26-Availability-midTech.png)
:::

:::{tab-item} Fleet, upper tech
![Fleet composition, Breakthrough availability, upper technology](../figures/figs_results/single_policy/SSP2-26-Availability-upTech/fleet_SSP2-26-Availability-upTech.png)
:::

:::{tab-item} Energy, low tech
![Energy-mix, Breakthrough availability, low technology](../figures/figs_results/single_policy/SSP2-26-Availability-lowTech/energy_SSP2-26-Availability-lowTech.png)
:::

:::{tab-item} Energy, mid tech
![Energy-mix, Breakthrough availability, mid technology](../figures/figs_results/single_policy/SSP2-26-Availability-midTech/energy_SSP2-26-Availability-midTech.png)
:::

:::{tab-item} Energy, upper tech
![Energy-mix, Breakthrough availability, upper technology](../figures/figs_results/single_policy/SSP2-26-Availability-upTech/energy_SSP2-26-Availability-upTech.png)
:::

::::

**Breakthrough low-demand**

::::{tab-set}

:::{tab-item} Fleet, low tech
![Fleet composition, Breakthrough low-demand, low technology](../figures/figs_results/single_policy/SSP2-26-LowDemand-lowTech/fleet_SSP2-26-LowDemand-lowTech.png)
:::

:::{tab-item} Fleet, mid tech
![Fleet composition, Breakthrough low-demand, mid technology](../figures/figs_results/single_policy/SSP2-26-LowDemand-midTech/fleet_SSP2-26-LowDemand-midTech.png)
:::

:::{tab-item} Fleet, upper tech
![Fleet composition, Breakthrough low-demand, upper technology](../figures/figs_results/single_policy/SSP2-26-LowDemand-upTech/fleet_SSP2-26-LowDemand-upTech.png)
:::

:::{tab-item} Energy, low tech
![Energy-mix, Breakthrough low-demand, low technology](../figures/figs_results/single_policy/SSP2-26-LowDemand-lowTech/energy_SSP2-26-LowDemand-lowTech.png)
:::

:::{tab-item} Energy, mid tech
![Energy-mix, Breakthrough low-demand, mid technology](../figures/figs_results/single_policy/SSP2-26-LowDemand-midTech/energy_SSP2-26-LowDemand-midTech.png)
:::

:::{tab-item} Energy, upper tech
![Energy-mix, Breakthrough low-demand, upper technology](../figures/figs_results/single_policy/SSP2-26-LowDemand-upTech/energy_SSP2-26-LowDemand-upTech.png)
:::

::::

The Breakthrough optimizations and comparison figures are produced by these scripts:

```{eval-rst}
.. minigallery:: examples/optimization/breakthrough/*.py
```
