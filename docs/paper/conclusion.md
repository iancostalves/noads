<!--
 Copyright 2025 ISAE-SUPAERO, https://www.isae-supaero.fr/en/
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

(sec-conclus)=

# Conclusion

Overall results show that, under trend demand growth: baseline scenarios display a
peak in emissions between 2035 and 2040, mitigation scenarios based solely on SAF has
limited emissions reductions due to energy availability constraints (2070 emissions
are still 75 % of 2019 with preferential availability), breakthrough aircraft
technologies can allow for reaching near zero emissions, but their impact is delayed
to after 2045 due to late Entry-Into-Service and slow fleet renewal.

The choice of which aircraft architecture and energy carrier to embark is highly
dependent on vehicle performance (determined by TLAR and technology assumptions) and
on the background energy system (due to the timing of electricity decarbonization and
to limited availability of electricity and biomass).

In order to respect Paris Agreement targets under an effort-sharing principle, drop-in
mitigation will put the system to a higher stress: either by constraining traffic, or
by consuming too much biomass and electricity. New aircraft with alternative energy
carriers allows to reduce such stress by making better use of the same energy
resources, even if their energy consumption is higher than that of conventional
planes.

When comparing mitigation policies with different objective functions, it was found
that supply caps, energy availability, and the introduction of alternative aircraft
designs are complementary rather than competing measures. One strategy alone can
achieve reduction in emissions up to a certain level, but their combined use is
capable of more efficient mitigation by: using alternative aircraft in its feasible
markets (reducing needs for low-carbon electricity and biomass for a given service),
and avoiding emission-intense markets (further reducing consumption of fossil
kerosene).

Regarding the numerical methods, the use of GEMSEO-JAX allowed for both reducing
implementation burden and execution time for the optimization of mitigation
scenarios. The simulation of these optimization-based policy scenarios was prohibitive
without the speedups obtained, which are of two orders of magnitude at the scenario
optimization level and three orders of magnitude at the scenario computation and
linearization level.

Our research offers practical insights on how to efficiently use optimization
algorithms for mitigation scenarios. Applying these methods for the aviation sector,
also shed light on how to optimally allocate aircraft architectures, energy resources,
and demand-side measures to achieve stringent mitigation targets.
