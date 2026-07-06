<!--
 Copyright 2025 ISAE-SUPAERO, https://www.isae-supaero.fr/en/
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

(sec-discus)=

# Discussion

Considering trend traffic growth and allocation of energy resources, even with
optimistic technological breakthrough, aviation share of emissions will grow as the
wider economy decarbonizes in a 2°C temperature increase scenario. This is in line
with scenarios from the IPCC's 6th Assessment Report {cite:p}`long_ar6_wg3` (Figure
3.19), in which transport emissions take the longest to reach net zero even in
scenarios below 1.5°C with little overshoot.

Regarding the implications for the energy system, fulfilling aviation's needs will
require significant volumes of low-carbon electricity and biomass. Limited production
amounts will require dedicated policy on which sectors to prioritize the access to
these resources. These findings are also in line with recent studies focused on the
aviation sector {cite:p}`becken_implications_2023,eaton_regional_2024,drunert_ptl`.
Dedicated production facilities integrated near airports might be a way to reducing
associated supply chain losses and costs {cite:p}`hoelzen_h2-powered_2025`.

As the efficiencies of energy conversion processes can vary strongly depending on the
primary energy source, their comparison must be made on a per unit of service basis.
While some studies on several transport modes {cite:p}`wallington_green_2024` consider
the base service unit as 1 MJ of thrust (or to wheels), this work instead, considers
it to be a traffic measure (passenger-kilometer). This distinction is fundamental for
aviation due to the different weights of the propulsion architectures to power these
vehicles.

Air transportation is highly unequal sector, both across (see the
{ref}`regional demand calibration <fig-regional-demand>`) and within
{cite:p}`GOSSLING2020102194` countries.
Given that most of the traffic growth in the next decades will happen in emerging
economies, which have not yet reached demand saturation. Many of these, however, are
still steering public policy to increase access to air travel, e.g.,
{cite:p}`voa-brasil`. The public acceptance and effectiveness of demand-side measures
can be questioned, and using pricing mechanisms to address this problem also raises
questions of social justice {cite:p}`Buchs02012024`.

Also, climate change is but one within the Planetary Boundaries
{cite:p}`rockstrom_safe_2009,richardson_earth_2023`. Some recent assessments that
expand the scope for other Planetary Boundaries {cite:p}`pais_current_2024` stress on
the need for acknowledging other limits to avoid shifting the problem, such as
biodiversity loss and eutrophication of freshwater ecosystems, especially when
considering a strong uptake of biofuels, which are present in most of the industry
decarbonization roadmaps {cite:p}`becken_implications_2023`.

Finally, given a set of technology forecasts, optimization can be useful for deciding
which technologies to prioritize and when, especially when they compete for similar
resources. Yet, in many IAM applications that use optimization, the formulation of the
optimization problem is often left unchanged or is little discussed, even if the
choice of policy goal (objective) and non-negotiables (constraints) are a fundamental
part of the process.
