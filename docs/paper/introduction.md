<!--
 Copyright 2025 ISAE-SUPAERO, https://www.isae-supaero.fr/en/
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

(sec-intro)=

# Introduction

Air transportation is often considered a hard-to-abate sector, because carbon-free
commercial aircraft are not readily available, and the production of alternative
energy carriers to fossil kerosene on a global scale requires significant amounts of
biomass and low-carbon electricity to meet the growing demand
{cite:p}`becken_implications_2023`. However, the sector will play a critical role in
long-term climate mitigation of transportation {cite:p}`girod_global_2012`.

Several key energy carriers have emerged as potential substitutes of conventional
kerosene, such as synthetic kerosene (from biomass-to-liquid or power-to-liquid
pathways), liquid hydrogen, ammonia, liquid natural gas, ethanol, methanol, and
batteries {cite:p}`ansell_review_2023`. Among these, only synthetic kerosene can be
used in today's fleet without aircraft re-deisgn. Current engines are limited by
certification to a 50 % mixing ratio with conventional kerosene, but there is ongoing
advances on the operation with 100 % Sustainable Aviation Fuels (SAF)
{cite:p}`full-saf`. There is, however, among industry roadmaps a systematic
over-reliance on SAF deployment, which could require 9 % of global renewable
electricity and 30 % of sustainably available biomass by 2050
{cite:p}`becken_implications_2023`.

Compared to ground and marine transportation vehicles, aircraft are more weight
sensitive because they must generate lift in order to remain airborne. Increasing mass
leads to an increase in induced drag, which further increases the structural mass to
support extra loads, leading to more drag, and so on. This snowball effect is a
central phenomenon in the conception of aircrafts {cite:p}`raymer_aircraft_2018`,
which means that estimating energy consumption is a fundamentally coupled problem.
Alternative aircraft designs, flying with different energy carriers, may display
higher energy consumption and/or limited payload and ranges compared to conventional
jet-fueled planes, due to low specific energy (batteries) or due to the need for
heavier cryogenic fuel tanks (liquid hydrogen).

> "Since the IPCC's Fifth Assessment Report (AR5) there has been a growing awareness
> of the need for demand management solutions combined with new technologies, such as
> the rapidly growing use of electromobility for land transport and the emerging
> options in advanced biofuels and hydrogen-based fuels for shipping and aviation."
> — {cite:p}`transport_ar6_wg3` (Executive Summary)

Integrated Assessment Models (IAM) are the tools used to simulate the evolution of the
coupled climate-energy-land system from socioeconomic assumptions on climate
mitigation and adaptation. Since the publication of the Shared Socioeconomic Pathways
(SSP) {cite:p}`riahi_shared_2017`, great improvements have been made in the detail
given to transportation modes within global mitigation. The modeling of aviation
within IAM's, has shown how demand-side measures may play an important role in keeping
temperature increase between 1.5 and 2° C above preindustrial levels
{cite:p}`sharmina_decarbonising_2021`, but also how the incorporation of synthetic
kerosene and the deployment of new aircraft with alternative energy carriers (such as
hydrogen and biofuels) can also play a significant role in reaching stringent targets
{cite:p}`napp_role_2019,speizer_integrated_2024`.

However, the detail these studies have concerning the fleet composition and how they
are arranged within market segments is limited, e.g.,
{cite:p}`speizer_integrated_2024` considers kerosene, electric, and hydrogen aircraft
to have similar energy consumption even for long-haul (where electric is unfeasible,
and hydrogen is less efficient), while {cite:p}`napp_role_2019` accounts for lower
efficiencies for hydrogen, it does so with a singular value applied for the entire
fleet. The performance of hydrogen aircraft is highly dependent on their range and the
cryogenic tank technology {cite:p}`adler_hydrogen-powered_2023`, which can lead to
increased energy consumption {cite:p}`icct_hydrogen` or limited operation in terms of
payload and ranges {cite:p}`icct_fuelcell`. For electric aircraft, this is even more
pronounced as the battery weight significantly shortens the maximum achievable range
and payload, making these aircraft only suited to regional flights
{cite:p}`icct_electric`.

Sectoral specific scenarios can account for this either with a detailed network of
Origin-Destination pairs
{cite:p}`grewe_evaluating_2021,eaton_regional_2024,hoelzen_h2-powered_2025`, or by
separating the traffic into market segments according to flown distance. A review of
sectoral aviation scenarios {cite:p}`delbecq_sustainable_2023` shows that
methodologies may differ regarding: traffic evolution, mitigation levers, inclusion of
non-CO2 effects, and resource consumption. Employing open-source tools, such as the
AeroMAPS framework {cite:p}`planes_aeromaps_2023`, to simulate these types of
scenarios can be greatly beneficial to explicit modeling assumptions and finding a
common ground for high-level decision making.

From a technological perspective, it is important to map the range of components that
are required for novel system architectures. In the context where many of such
components are still at immature technologies and require significant allocation of
resources to develop, quantifying upper and lower bounds associated to each component
is paramount to compare architectural choices in terms of system-level performances
{cite:p}`kirby_forecasting_1999` and in end-point metrics, such as financial returns
{cite:p}`de_weck_technology_2022`. Regarding climate mitigation, determining the best
"carbon-neutral" propulsion architecture choice for each flight is subject of recent
interest {cite:p}`adler_energy_2025`, but rely on the strong assumption of dedicated
renewable electricity to power aviation, which may be supply constrained.

## Research positioning

Global IAM's lack technology and fleet detail concerning aviation emissions, while
industry roadmaps and technology assessments rely on strong assumptions of abundant
renewable energy. We propose to bridge these together using optimization to
endogenously chose appropriate timing and market penetration of conventional and
alternative aircraft concepts, subject to a background scenario using the AR6 scenario
database {cite:p}`ar6_database`.

These policy scenarios are nonlinear, with many variables due to disaggregation (per
market and technologies), and time-dependent (high-dimensional arrays). In practice,
even the simpler policy optimization problems may: fail to converge under
gradient-free nonlinear optimizers, display significant online linearization cost or
offline implementation burden for gradient-based optimization. The numerical
methodology aspect of this paper is therefore a central contribution, reducing policy
optimization time from 1.5 hours to a single minute on a conventional laptop.

This work builds on the literature of aviation climate mitigation scenarios by linking
sectoral mitigation with a background system based on the Shared Socioeconomic
Pathways (SSP). Socioeconomic drivers are used to estimate future traffic volumes,
based on the assumption that historical trends on traffic growth from personal income
and population growth remain unchanged. Biomass and electricity consumed by scenarios
are limited by using the concept of a fair share of global production that is
allocated to the sector, which avoids over-consumption that can be detrimental to
other economic sectors. Fleet and technology detail are also incorporated by modeling
conceptual design and possible Entry-Into-Service (EIS) of 4 aircraft propulsion
architectures, each linked to component-level performances that mature over time.

This methodology was introduced in {cite:p}`costaalves-isabe`, which addressed the
exploration of technology-based decarbonization scenarios with optimization
algorithms. Novel developments include the incorporation of voluntary and price-based
demand avoidance strategies, the timing of introduction of new technologies, and
scenario-robust optimizations.

It is important to note that the generated scenarios are considered as
decarbonization scenarios rather than climate mitigation scenarios, because non-CO2
emissions are unaccounted for. While these effects can make up for more than half of
aviation's contribution to global warming in the 2000-2018 period
{cite:p}`lee_contribution_2021`, there are still significant uncertainties concerning
their estimation, especially when considering novel propulsion systems, for which
little data is available. Furthermore, because of the short-lived nature of these
warming effects, the debate on which metrics to use for comparing them to CO2 is still
ongoing {cite:p}`megill_alternative_2024`.

While the usage of optimization within IAM's is not new
{cite:p}`nordhaus_optimal_1992,barrage_policies_2023,gcam-2019,witness`, there are
several limitations with the tools used to solve them, which often imply a
simplification of the problem: reducing time resolution, linearization of the problem
{cite:p}`huppmann_messageix_2019`, or even simplification of two-way couplings
{cite:p}`jgcricassandra_2024`. Multidisciplinary Optimization (MDO) frameworks can
offer methods that allow for: formalizing the coupled problem within optimization
routines, and using gradient-based algorithms, which allows nonlinear optimization to
scale well despite increasing number of variables. On this methodological front, the
present work also contributes to the formulation of optimal mitigation policies, and
to alleviate the main drawback of using IAM's with MDO: the extra implementation
burden, which is dealt with by using libraries with automatic differentiation
capabilities.

The paper is organized as follows: first the quantitative models used are briefly
described, along with the methodology and implementation behind optimal scenario
formulation; then the principal results from proposed scenarios are presented,
analyzed, and compared with current aviation scenarios based on IAMs; the shortcomings
of models are made explicit in the limitations section; a critical view on the results
and implications from the modeled future are addressed in the discussion; finally, the
conclusion summarizes the contributions and key findings of the study.
