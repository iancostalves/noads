# Introduction

## Context

Aviation is often considered a hard-to-abate sector because carbon-free commercial aircraft are not readily available, and the production of alternative energy carriers on a global scale requires significant amounts of biomass and low-carbon electricity [@becken_implications_2023]. However, the sector will play a critical role in long-term climate mitigation of transportation [@girod_global_2012].

Several key energy carriers have emerged as potential substitutes for conventional kerosene:
- Synthetic kerosene (biomass-to-liquid and power-to-liquid pathways)
- Liquid hydrogen
- Ammonia
- Liquid natural gas
- Ethanol and methanol
- Batteries

Among these, only synthetic kerosene can be used in today's fleet without aircraft redesign. Current engines are limited by certification to 50% mixing with conventional kerosene, though advances toward 100% Sustainable Aviation Fuels (SAF) are ongoing.

### Aircraft Design Challenges

Compared to ground and marine transportation, aircraft are more weight-sensitive because they must generate lift to remain airborne. Increasing mass leads to:
- Increased induced drag
- Higher structural mass to support extra loads
- Further drag increase (snowball effect)

This means estimating energy consumption is a fundamentally coupled problem. Alternative aircraft designs may display:
- Higher energy consumption (batteries due to low specific energy)
- Limited payload and range (liquid hydrogen due to cryogenic tank weight)

## Research Positioning

> "Since the IPCC's Fifth Assessment Report (AR5) there has been a growing awareness of the need for demand management solutions combined with new technologies, such as the rapidly growing use of electromobility for land transport and the emerging options in advanced biofuels and hydrogen-based fuels for shipping and aviation." [@transport_ar6_wg3, Executive Summary]

Integrated Assessment Models (IAMs) are used to simulate the evolution of the coupled climate-energy-land system. Since the publication of the Shared Socioeconomic Pathways (SSP) [@riahi_shared_2017], improvements have been made in transportation mode detail within global mitigation. Aviation modeling in IAMs has shown:
- Demand-side measures may be important for keeping temperature increase between 1.5 and 2°C [@sharmina_decarbonising_2021]
- Synthetic kerosene and alternative aircraft (hydrogen, electric) can play significant roles in reaching stringent targets [@napp_role_2019; @speizer_integrated_2024]

However, these studies have limited detail concerning fleet composition and market segmentation. For example:
- Some consider electric and hydrogen aircraft to have similar energy consumption even for long-haul, where electric is unfeasible [@speizer_integrated_2024]
- Others apply singular efficiency values to entire fleets [@napp_role_2019]

Aircraft performance is highly dependent on range and technology. Hydrogen aircraft performance depends on cryogenic tank technology [@adler_hydrogen-powered_2023], leading to increased energy consumption [@icct_hydrogen] or limited operation [@icct_fuelcell]. For electric aircraft, battery weight significantly shortens maximum range and payload, making them only suited to regional flights [@icct_electric].

### Bridging IAMs and Technology Assessments

**Global IAMs lack technology and fleet detail** concerning aviation emissions, while **industry roadmaps rely on strong assumptions** of abundant renewable energy. This work bridges these approaches using **optimization to endogenously choose** appropriate timing and market penetration of conventional and alternative aircraft concepts.

This work builds on aviation climate mitigation scenario literature by:
- Linking sectoral mitigation with background SSP scenarios
- Using socioeconomic drivers to estimate future traffic volumes
- Limiting biomass and electricity consumption by fair share allocation
- Incorporating fleet and technology detail with 4 aircraft propulsion architectures
- Modeling component-level performances that mature over time

### Key Contributions

1. **Link sectoral mitigation with background SSP scenarios** from AR6 database
2. **Incorporate fleet and technology detail** with market segmentation
3. **Apply efficient numerical methods** for high-dimensional nonlinear optimization
4. **Demonstrate scenario-robust policy optimization** across multiple futures

### Numerical Methodology Contribution

Policy scenarios are nonlinear with many variables (disaggregated per market and technologies) and time-dependent (high-dimensional arrays). Even simpler optimization problems may:
- Fail to converge under gradient-free nonlinear optimizers
- Display significant linearization cost or implementation burden for gradient-based optimization

**This work reduces policy optimization time from 1.5 hours to a single minute** on a conventional laptop using automatic differentiation and just-in-time compilation.

### Novel Developments

Building on previous work [@costaalves-isabe], novel developments include:
- Incorporation of voluntary and price-based demand avoidance strategies
- Timing of introduction of new technologies
- Scenario-robust optimizations

### Important Note on Scope

The generated scenarios are **decarbonization scenarios rather than climate mitigation scenarios** because non-CO2 emissions are unaccounted for. While these effects can make up more than half of aviation's warming contribution (2000-2018) [@lee_contribution_2021], significant uncertainties remain in their estimation, especially for novel propulsion systems. Furthermore, debate continues on appropriate metrics for comparing short-lived effects to CO2 [@megill_alternative_2024].

## Organization

The documentation is organized as follows:
- **Methods**: Methodology, formulation, and numerical implementation
- **Model Documentation**: Detailed description of each model discipline
- **Main Results**: Principal results from proposed scenarios
- **Full Results**: Complete results similar to supplementary information
- **Discussion and Conclusion**: Critical view on results and key findings
