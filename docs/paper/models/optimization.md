<!--
 Copyright 2025 ISAE-SUPAERO, https://www.isae-supaero.fr/en/
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

(sec-paper-optimization)=

# Optimization formulation

Optimal mitigation policies are formulated as the general nonlinear programming
problem. Two formulations of optimal mitigation scenarios are proposed: one relying on
minimizing cumulative emissions by **supplying for trend demand**, while changing
energy carriers and aircraft within the global fleet, here called trend formulation;
and other that **partially avoids trend traffic** and minimizes the relative ticket
price increase in order to align to a target cumulative emissions, here called
low-demand formulation.

The optimization variables consist of aircraft market penetration parameters (per
aircraft and per market), energy mix parameters (shares of energy production per
pathway per period), and, for the low-demand formulation, the supply cap parameters
(share of avoided demand per market per period). Entry-Into-Service is bounded between
2035 and 2060 for all aircraft types, except for the first generation of conventional
aircraft (which can be launched and deployed as soon as 2030), the energy mix
variables between 0 and 1, and $SR$ between 0 and 0.9 (at least 10 % of the trend
demand is satisfied at each market).

The optimization constraints applied to all scenarios consist in limiting the sum of
the control inputs among aircraft types for each market to 1, limiting the share of
pathway production to positive values among each produced energy, and limiting the
consumption of biomass and electricity to a fixed fair share of the global production.
Also, in the case where scenarios add electric aircraft to the fleet, an extra
constraint on aircraft energy consumption is added to ensure designs are feasible by
EIS.

In the trend formulation the objective function is the cumulative CO2 emissions. In
the low-demand formulation, cumulative emissions are constrained, and the objective is
the time-discounted mean burden of demand aversion ({eq}`eq-burden-avoidance`).

Numerical optimization is performed with the SLSQP Algorithm
{cite:p}`kraft1988software`, using forward-mode AD to obtain the gradients of
objective and constraints. Forward-mode is chosen over reverse-mode, or
backpropagation, due to the larger memory footprint when multiple constrains are
involved (which is the case here) {cite:p}`blondel2024`.

## Robustness to background scenario

The robustness to the background scenario was also performed as a policy optimization
problem. First a formulation must be chosen (results demonstrate the case for the
trend formulation) and is performed in a multi-scenario simulation. The objective of
the optimization problem is then considered as the mean of policy objectives under the
scenario ensemble.

The optimization variables are then divided into two sets: one that is kept fixed
among all scenarios (the aircraft mix variables), and one that may be specific to each
scenario (the energy mix variables). This allows to ensure the robustness of the
choice of aircraft mix while allowing for each scenario to individually optimize the
energy production system in order to respect their respective availability
constraints.
