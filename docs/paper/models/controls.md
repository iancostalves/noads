<!--
 Copyright 2025 ISAE-SUPAERO, https://www.isae-supaero.fr/en/
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

(sec-paper-controls)=

# Time-dependent controls

First-order delays are commonly used to account for simplified inertia regarding:
measuring and reporting information, receiving information and decisions being made,
and even for decisions to have a visible effect on the state of a system
{cite:p}`meadows_thinking_2009,sterman_business_2009`. While the present work
automates decision-making based on system outcomes, we still account for delays
regarding the time-dependent optimization variables (here called controls).
Furthermore, this also allowed for numerical benefits, such as reduced dimensionality
and improved the optimization stability.

```{math}
:label: eq-delay

\frac{d}{dt}o(t) = \frac{i(t)-o(t)}{\tau}
```

Each output variable $o(t)$ is modeled as an Ordinary Differential Equation
({eq}`eq-delay`), which is dependent on a given input variable $i(t)$ and a delay-time
$\tau$. {numref}`fig-delay` shows the output response of the first-order delay system
for a set of inputs. For ramped step input (with ramp duration of $2\tau$) it takes
$4\tau$ for the output to reach 95 % of the step value, considered as one replacement
lifetime.

```{figure} ../figures/figs_models/first-order-delay.png
:name: fig-delay
:width: 100%

Response of the first order delay to a set of input paths (step, ramped step, and
pulse), for two time constants, subject to initial value of 0.
```

The supply shift ratio ({eq}`eq-segmented-avoidance`) and the shares of pathway
production ({eq}`eq-impact-energy` and {eq}`eq-prod-pathway`) are modeled with inputs
that are coarsely discretized in time (2.5 years, while the simulation step is 1
year), here the input values at each time are optimization variables. The shares of
aircraft market penetration are modeled with a ramped pulse, parameterized from 4
optimization variables (ramp start year, max value, lifetime, and ramp-down duration).

## Reproduce these figures

The delay response and how each kind of control (aircraft market share, energy pathway
shares, and demand avoidance) maps the optimization variables to smooth simulation
inputs are produced by the following example:

```{eval-rst}
.. minigallery:: examples/models/controls/plot_*.py
```
