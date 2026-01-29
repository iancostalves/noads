# Model Equations

This page provides the mathematical formulation of the models used in aviation decarbonization scenarios.

## Nomenclature

### Variables

| Symbol | Description |
|--------|-------------|
| $o$ | Delay output variable |
| $i$ | Delay input variable |
| $\tau$ | Delay-time (time constant) |
| $D$ | Aggregate demand |
| $Pop$ | Population |
| $I$ | Per-capita income |
| $p$ | Ticket price |
| $\epsilon_{\upsilon}$ | Elasticity of demand to variable $\upsilon$ |
| $\sigma$ | Calibration constant |
| $RPKpc_{trend}$ | Trend per-capita Revenue Passenger Kilometers |
| $RPK$ | Revenue Passenger Kilometers |
| $ASK$ | Available Seat Kilometers |
| $LF$ | Load Factor |
| $EIS$ | Entry-Into-Service year |
| $SR$ | Supply Reduction ratio (demand avoidance) |
| $S$ | Share (aircraft, pathway, market) |
| $EC$ | Energy Consumption |
| $IF$ | Impact Factor (emissions per unit energy) |
| $CF$ | Consumption Factor (input per unit output) |
| $P$ | Production |
| $C$ | Consumption |

## First-Order Delay

Time-dependent controls are subject to a first-order delay representing policy implementation inertia:

$$\frac{do}{dt} = \frac{i - o}{\tau}$$

where:
- $i$ is the input (control setpoint)
- $o$ is the output (actual state)
- $\tau$ is the time constant (delay time)

Solution for step input:
$$o(t) = i \left(1 - e^{-t/\tau}\right)$$

## Demand Models

### Logistic Demand Function

Per-capita demand as a function of per-capita income:

$$RPKpc_{trend}(t) = L + (R-L) \left( \frac{C}{C+\exp(-B(I(t)-\iota))} \right)^{1/\nu}$$

Parameters:
- $L$: Left asymptote (propensity at zero income)
- $R$: Right asymptote (propensity at infinite income)
- $\iota$: Income at inflection point
- $B$: Logistic growth rate
- $C$: Transition duration parameter
- $\nu$: Asymptote preference parameter

### Storyline Multiplier

SSP-specific adjustment to trend demand:

$$\frac{RPKpc_{story}(t)}{RPKpc_{trend}(t)} = 1 + \frac{F_{SSP}}{1+\exp\left(- \frac{I(t)-2I_{2024}}{2I_{2024}}\right)}$$

where $F_{SSP}$ is the storyline factor (e.g., -0.1 for SSP1, 0 for SSP2, 0.5 for SSP5).

### Total Demand

$$RPK_{trend}(t) = Pop(t) \times RPKpc_{story}(t)$$

### Load Factor

Converts passenger-km to seat-km:

$$LF(t) = \frac{RPK_{trend}(t)}{ASK_{trend}(t)}$$

Grows quadratically from 82.4% (2019) to 92% (2075).

## Demand Avoidance

### Segmented Supply

$$ASK(t) = \sum_{m \in markets} ASK_m(t) = \sum_{m \in markets} S_m \times ASK_{trend}(t) \times (1 - SR_m(t))$$

where:
- $S_m$ is the constant market share
- $SR_m(t)$ is the supply reduction ratio (0 = no avoidance, 1 = complete ban)

### Price Elasticity

From definition $\epsilon_p = \frac{dRPK/RPK}{dp/p}$, integrating yields:

$$\ln\left(\frac{p_{cap}}{p_{trend}}\right) = \frac{1}{\epsilon_p} \ln\left(\frac{D_{cap}}{D_{trend}}\right)$$

Therefore:
$$\frac{p_{cap}}{p_{trend}} = \left(\frac{D_{cap}}{D_{trend}}\right)^{1/\epsilon_p} = (1-SR)^{1/\epsilon_p}$$

### Annual Burden

Relative price increase due to demand avoidance:

$$\theta_{avoidance}(t) = \frac{\Delta P}{P} = \frac{\sum_{m} S_m (1-SR_m(t))^{1/\epsilon_p}}{ASK(t)}$$

### Present Value of Burden

Objective for low-demand formulation:

$$\Theta_{avoidance} = \int_{t_0}^{t_1} (1+d_f)^{t_0-t} \theta_{avoidance}(t) \, dt$$

where $d_f$ is the social discount rate (3%).

## Fleet Models

### Aircraft Share Control

Share of new aircraft in market follows delayed ramp response:

Input signal:
$$u(t) = \begin{cases}
0 & t < EIS \\
\frac{S_{max}}{t_{ramp}}(t - EIS) & EIS \leq t < EIS + t_{ramp} \\
S_{max} & t \geq EIS + t_{ramp}
\end{cases}$$

Actual share $S_{aircraft}(t)$ follows first-order delay with time constant $\tau_{fleet}$.

### Aircraft Consumption

Energy consumed by aircraft operation:

$$C_{aircraft}(t) = ASK_{aircraft}(t) \times EC_{aircraft}$$

where $EC_{aircraft}$ is the design energy consumption per seat-km.

### Carrier Consumption Aggregation

Total direct consumption of each energy carrier:

$$C^{direct}_{energy} = \sum_{a \in energy\ architectures} C_a$$

## Energy Mix Models

### Impact Factor of Production Pathway

Impact per unit production (e.g., g CO2 / MJ):

$$IF_{pathway} = IF^{direct}_{pathway} + \sum_{i \in pathway\ inputs} CF_{pathway,i} \times IF_i$$

where:
- $IF^{direct}$ is the direct impact of the process
- $CF_{pathway,i}$ is consumption of input $i$ per unit output
- $IF_i$ is the impact factor of input $i$

### Impact Factor of Mixed Energy

Weighted average by pathway shares:

$$IF_{energy} = \sum_{p \in energy\ pathways} S_p \times IF_p$$

### Energy Production

Production of each energy type:

$$P_{energy} = C^{direct}_{energy} + \sum_{o \in energy\ outputs} CF_{o,energy} \times P_o$$

Starts with final energies (where second term is zero) and works backward to primary energies.

### Pathway Production

$$P_{pathway} = S_{pathway} \times P_{energy}$$

### Input Consumption

$$C_{pathway,input} = CF_{pathway,input} \times P_{pathway}$$

### Total Input Consumption

$$C_{energy} = \sum_{p \in energy\ pathways} C_{p,energy}$$

## Constraint Equations

### Fleet Retirement

Current fleet retirement must be non-negative:

$$\text{Retired}_{current,market}(t) \geq 0$$

### Energy Pathway Shares

Shares must sum to 1:

$$\sum_{p \in pathways} S_p = 1$$

### Resource Fair Share

Total consumption bounded by fair share:

$$\int_{t_0}^{t_1} C_{biomass}(t) \, dt \leq \text{FairShare}_{biomass}$$

$$\int_{t_0}^{t_1} C_{electricity}(t) \, dt \leq \text{FairShare}_{electricity}$$

### Carbon Budget (Low-Demand)

Total emissions bounded by budget:

$$\int_{t_0}^{t_1} CO_2(t) \, dt \leq \text{Budget}$$

### Aircraft Feasibility

New aircraft design must be feasible at EIS:

$$\text{MTOW}_{aircraft}(EIS) < \text{MTOW}_{max}$$

where MTOW is Maximum Take-Off Weight.

## Objective Functions

### Trend Formulation

Minimize cumulative emissions:

$$\min \int_{2020}^{2070} CO_2(t) \, dt$$

### Low-Demand Formulation

Minimize present value of demand avoidance burden:

$$\min \int_{2020}^{2070} (1+d_f)^{2020-t} \theta_{avoidance}(t) \, dt$$

### Robust Formulation

Minimize mean across scenarios:

$$\min \frac{1}{N} \sum_{s=1}^{N} \int_{2020}^{2070} CO_2^{(s)}(t) \, dt$$

where $N$ is the number of background scenarios.

## Total Emissions

### Well-to-Wake Emissions

$$CO_2(t) = \sum_{e \in energy\ carriers} C_e(t) \times IF_e(t)$$

where:
- $C_e(t)$ is consumption of carrier $e$ at time $t$
- $IF_e(t)$ is the carbon intensity (g CO2/MJ) of carrier $e$ at time $t$

### Cumulative Emissions

$$\text{Cumulative}_{CO_2} = \int_{2020}^{2070} CO_2(t) \, dt$$

## Energy and Carbon Intensity

### Energy Intensity

Energy per passenger-km:

$$\text{Energy Intensity}(t) = \frac{\sum_e C_e(t)}{RPK(t)}$$

### Carbon Intensity

Emissions per passenger-km:

$$\text{Carbon Intensity}(t) = \frac{CO_2(t)}{RPK(t)}$$

## References

For detailed derivations and additional equations, see:
- Supplementary Information, Section 1 (Models)
- Main paper, Section 2 (Methods)
