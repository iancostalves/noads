# Model Assumptions

This page details the main assumptions for each model discipline, along with their limitations.

## Summary Table

The following table summarizes the main assumptions and limitations for each model discipline:

| Model Discipline | Assumptions | Limitations |
|-----------------|-------------|-------------|
| **Policy controls** | Policy variables act on fleet replacement (per aircraft per market) and energy production shares (per pathway). The low-demand formulation also includes a cap in supply (per market). Controls evolve gradually in time subject to a linear delay. | Time-delayed share-based control creates instability under rapid production volume changes. |
| **Air traffic demand** | Future trend Revenue Passenger Kilometers (RPK) demand estimated based on scenario-dependent population and income per capita. Demand grows with income, but is assumed to saturate at around 2700-4300 pax-km/capita (scenario-dependent) and represented via sigmoidal (logistic-type) functions calibrated on historical data. The supply, in terms of Available Seat Kilometers (ASK), is then estimated using a quadratic load factor that grows from 84.4% (2019) to 92% (2075). | Prospective analysis uses logistic functions outside range of calibrated data, problematic for estimating saturation levels of developing countries. Global aggregation ignores regional disparities. Price elasticity not coupled with energy costs. |
| **Current aircraft** | The initial fleet composition and performance are fixed and calibrated to historical data. Fleet parameters (efficiency, lifetime) are exogenous and differentiated by market segments, but may vary due to technology assumptions. | Potential mismatch between current efficiency (aggregated by flight distance) and aircraft lifetimes (aggregated by number of seats). |
| **New aircraft** | Generic Airplane Model (GAM) for conceptual design of future aircraft with conventional and alternative propulsion architectures: (i) conventional kerosene turbofan, (ii) liquid-hydrogen turbofan, (iii) liquid-hydrogen fuel-cell and electric motors, and (iv) battery-electric aircraft. Aircraft are designed for market-specific missions, and deployed by Entry-Into-Service (EIS) year, which are also optimization variables (per market per aircraft). Technology maturing is incorporated with assumptions on component-level performances (structures, batteries, cryogenic tanks, fuel cell, ...) that vary in time and according to technology assumptions, reducing energy consumption at later EIS. | TLAR fixed within markets (size, speed, altitude not optimized). Cryogenic tank efficiency fixed across markets, ignoring size dependency. Gas turbine weight and consumption assumed identical for kerosene and hydrogen. |
| **Fleet replacement** | Current aircraft is replaced by new aircraft types, designed specifically for market range. Replacement is gradual with market-specific lifetimes of 15.3-33.8 years (variable with technology assumptions). | Airlines may operate aircraft at shorter distances than designed mission, increasing fuel consumption beyond modeled values. |
| **Energy mix** | Aviation energy demand may be supplied by a mix of different energy carriers (Jet-A fuel, Batteries, Liquid-hydrogen), each of them being produced by different production pathways (fossil kerosene, biofuels, e-fuels, electrolysis, ...). Well-to-wake (WTW) lifecycle emissions are considered for all carriers, with constant in time carbon intensities for kerosene and biofuels, while electricity-based pathways have emission intensities calculated based on process efficiencies and scenario-dependent grid emission intensity. | No differentiation between biomass types (all pathways compete for same resource). PtL consumption assumes only Direct Air Capture as source for carbon, concentrated CO2 sources are not considered. |
| **Constraints** | Feasibility is evaluated with: initialization constraints (each aircraft must be feasible by EIS), and path-wise constraints (current fleet retirement per market, pathway-mix per carrier, fair share of biomass and electricity consumption). The low-demand formulation also includes an end-time constraint (cumulative emissions). | Fair shares of global production allocated to aviation estimated with grandfathering approaches based on energy consumption. |
| **Objectives** | Minimize cumulative emissions (trend formulation), or minimize discounted relative price increase due to supply-cap (low demand formulation). | Simplified modeling of relative price increase with constant price elasticities among markets. |

## Detailed Assumptions by Discipline

### Policy Controls

Policy variables control:
- Fleet replacement rates per aircraft type per market segment
- Energy production shares per pathway
- Demand caps per market (in low-demand formulation only)

**Control Dynamics**: Controls evolve gradually in time subject to a first-order linear delay. This represents the inertia in policy implementation and fleet replacement.

**Delay Equation**:
$$\frac{dx}{dt} = \frac{u - x}{\tau}$$

where $u$ is the control input, $x$ is the actual state, and $\tau$ is the time constant.

### Air Traffic Demand

**Trend Demand**: Based on logistic function of per-capita income:

$$RPKpc_{trend}(t) = L + (R-L) \left( \frac{C}{C+\exp(-B(I(t)-\iota))} \right)^{1/\nu}$$

where:
- $L$ and $R$ are left and right asymptotes (propensity to travel at $0$ and $\infty$ income)
- $\iota$ is income per capita at inflection point
- $B$ is logistic growth rate
- $C$ controls transition duration
- $\nu$ controls near which asymptote maximum growth occurs

**Storyline Factor**: SSP-specific multiplier adjusts trend demand:

$$\frac{RPKpc_{story}(t)}{RPKpc_{trend}(t)} = 1 + \frac{F_{SSP}}{1+\exp\left(- \frac{I(t)-2I_{2024}}{2I_{2024}}\right)}$$

where $F_{SSP}$ is storyline-dependent (e.g., 0.9 for SSP1, 1.0 for SSP2, 1.5 for SSP5).

**Total Demand**:
$$RPK_{trend}(t) = Pop(t) \times RPKpc_{story}(t)$$

**Load Factor**: Converts RPK to ASK (Available Seat Kilometers):
$$LF(t) = RPK_{trend}(t) / ASK_{trend}(t)$$

Load factor grows quadratically from 82.4% (2019) to 92% (2075).

### Current Aircraft Fleet

**Market Segmentation**: Fleet is divided into 5 market segments based on design range:

| Market | Range (km) | Seats | Consumption (MJ/seat-km) | Lifetime (years) |
|--------|-----------|-------|-------------------------|------------------|
| General | 500 | 19 | 2.74-1.47 | 33.8-22.2 |
| Commuter | 1500 | 50 | 1.25-0.88 | 26.5-21.9 |
| Regional | 4500 | 80 | 0.87-0.73 | 20.5-15.3 |
| Short-medium | 8000 | 120 | 1.00-0.82 | 33.8-24.8 |
| Long range | 15000 | 250 | 1.03-0.83 | 29.5-23.6 |

**Technology Scenarios**: Three scenarios (Lower, Mid, Upper) affect:
- Current aircraft efficiency (lower quartile, median, upper quartile)
- Fleet lifetimes (upper quartile, intermediate, median)

### New Aircraft Design

**Propulsion Architectures**:
1. Conventional kerosene turbofan
2. Liquid-hydrogen turbofan
3. Liquid-hydrogen fuel-cell with electric motors
4. Battery-electric

**Top-Level Aircraft Requirements (TLAR)**:

*By Architecture*:

| Architecture | Speed (Mach) | Altitude (1000 ft) |
|--------------|--------------|-------------------|
| Jet-A Gas Turbine | 0.75 | 27 |
| LH2 Gas Turbine | 0.75 | 27 |
| Battery E-motor | 0.5 | 20 |
| LH2 Fuel-cell E-motor | 0.5 | 20 |

*By Market*: Range and seats determined by market segment (see table above).

**Technology Maturing**: Component-level parameters evolve over time:

| Technology Parameter | Unit | 2020 | 2040 | 2060 |
|---------------------|------|------|------|------|
| Battery Specific Energy | Wh/kg | 200 | 350-800 | 600-1500 |
| E-motor Specific Power | kW/kg | 2 | 10-25 | 15-28 |
| Electronics Specific Power | kW/kg | 2 | 15-25 | 20-32 |
| Fuel cell Specific Power | kW/kg | 1 | 2-3 | 3-6 |
| Fuel cell Efficiency | % | 40 | 45-55 | 50-65 |
| LH2 tank Gravimetric Index | % | 20 | 30-65 | 35-80 |
| Structural weight reduction | % | 0 | 15-30 | 20-40 |

### Energy Mix

**Energy Carriers**:
- Fossil Jet-A (kerosene)
- Biofuels (HEFA, ATJ, FT)
- Electrofuels (Power-to-Liquid)
- Gaseous hydrogen
- Liquid hydrogen
- Electricity (for electric aircraft)

**Production Pathways**:

| Energy Carrier | Pathway | Energy Consumption (MJ/MJ) | Direct Emissions (g CO2/MJ) |
|---------------|---------|---------------------------|---------------------------|
| Fossil Jet-A | Refinery | 1.16 oil | 88.7 |
| Biofuel | HEFA | 1.95 biomass | 62.73 |
| Biofuel | ATJ | 3.33 biomass | 51.55 |
| Biofuel | FT | 5.0 biomass | 35.3 |
| Gas H2 | Electrolysis | 1.41-1.33 electricity | 0 |
| Electrofuel | Power-to-liquid | 0.65-0.56 elec, 1.89-1.68 H2 | 0 |
| Liquid H2 | Liquefaction | 0.22-0.16 elec, 1.0 H2 | 0 |

**Well-to-Wake Emissions**: Lifecycle emissions include:
- Direct emissions from production process
- Indirect emissions from consumed inputs
- Scenario-dependent electricity grid carbon intensity

### Constraints

**Initialization Constraints**: Each aircraft design must be feasible at its Entry-Into-Service date.

**Path Constraints**:
- Current fleet retirement must be non-negative (cannot retire more than available)
- Energy pathway shares must sum to 1 for each carrier
- Total biomass consumption ≤ fair share of global production
- Total electricity consumption ≤ fair share of global production

**End-Time Constraints** (low-demand only):
- Cumulative CO2 emissions ≤ carbon budget

**Fair Share Allocation**: Aviation's share of global energy production calculated using grandfathering approach based on 2019 consumption shares.

### Objectives

**Trend Formulation**: Minimize cumulative CO2 emissions from 2020 to 2070.

$$\min \int_{2020}^{2070} CO_2(t) \, dt$$

**Low-Demand Formulation**: Minimize present value of demand management burden:

$$\min \int_{2020}^{2070} (1+d_f)^{2020-t} \theta_{avoidance}(t) \, dt$$

where:
- $d_f$ = 3% social discount rate
- $\theta_{avoidance}$ = relative ticket price increase

**Relative Price Increase**: From price elasticity:

$$\theta_{avoidance}(t) = \frac{\sum_m S_m (1-SR_m(t))^{1/\epsilon_p}}{ASK(t)}$$

where $SR_m$ is supply reduction ratio for market $m$ and $\epsilon_p$ is price elasticity.

## Key Simplifications and Limitations

1. **Non-CO2 effects**: Not included (contrails, NOx, etc.)
2. **Network effects**: Global aggregation ignores route-specific factors
3. **Economic detail**: Simplified cost modeling
4. **Regional variation**: Single global model
5. **Technology uncertainty**: Fixed maturation pathways
6. **Behavioral responses**: Constant elasticities

## Model Verification

Models have been verified through:
- Comparison with historical data (1980-2019)
- Cross-validation with other aviation models (AeroMAPS)
- Sensitivity analysis on key parameters
- Expert review of assumptions
