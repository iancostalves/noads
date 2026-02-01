# Fleet Model

## Overview

The fleet model handles fleet composition, aircraft deployment, and gradual replacement dynamics across market segments.

## Model Structure

### Fleet Assembly

Fleet assembled from:
- **Current aircraft**: Existing fleet with calibrated performance
- **New aircraft**: Designed aircraft with evolving technology
- **Market segments**: 5 segments by design range (general, commuter, regional, short-medium, long-range)

### Replacement Dynamics

Fleet replacement follows first-order delay model:

$$\frac{dS_{aircraft}}{dt} = \frac{u(t) - S_{aircraft}(t)}{\tau_{fleet}}$$

where:
- $S_{aircraft}$ is aircraft share in market
- $u(t)$ is ramped input signal (from EIS to max share)
- $\tau_{fleet}$ is time constant based on market lifetime

### Aircraft Share Control

Input signal structure:
- Zero before Entry-Into-Service (EIS)
- Linear ramp from 0 to $S_{max}$ over duration $t_{ramp}$
- Constant at $S_{max}$ afterward

### Consumption Calculation

Energy consumed by aircraft operation:

$$C_{aircraft}(t) = ASK_{aircraft}(t) \times EC_{aircraft}$$

where:
- $ASK_{aircraft}$ is available seat-km from aircraft
- $EC_{aircraft}$ is design energy consumption per seat-km

Total carrier consumption:

$$C^{direct}_{energy} = \sum_{a \in energy\ architectures} C_a$$

## Assumptions

**Key Assumptions:**
- Fleet lifetime by market segment (15.3-33.8 years)
- Gradual replacement with first-order delay
- Aircraft designed for specific market ranges
- Each market has constant share of total supply

**Limitations:**
- No route-specific optimization
- Airlines may operate suboptimally
- Fleet assignment not modeled
- No operational constraints (gates, slots)

## Related Documentation

For architectural diagrams showing fleet assembly, see:
[User Guide - Fleet Models](../user_guide/index.md#models)
