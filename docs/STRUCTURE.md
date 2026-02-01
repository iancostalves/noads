# Documentation Structure

This document describes the reorganized documentation structure for the NOADS project.

## Overview

The documentation has been reorganized to follow the structure of a scientific paper, making it easier to navigate and understand the research.

## New Structure

### 1. Introduction (`docs/introduction/`)
- **Purpose**: Provide context and research positioning
- **Content**: 
  - Aviation as hard-to-abate sector
  - Energy carrier options and aircraft design challenges
  - Research positioning: bridging IAMs and technology assessments
  - Key contributions and novel developments
  - Numerical methodology importance
- **Source**: Reused from paper introduction and existing documentation

### 2. Methods (`docs/methods/`)
- **Purpose**: Explain methodology and numerical approach
- **Content**:
  - `index.md`: Overview of optimization process and model disciplines
  - `formulation.md`: Detailed optimization formulations, numerical methods, computational gains, scenario definitions
- **Source**: Reused from paper methods section and existing documentation

### 3. Model Documentation (`docs/models/`)
- **Purpose**: Detailed documentation for each model discipline
- **Content**:
  - `index.md`: Overview and general assumptions
  - `demand.md`: Demand model with logistic functions and calibration
  - `aircraft.md`: Current fleet and prospective aircraft design
  - `fleet.md`: Fleet composition and replacement dynamics
  - `energy.md`: Energy mix and production pathways
- **Source**: Reused from Supplementary Information and existing model documentation
- **Links**: References to interactive examples in `docs/examples/models/`

### 4. Results (`docs/results/`)
- **Purpose**: Present scenario results
- **Content**:
  - `index.md`: Results overview with links
  - `main_results.md`: Detailed analysis of key findings (baseline, drop-in, breakthrough, robust)
  - `full_results.md`: Comprehensive results for all scenarios (similar to SI)
- **Source**: Main results reused from existing content; full results includes new descriptive text with links to all PDFs

### 5. Discussion and Conclusion (`docs/discussion/`)
- **Purpose**: Interpret results and provide conclusions
- **Content**:
  - Key implications for policy and technology
  - Equity and demand management
  - Beyond climate (planetary boundaries)
  - Role of optimization
  - Model limitations
  - Conclusion section
- **Source**: Reused from paper discussion and conclusion sections

### 6. Reference Sections
- **User Guide** (`docs/user_guide/`): Technical reference documentation
- **Examples** (`docs/examples/`): Interactive examples and result files
- **API Documentation**: Generated from code

## Navigation Flow

The documentation follows a logical progression:

```
Introduction → Methods → Model Documentation → Results → Discussion & Conclusion
     ↓            ↓              ↓                  ↓              ↓
  Context    Formulation    Per-model         Main & Full    Implications
  Research    Numerical      Details          Scenarios      Conclusions
  Position    Methods        Calibration
```

## Content Principles

1. **No New Text**: Only reuse or summarize existing content from paper and current docs (except Full Results descriptions)
2. **Paper-Based**: Structure follows the paper organization
3. **Modular**: Each model has separate documentation
4. **Complete**: Full results link to all available scenario PDFs
5. **Referenced**: Links to examples and interactive scripts

## Key Features

### Clear Hierarchy
- Top-level sections match paper structure
- Each section has clear purpose
- Easy to find specific information

### Model Separation
- Each model discipline has dedicated page
- Includes assumptions, limitations, equations
- Links to calibration data and examples

### Comprehensive Results
- Main results: High-level findings and interpretation
- Full results: Complete scenario catalog with links to all PDFs
- Clear organization by scenario type

### Interactive Elements
- Links to Jupyter notebooks in examples
- References to plotting scripts
- Direct links to result PDFs

## File Organization

```
docs/
├── introduction/
│   └── index.md                    # Research context and positioning
├── methods/
│   ├── index.md                    # Methods overview
│   └── formulation.md              # Detailed formulation and numerical methods
├── models/
│   ├── index.md                    # Model overview
│   ├── demand.md                   # Demand model
│   ├── aircraft.md                 # Aircraft models
│   ├── fleet.md                    # Fleet model
│   └── energy.md                   # Energy mix model
├── results/
│   ├── index.md                    # Results overview
│   ├── main_results.md             # Main findings
│   └── full_results.md             # Complete results catalog
├── discussion/
│   └── index.md                    # Discussion and conclusion
├── user_guide/                     # Technical reference (kept)
├── examples/                       # Interactive examples (kept)
│   ├── models/                     # Model examples with plots
│   └── optimization/               # Result PDFs
└── paper/                          # LaTeX source files (kept)
```

## Removed Content

The following directories were removed as their content was consolidated:
- `docs/user_guide_extended/` - Integrated into Methods
- `docs/scenario_trends/` - Integrated into Results
- `docs/model_documentation/` - Reorganized into Models
- `docs/full_results/` - Reorganized into Results

## Building the Documentation

To build the documentation:

```bash
pip install -e ".[doc]"
mkdocs serve  # Local preview
mkdocs build  # Production build
```

## Maintenance

When updating:
1. **Paper changes**: Update corresponding section in docs
2. **Model changes**: Update relevant model page
3. **New results**: Add to full_results.md with links
4. **New examples**: Reference from model pages

## Benefits

1. **Paper-Like Structure**: Familiar to academic readers
2. **Modular**: Easy to update individual sections
3. **Complete**: All scenarios documented with links
4. **Navigable**: Clear hierarchy and cross-references
5. **Maintainable**: Reused content, not duplicated
