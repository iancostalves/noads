# Documentation Reorganization Summary

This document summarizes the reorganization of the `noads` documentation structure.

## Overview

The documentation has been reorganized from a collection of examples and a basic user guide into a comprehensive, multi-level documentation system that includes:

1. **Extended User Guide** - Deep dive into architecture and design patterns
2. **Model Documentation** - Detailed model assumptions, equations, and calibration
3. **Scenario Trends** - Online version of the main paper
4. **Full Results** - Online version of the supplementary information

## New Documentation Structure

```
docs/
├── user_guide_extended/        # Extended user guide (NEW)
│   ├── index.md               # Overview of extended guide
│   ├── oop_architecture.md    # OOP benefits with .dot visualizations
│   ├── workflow_management.md # Modular and dynamic workflows
│   └── core_concepts.md       # Fundamental concepts
│
├── model_documentation/        # Model documentation (NEW)
│   ├── index.md               # Overview of models
│   ├── assumptions.md         # Detailed assumptions and limitations
│   ├── equations.md           # Mathematical formulation
│   └── calibration.md         # Calibration methodology
│
├── scenario_trends/            # Main paper online (NEW)
│   ├── index.md               # Overview and summary
│   ├── intro_methods.md       # Introduction and methodology
│   ├── main_results.md        # Key findings and results
│   └── discussion.md          # Implications and discussion
│
├── full_results/               # Supplementary info online (NEW)
│   ├── index.md               # Organization and navigation
│   ├── single_policy/         # Single-policy optimization results
│   │   ├── ssp1_fossil.md     # SSP1 baseline
│   │   └── breakthrough.md    # Breakthrough scenarios
│   └── robust_policy/         # Robust policy results
│       └── index.md           # Robust optimization results
│
├── user_guide/                 # Original user guide (KEPT)
│   ├── index.md
│   ├── core.html
│   ├── models.html
│   └── scenarios.html
│
├── examples/                   # Original examples (KEPT)
│   ├── models/
│   └── optimization/
│
└── paper/                      # LaTeX source files (UNCHANGED)
    ├── my-manuscript.tex
    ├── supplementary-information.tex
    └── biblio.bib
```

## Content Mapping

### From LaTeX to Markdown

Content was extracted and converted from the original `.tex` files:

| Source | Destination | Content |
|--------|-------------|---------|
| `supplementary-information.tex` | `model_documentation/` | Model assumptions, equations, calibration |
| `my-manuscript.tex` | `scenario_trends/` | Introduction, methods, results, discussion |
| Both `.tex` files | `model_documentation/assumptions.md` | Model assumptions table |
| `supplementary-information.tex` | `model_documentation/equations.md` | Mathematical formulations |
| `supplementary-information.tex` | `model_documentation/calibration.md` | Calibration data and methods |
| `my-manuscript.tex` | `scenario_trends/main_results.md` | Scenario results and analysis |
| `my-manuscript.tex` | `scenario_trends/discussion.md` | Implications and limitations |

### Key Conversions

1. **LaTeX equations** → **Markdown with MathJax**: All equations converted to `$$...$$` format
2. **LaTeX citations** → **Markdown citations**: `\cite{ref}` → `[@ref]` format
3. **LaTeX tables** → **Markdown tables**: Complex tables simplified for readability
4. **LaTeX figures** → **Figure references**: Links to PDF files in results directories

## Extended User Guide

### Purpose
Demonstrate the benefits of object-oriented programming for managing complex multidisciplinary optimization problems.

### Key Features
- **OOP Architecture** (`oop_architecture.md`):
  - Class hierarchies and inheritance
  - Modular energy system design
  - Visualization with .dot file references
  - Benefits: modularity, reusability, extensibility

- **Workflow Management** (`workflow_management.md`):
  - Temporal scenarios with selective vectorization
  - Multi-scenario analysis for robustness
  - Dynamic workflow assembly
  - Automatic coupling and dependency resolution

- **Core Concepts** (`core_concepts.md`):
  - Model abstraction layers
  - Stream-based flow modeling
  - Production pathway composition
  - Technology evolution over time
  - Time-dependent controls with delays

### .dot File Integration

The OOP architecture page references the graphviz `.dot` files in `docs/examples/optimization/single_policy/`:
- `general.dot` - High-level system overview
- `fleet_assembly.dot` - Fleet composition patterns
- `all_couplings.dot` - Complete model coupling
- `energy_mix_*.dot` - Energy system visualizations

These visualizations demonstrate the modular structure enabled by OOP design.

## Model Documentation

### Purpose
Provide comprehensive technical documentation of model assumptions, mathematical formulation, and calibration.

### Structure

1. **Assumptions** (`assumptions.md`):
   - Summary table from main paper
   - Detailed explanations by discipline
   - Key simplifications and limitations
   - Model verification approach

2. **Equations** (`equations.md`):
   - Complete mathematical formulation
   - Nomenclature and variable definitions
   - Equations organized by model type:
     - Demand models (logistic functions)
     - Fleet models (deployment dynamics)
     - Energy models (production pathways)
     - Constraint equations
     - Objective functions

3. **Calibration** (`calibration.md`):
   - Data sources and coverage
   - Calibration methodology
   - Parameter estimation results
   - Validation approaches
   - Uncertainty quantification

### Data Sources

All calibration data sources documented:
- ICAO: Historical RPK data
- World Bank: Socioeconomic drivers
- AeroSCOPE: Flight-level performance
- Planespotters: Aircraft lifetimes
- AR6 Database: Background scenarios
- Literature: Technology parameters

## Scenario Trends (Main Paper)

### Purpose
Online version of the main paper, presenting scenario optimization results and analysis.

### Structure

1. **Introduction and Methods** (`intro_methods.md`):
   - Research context and positioning
   - Methodology overview
   - Numerical methods and computational gains
   - Background scenarios (SSPs and RCPs)
   - Optimization formulations
   - Technology scenarios

2. **Main Results** (`main_results.md`):
   - Baseline scenarios (SSP1, SSP2, SSP5)
   - Drop-in mitigation (SAF only)
   - Breakthrough mitigation (SAF + alternative aircraft)
   - Scenario-robust policies
   - Comparison with AR6 ensemble
   - Summary metrics table

3. **Discussion** (`discussion.md`):
   - Key implications for climate mitigation
   - Energy system requirements
   - Equity and demand management
   - Planetary boundaries beyond climate
   - Role of optimization in policy design
   - Technology uncertainty and hedging
   - Model limitations and future work
   - Policy recommendations

### Key Features

- Comprehensive scenario summary table
- Visual result references
- Technology sensitivity analysis
- Comparison with literature
- Policy implications

## Full Results (Supplementary Information)

### Purpose
Detailed results for all simulated scenarios, organized as online SI.

### Organization

```
full_results/
├── index.md                    # Navigation and overview
├── single_policy/              # Single-policy optimizations
│   ├── ssp1_fossil.md         # Baseline SSP1
│   ├── ssp2_fossil.md         # Baseline SSP2 (template)
│   ├── ssp5_fossil.md         # Baseline SSP5 (template)
│   ├── dropin.md              # Drop-in scenarios (template)
│   ├── dropin_availability.md # Drop-in high availability (template)
│   ├── dropin_lowdemand.md    # Drop-in with demand caps (template)
│   ├── breakthrough.md        # Breakthrough scenarios (created)
│   ├── breakthrough_availability.md  # (template)
│   └── breakthrough_lowdemand.md     # (template)
└── robust_policy/              # Robust optimizations
    └── index.md                # Robust trends and low-demand (created)
```

### Content for Each Scenario

Each scenario page includes:
- Scenario description and parameters
- Key metrics by technology scenario
- Results summary
- Links to detailed result PDFs
- Fleet composition patterns
- Energy system evolution
- Key observations
- Comparison with other scenarios
- Policy implications

### Result File Organization

Results stored in:
- `docs/examples/optimization/single_policy/single_policy/[scenario-name]/`
  - `fleet_*.pdf` - Fleet composition
  - `energy_*.pdf` - Energy production
  - `conso_*.pdf` - Energy consumption
  - `demand_*.pdf` - Demand evolution (low-demand only)

- `docs/examples/optimization/single_robust_policy/robust_policy/[scenario-name]/`
  - Similar structure for robust policies
  - Includes results for each background scenario
  - Mean results across scenarios

## Navigation Structure

The `mkdocs.yml` has been updated with the new structure:

```yaml
nav:
- Home: [Overview, Changelog, Credits, Licenses]
- User Guide (Original): user_guide/
- Extended User Guide: [Index, OOP Architecture, Workflow, Core Concepts]
- Model Documentation: [Index, Assumptions, Equations, Calibration]
- Scenario Trends (Paper): [Index, Intro/Methods, Main Results, Discussion]
- Full Results (SI): [Index]
- Examples: [Optimization, Models]
- API documentation: reference/noads/
```

## Benefits of New Structure

### For Users

1. **Multiple Entry Points**:
   - Quick overview (scenario trends)
   - Technical details (model documentation)
   - Implementation guide (extended user guide)
   - Complete results (full results)

2. **Progressive Disclosure**:
   - Start with high-level results
   - Drill down to model details
   - Explore specific scenarios
   - Understand implementation

3. **Cross-References**:
   - Links between sections
   - Related scenarios linked
   - Methods referenced from results
   - Equations referenced from assumptions

### For Developers

1. **Architecture Patterns**:
   - OOP design demonstrated
   - Workflow patterns explained
   - Code organization principles

2. **Model Understanding**:
   - Complete mathematical formulation
   - Calibration methodology
   - Data sources documented

3. **Result Interpretation**:
   - Scenario context provided
   - Key metrics highlighted
   - Comparisons facilitated

### For Researchers

1. **Reproducibility**:
   - All assumptions documented
   - Equations fully specified
   - Data sources cited
   - Calibration described

2. **Validation**:
   - Results compared with literature
   - Sensitivity analysis documented
   - Limitations explicitly stated

3. **Extension**:
   - Clear structure for adding scenarios
   - Template pages for new results
   - Modular organization

## Markdown Extensions Used

The documentation uses several Markdown extensions configured in `mkdocs.yml`:

- **pymdownx.arithmatex**: LaTeX math rendering with MathJax
- **pymdownx.superfences**: Code blocks and Mermaid diagrams
- **pymdownx.details**: Collapsible sections
- **admonition**: Note/warning/tip boxes
- **toc**: Table of contents with permalinks
- **pymdownx.snippets**: File inclusion
- **footnotes**: Footnote support
- **mkdocs-bibtex**: Bibliography management

## Building the Documentation

### Requirements

Install documentation dependencies:

```bash
pip install -e ".[doc]"
```

This installs packages from `pyproject.toml`:
- mkdocs-material
- mkdocstrings
- mkdocs-gallery
- mkdocs-bibtex
- And other extensions

### Build Commands

**Serve locally** (with auto-reload):
```bash
mkdocs serve
```

**Build static site**:
```bash
mkdocs build
```

**Deploy to GitLab Pages**:
```bash
mike deploy --push --update-aliases [version] latest
```

### Viewing Results

After serving locally, access at `http://127.0.0.1:8000/`

## Future Extensions

### Additional Result Pages

Templates are provided for all scenarios. To complete:

1. Copy template (e.g., `ssp2_fossil.md`)
2. Update scenario-specific details
3. Add key metrics from result files
4. Link related scenarios
5. Add to navigation if desired

### Additional Sections

Potential additions:
- **Tutorials**: Step-by-step guides for common tasks
- **How-To Guides**: Recipes for specific problems
- **API Examples**: Code snippets using the package
- **Case Studies**: Real-world applications
- **FAQ**: Common questions and answers

### Interactive Elements

Could add:
- Interactive plots (Plotly)
- Jupyter notebooks embedded
- Parameter exploration tools
- Scenario comparison widgets

## Migration Notes

### What Was Kept

- Original `user_guide/` - Preserved for backward compatibility
- Original `examples/` - Gallery of working examples
- Original `.tex` files in `paper/` - Source of truth
- API documentation - Generated from code

### What Was Added

- Four new major sections (14 new pages)
- Cross-references between sections
- Template pages for expansion
- Navigation structure in mkdocs.yml

### What Was Changed

- `mkdocs.yml` navigation updated
- No changes to source code
- No changes to examples
- No changes to LaTeX files

## Maintenance

### Keeping Content Updated

When updating models or results:

1. **LaTeX files remain source**: Update `.tex` files first
2. **Manual sync to Markdown**: Changes must be manually propagated
3. **Equations and tables**: Check for discrepancies
4. **Citations**: Ensure bibliography is current

### Adding New Scenarios

1. Run optimization to generate results
2. Copy appropriate template page
3. Fill in scenario-specific metrics
4. Link to result PDF files
5. Update cross-references

### Documentation Reviews

Recommended review frequency:
- After major code changes: Update API docs
- After model changes: Update model documentation
- After new results: Add result pages
- Before releases: Full documentation review

## References

### Original Sources

- `docs/paper/my-manuscript.tex` - Main paper LaTeX source
- `docs/paper/supplementary-information.tex` - SI LaTeX source
- `docs/paper/biblio.bib` - Bibliography

### Result Files

- `docs/examples/optimization/single_policy/single_policy/` - Single-policy results
- `docs/examples/optimization/single_robust_policy/robust_policy/` - Robust policy results

### Key Documentation Pages

- `docs/user_guide_extended/oop_architecture.md` - OOP design patterns
- `docs/model_documentation/assumptions.md` - Complete model assumptions
- `docs/scenario_trends/main_results.md` - Main findings
- `docs/full_results/index.md` - Results navigation

## Summary

The documentation has been successfully reorganized into a comprehensive, multi-level structure that serves multiple audiences:

- **Users**: High-level results and guidance
- **Developers**: Architecture and implementation
- **Researchers**: Complete technical details

All content extracted from original LaTeX sources and converted to Markdown, with proper cross-referencing and navigation. The structure supports progressive disclosure, allowing readers to start with overviews and drill down to details as needed.
