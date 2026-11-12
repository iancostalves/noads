## Summary

Opt-in update of the aircraft design model (`aircraft_model="update"`, GAM V3.0) with its
pre-computed optima, and a document comparing the update with the main paper.

- **Model (opt-in, paper stays the default and is guarded by `tests/test_gam_paper_baseline.py`)**
  - GAM V3.0 (small-turbofan scale factor, refitted regressions), powertrain scale laws and
    maximum power per propulsor, fuel cell with thermal management, size-dependent LH2 tank
    gravimetric index, engine efficiency trend, four-point technology scenarios (2020/2040/2060/2080).
  - Per-market feasibility filter for battery-electric and fuel cell designs; new constraints
    (`unit_power_ratio <= 1`, `closure_gap <= 1e-4`, `relative_efficiency_gain >= 1`).
  - Entry into service from 2035 (Jet-A) and 2040 (alternatives); same start for the LH2 gas turbine
    and fuel cell; no turboprops in the optimized fleets. Energy pathways are those of the paper.
- **Optimization tooling**: checkpoint/warm start for the robust runs, every optimizer iteration saved
  (`iterations.npz`), iteration videos. Iteration archives, checkpoints and videos are git-ignored.
- **Results**: 29 optima in `docs/examples/optimization/results/update/` (27 single, 2 robust), all feasible.
- **Comparison**: `docs/update/comparison/update_vs_paper_comparison.pdf` (organised like the paper
  Methods and Results), with the scripts, figures and `results_summary.csv` to regenerate it.

## Main findings (Mid technology, cumulative CO2 as % of the aviation budget, paper → update)

- Breakthrough trend 166 → 216, availability 130 → 177; baseline SSP2 231 → 290; drop-in trend 189 → 247.
- The low-demand variants stay at 100 % but need more avoided demand (breakthrough: 39 → 45 % in 2075).
- Scenario ranking unchanged. Fuel cell aircraft lose share (breakthrough trend, 2075: 57 → 29 %);
  LH2 gas turbine, Jet-A v2 and battery-electric gain share.
- Causes (small-turbofan penalty on small markets, fuel cell and power limit, later entry-into-service
  bounds) are read from the design results; no ablation study quantifies them individually.

## Where to look

- Update optima: `docs/examples/optimization/results/update/<name>/opt_result.json`; paper optima: `.../results/<name>/opt_result.json`.
- Comparison: `docs/update/comparison/` (PDF, `results_summary.csv`, `figures/`, `build_pdf.py`).
- Update documentation: `docs/update/`.

## Not included

- Extended timeline (EIS to 2080, end year 2100): on hold pending validation of these results.

## Test plan

- [x] `pytest tests/` (paper baseline test included)
- [x] pre-commit on the changed files
- [x] All 29 update optima report a feasible solution
- [ ] Review `update_vs_paper_comparison.pdf`
