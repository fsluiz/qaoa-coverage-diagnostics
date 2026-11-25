# Offline reproduction

Run from a fresh checkout or extracted release. Python 3.11 and the versions in
`requirements-reproduce-v7.txt` describe the audited environment. Qiskit and an
IBM account are optional and are unnecessary for the main reproduction command.

```sh
python3 -m pip install -r requirements-reproduce-v7.txt
python3 scripts/reproduce_diagnostics.py --output /tmp/qaoa-reproduced --full
```

Omit `--full` for raw-count point checks and figure generation without repeating
the archived bootstrap draws. Full mode repeats the 14 block resamplings at
10,000 draws and the three campaign analyses at 4,000 draws, using archived seeds.
The script verifies DATA_MANIFEST hashes, runs the scientific regression tests,
reconstructs exact controls and physical/readout audits, and compares numerical
outputs at absolute/relative tolerance 1e-10. Outputs go to the selected directory;
archived inputs are preserved. No command in this workflow submits hardware jobs.

## Figure and input map

| Article result | Generator | Released inputs |
| --- | --- | --- |
| Fig. 1: geography, placement and block success | `scripts/make_block_figure.py` | `data/frozen_rung1.json`, outline, `results/rung1_physical_layout.json`, block and joint/slice archives |
| Fig. 2: three fixed-circuit jobs | `analysis/revision_v7/analyze_epoch.py`, `combine_epochs.py`, `campaign_artifacts.py` | `analysis/ibm_variance_campaign/results/e0*_plan.json`, `e0*_counts.json` |
| Fig. 3: routed cones and cross-block correlations | `scripts/make_exact51_layout_figure.py` | `results/ibm_circuits/exact51_*.ops.json`, `results/exact51_physical_layout.json`, `results/exact51_cross_block.json` |
| Fig. 4: block covariance profiles | `analysis/revision_v7/block_covariance.py`, `block_covariance_artifacts.py` | historical faithful counts and all four faithful pubs in each campaign job |
| Physical calibration tables | `analysis/revision_v8/audit_physical_data.py` | QPY registry, instruction lists, plans/counts, layouts and full job properties |
| Independent readout control | `analysis/revision_v8/readout_audit.py` | frozen incidence/angles, layout and directional rates in the three calibration archives |
| Supplement controls and ensemble | `diagnostic_controls.py`, `mixer_summary.py`, `ensemble_analysis.py` under `analysis/revision_v7/` | frozen instances, source spreadsheet, `results/qwoa_20seeds.json`, 600-instance archives |

Fig. 3's plotted correlation statistics are recomputed from both historical raw
count archives by `scripts/calc_exact51_cross_block.py` (10,000 permutations per
overlapping pair). It writes `results/exact51_cross_block.json`; use a disposable
checkout if rerunning that historical generator. The full main workflow verifies
the published archive hash, independently recomputes all 156 plotted correlations,
normalizations and cone classifications from weighted raw counts with
`analysis/revision_v8/cross_block_points.py`, and regenerates the figure.
Permutation p-values are retained with their original source and random seed.

## Provenance and scope

`DATA_MANIFEST.json` gives hashes, byte counts, roles and original input paths.
`PATH_MAPPING.json` records relocation of historical `paper/` analysis paths to
`analysis/`. Original count, plan, QPY and calibration bytes are unchanged.
The calibration archives retain job IDs, parameter timestamps/units, SDK/retrieval
metadata and execution times; they contain no account credentials.

Hardware counts are aggregate bitstring counts; within-job twirling strata and
execution-time schedules not saved by the original jobs cannot be reconstructed.
Offline reproduction does not repeat historical optimizer searches, source-data
selection decisions, circuit transpilation or hardware execution. Frozen inputs
and archived numerical endpoints support the reported conditional analyses.

The old `analysis/revision_v7/reproduce.py` and manuscript-layout tests belong to
the historical full manuscript workspace. Use `scripts/reproduce_diagnostics.py`
for this code-and-data distribution, which excludes manuscript sources and PDFs.
