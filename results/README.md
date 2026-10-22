# Audited version-5 result manifest

These JSON files are the evidence snapshot for the audited paper revision. Expensive and remote
outputs are retained so every reported value can be checked without rerunning paid
hardware or long optimizations.

## Hardware

- `kingston_100k_rung1.json`, `kingston_100k_rung1_counts.json`, and
  `kingston_rerun_plan_rung1.json`: 24-qubit Rung-1 depth series.
- `kingston_100k_arco_optangles.json`,
  `kingston_100k_arco_optangles_counts.json`, and
  `kingston_rerun_plan_arco.json`: 79-qubit own-angle Arco depth series.
- `kingston_transferred_arco_plan.json`,
  `kingston_transferred_arco_counts.json`, and
  `kingston_transferred_arco.json`: isolated publication 3 of the earlier
  transferred-angle Arco job.
- `exact51_plan.json`, `exact51_counts.json`, and `exact51_results.json`: faithful
  51-qubit encoding.
- `resource_model_exact.json`: physical circuits, edge multiplicities, measured
  qubits, job/calibration dates, and the exact job-time success proxy.

## Geometry and classical optimization

- `m_ladder_results.json`, `triviality.json`, and `arco_provenance.json` describe
  the frozen geographic instances.
- `ilp_geographic_ibge_2024_centroids.json` and `continental_blocks.json` are the
  version-5 Legal Amazon centroid results.
- `ilp_symmetry_corrected.json` is the corrected regular-grid hardness sweep.
- `dimension_sweep.json`, `coverage_preserving_2d.json`,
  `frozen_target_sweep.json`, and `ensemble_squeeze.json` support the geographic
  squeeze and its controls.
- `protection_2d.json`, `graceful_degradation.json`, and
  `tactical_replanning.json` support the protection and static sensitivity sections.
  `persistent_surveillance.json` is retained as historical output but is excluded
  from the active manuscript and reproduction workflow.

## Quantum and circuit calculations

- `block_decomposition.json`, `block_factorisation.json`, `ladder_rungs_v5.json`,
  `qwoa_simulability_v5.json`, `exact_encoding.json`, and
  `exact_circuit_51q.json` contain the exact structural checks.
- `qwoa_convergence_pilot.json`, `qwoa_p1_high_resolution.json`, and
  `qwoa_convergence_audit.json` contain the expectation-value convergence audit
  that supersedes the older direct-target comparisons. Those legacy JSONs are
  stored in `../quantum-amazon-surveillance-legacy-20260915/results/`, not in the
  active version-5 snapshot.
- `same_sector_mixer_control.json`, `mixer_scale_sensitivity.json`, and
  `same_sector_mixer_audit.json` contain the controlled `p=1` comparison in which
  both mixers use the same Dicke state, `k_opt` sector and exact cost. They find
  no stable benefit from shared-task weights or the near-complete topology.
- `statistical_audit.json` supplies the coherent IBM estimands, intervals and
  cardinality-matched null used in the current text.
- `rydberg_encoding_audit.json` supplies the exact negative test of the former
  Rydberg interpretation. The superseded positive scan and decay JSONs are
  preserved in `../quantum-amazon-surveillance-legacy-20260915/results/`; they
  are not inputs to this revision or evidence of a native MDS encoding.
- `factorisation_scope.json` supplies the algebraic hypotheses, counterexamples,
  and metric-geometric corollary used in the current theorem statement.

`scripts/verify_paper_claims.py --version 5`,
`scripts/verify_research_audit.py --version 5`,
`scripts/verify_v5_corrections.py`, `tests/test_v5_corrections.py`, and
`reproducibility/manifest_v5.json` define the active consistency gate. The two broad
scripts default to their frozen v4 modes when `--version 5` is omitted. Files not
listed here are not active version-5 evidence.

## Retained audit evidence

The statistical, encoding, factorisation, convergence-first, and same-sector
control audits below have been incorporated into the current text. The older
QWOA breadth and stability files remain exploratory and do not support a
positive manuscript claim. Their target-assisted tables and figures have been
removed from the active manuscript, while the JSONs remain as provenance:

- `statistical_audit.json`: coherent full-register and conditional concentration
  measures, binomial intervals/tests, and the 51-qubit cardinality-matched and
  measured-slack analyses.
- `rydberg_encoding_audit.json`: exact comparison among grid MDS states, hard-
  blockade-compatible states, the final Rydberg ground manifold, and the
  statevector target overlap.
- `factorisation_scope.json`: 250 random-incidence and 768 spatial checks,
  explicit counterexamples, and the algebraic conditions for circuit
  factorisation.
- `qwoa_expanded_p1.json`: 24 unselected seeds, three paired multiplicities, and
  oracle-assisted `p=1` QWOA/QAOA comparisons with bootstrap and paired tests.
- `qwoa_optimizer_stability.json`: a denser-grid/eight-start stress test showing
  that the preliminary QWOA/QAOA point estimates are not yet optimizer-converged.
- `qwoa_convergence_pilot.json`: six prespecified expectation-value cases over
  `m=1,2,3` and `p=1,2,3`, with exact-gradient refinement, warm-start controls,
  absolute and cardinality-conditioned probabilities, and per-endpoint
  convergence diagnostics.
- `qwoa_p1_high_resolution.json`: adaptive SHGO replication of the two `p=1`
  QWOA cases whose initial resolutions disagreed; generated by
  `scripts/run_qwoa_p1_high_resolution.py`.
- `qwoa_convergence_audit.json`: consolidated scientific decision: all six
  `p=1` pairs converge after the adaptive check, none of the `p=2,3` pairs does,
  and the old robust-mixer-advantage interpretation is unsupported.
- `qwoa_gradient_check.json`: central-difference validation of the exact adjoint
  QWOA and QAOA gradients used by the convergence pilot.
- `same_sector_mixer_control.json`: converged support-matched unit-weight and
  complete-graph unit-weight XY controls for all six prespecified cases.
- `mixer_scale_sensitivity.json`: RMS-normalized weighted-mixer rerun, separating
  relative edge-weight structure from the arbitrary global coupling scale.
- `same_sector_mixer_audit.json`: consolidated negative result: energy splits
  `3--3` and normalized `P(opt)` splits `3--3` against the support-matched
  control; no population inference is made from two independent seeds.
- `same_sector_mixer_gradient_check.json`: finite-difference checks for all new
  mixer families.
- `reproduction_report.json`: machine-readable record of the most recent
  one-command reproduction, including environment versions, executed commands,
  durations, status and PDF page counts.

Run `scripts/reproduce_v5.py --checks-only --skip-pdf` for the active offline gate.
