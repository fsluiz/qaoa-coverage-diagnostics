#!/usr/bin/env python3
"""Consistency checks for the post-v3 scientific audit outputs (frozen version-3 copy)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

from _paths import RESULTS_DIR


checks: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    checks.append((name, bool(condition), detail))


def load(name: str):
    return json.loads((RESULTS_DIR / name).read_text(encoding="utf-8"))


def close(left: float, right: float, tolerance: float = 1e-10) -> bool:
    return abs(left - right) <= tolerance


# Statistical definitions and raw-count consequences.
stats = load("statistical_audit.json")
rung1 = {row["tag"]: row for row in stats["rung1"]["circuits"]}
p1 = rung1["rung1_qaoa_p1"]
check("p=1 raw optimum count is 79/100000", p1["n_opt"] == 79 and p1["shots"] == 100000)
check("p=1 contains 15351 measured weight-10 shots", p1["n_in_optimal_sector"] == 15351)
check("all-register concentration is 46.02", close(p1["eta_all_register"], 46.0208355556, 1e-8))
check("conditional weight-10 concentration is 35.05", close(p1["eta_conditional_k_opt"], 35.0454964932, 1e-8))
check("ideal conditional weight-10 concentration is 71.17", close(
    p1["ideal_eta_conditional_k_opt"], 71.1712406655, 1e-8
))
check("legacy hybrid ratio is retained only as 5.38 diagnostic", close(p1["legacy_hybrid_eta_deprecated"], 5.379, 0.01))
check("ideal-null test uses null binomial variance", close(
    p1["ideal_null_test"]["null_binomial_sd_count"],
    math.sqrt(100000 * p1["ideal_p_opt"] * (1 - p1["ideal_p_opt"])),
))

exact51 = stats["faithful_51q"]
check("51q marginal post-cover count is 53569", exact51["post_bits_form_cover"]["count"] == 53569)
check("51q cardinality-matched null is about 0.447", close(
    exact51["post_bits_form_cover"]["cardinality_matched_uniform_within_sector_baseline"],
    0.4470232167,
    1e-8,
))
check("51q cardinality-matched cover enhancement is only about 1.20", close(
    exact51["post_bits_form_cover"]["cardinality_matched_enhancement"], 1.1983493921, 1e-8
))
check("no measured 51q string satisfies every encoded slack equality",
      exact51["all_measured_slack_equalities_satisfied"]["count"] == 0)

# Rydberg state-set and ground-manifold checks.
rydberg = load("rydberg_encoding_audit.json")
cases = {row["spacing_um"]: row for row in rydberg["cases"]}
check("3x3 grid has ten minimum dominating sets", all(
    row["grid"]["n_minimum_dominating_sets"] == 10 for row in cases.values()
))
check("even d=7 blockade excludes two grid MDS states",
      cases[7.0]["mds_vs_blockade"]["n_blockade_incompatible_mds"] == 2)
check("d=5 blockade excludes six grid MDS states",
      cases[5.0]["mds_vs_blockade"]["n_blockade_incompatible_mds"] == 6)
check("neither final Rydberg ground manifold contains a grid MDS", all(
    row["final_classical_hamiltonian"]["n_ground_states_that_are_grid_mds"] == 0
    for row in cases.values()
))
check("reintegrated d=7 target overlap reproduces 31.27%", close(
    cases[7.0]["dynamics_at_reported_duration"]["P_all_grid_mds"], 0.3127370389, 1e-8
))

# Algebraic-vs-geometric scope checks.
scope = load("factorisation_scope.json")
check("250 random incidence hypergraphs pass direct-sum check",
      scope["random_incidence_hypergraphs"]["direct_sum_checks_passed"] == 250)
check("scope audit contains 768 spatial instances", len(scope["spatial_rows"]) == 768)
check("all observed G_ZZ edges obey the metric 2R condition",
      scope["spatial_checks"]["all_G_ZZ_edges_obey_distance_at_most_2R"])
check("all spatial cost supports split over their own connected components",
      scope["spatial_checks"]["all_costs_direct_sum_over_their_support_components"])
check("sparse path counterexample is connected", (
    scope["counterexamples"]["sparse_connected_path"]["n_interactions"] == 23
    and scope["counterexamples"]["sparse_connected_path"]["n_components"] == 1
))
check("dense non-geographic counterexample factorises into four blocks",
      scope["counterexamples"]["dense_disconnected_cliques"]["block_sizes"] == [6, 6, 6, 6])

# Expanded QWOA breadth experiment.
expanded = load("qwoa_expanded_p1.json")
rows = expanded["rows"]
check("expanded QWOA study has 24 seeds x 3 multiplicities", len(rows) == 72)
check("expanded QWOA study uses no outcome-selected seed subset",
      sorted(set(row["seed"] for row in rows)) == list(range(24)))
check("every expanded instance has strict triple-cover support",
      min(row["minimum_task_degree"] for row in rows) >= 3)
check("each multiplicity has 24 paired instances", all(
    expanded["summary"][str(m)]["n_instances"] == 24 for m in (1, 2, 3)
))
check("noiseless p=1 result is heterogeneous across multiplicity", (
    expanded["summary"]["1"]["qwoa_wins"] == 24
    and expanded["summary"]["2"]["qaoa_wins"] == 8
    and expanded["summary"]["3"]["qaoa_wins"] == 1
))
check("median first-order hardware proxy favours QAOA for every multiplicity", all(
    expanded["summary"][str(m)]["median_hardware_proxy_ratio"] < 1 for m in (1, 2, 3)
))

stability = load("qwoa_optimizer_stability.json")
check("QWOA p=1 angle-optimizer convergence is explicitly not established", (
    not stability["optimizer_convergence_established"]
    and stability["winner_changed_in_any_case"]
    and stability["max_relative_probability_disagreement"] > 0.20
))

# Convergence-first expectation-value pilot and adaptive p=1 resolution audit.
pilot = load("qwoa_convergence_pilot.json")
convergence = load("qwoa_convergence_audit.json")
high_resolution = load("qwoa_p1_high_resolution.json")
gradient_check = load("qwoa_gradient_check.json")
same_sector = load("same_sector_mixer_control.json")
mixer_scale = load("mixer_scale_sensitivity.json")
mixer_audit = load("same_sector_mixer_audit.json")
mixer_gradient = load("same_sector_mixer_gradient_check.json")
check("convergence pilot uses the two prespecified random seeds",
      pilot["meta"]["prespecified_pilot_seeds"] == [12, 17])
check("convergence pilot contains six paired multiplicity cases",
      len(pilot["rows"]) == 6)
check("convergence pilot fixes the sufficient penalty at n+1", all(
    row["penalty"] == row["n"] + 1 for row in pilot["rows"]
))
check("adaptive p=1 audit contains only the two prespecified failures",
      {(row["seed"], row["m"]) for row in high_resolution["rows"]}
      == {(12, 2), (17, 3)})
check("adaptive p=1 results replicate a prior endpoint", all(
    row["replicates_pilot_level"].startswith("shgo_")
    for row in high_resolution["rows"]
))
check("all six p=1 pairs are optimizer-converged after adaptive audit",
      convergence["depth_convergence"]["1"]["jointly_converged"] == 6)
check("no p=2 or p=3 pair passes the convergence criterion", all(
    convergence["depth_convergence"][str(p)]["jointly_converged"] == 0
    for p in (2, 3)
))
check("QAOA has lower converged expected energy in every p=1 case",
      convergence["p1"]["qaoa_lower_expected_energy"] == 6)
check("conditioning on k_opt reverses both m=1 comparisons", all(
    row["conditional_p_opt_ratio"] < 1.0
    for row in convergence["p1"]["rows"] if row["m"] == 1
))
check("QWOA p=1 conditional comparisons remain heterogeneous",
      convergence["p1"]["conditional_p_opt_ratio"]["minimum"] < 1.0
      and convergence["p1"]["conditional_p_opt_ratio"]["maximum"] > 100.0)

root = Path(__file__).resolve().parents[1]
main_tex = (root / "paper" / "paper_v3.tex").read_text(encoding="utf-8")
supp_tex = (root / "paper" / "supplement_v3.tex").read_text(encoding="utf-8")
check("main text withdraws the robust QWOA-advantage interpretation",
      "do not establish a robust QWOA advantage" in main_tex)
check("main text reports that no p>1 pair is converged",
      "no pair satisfies the prespecified convergence criterion" in main_tex)
check("legacy equal-gate claim is no longer asserted as a result",
      "genuine but modest noiseless edge" not in main_tex
      and "The noiseless mixer-structure edge survives" not in supp_tex)
check("both adjoint gradients match central differences", all(
    row["scaled_maximum_gradient_error"] < 1e-7
    for row in gradient_check["rows"]
))
check("adjoint state evolvers reproduce the reference probabilities", all(
    row["maximum_probability_difference_from_reference_evolver"] < 1e-12
    for row in gradient_check["rows"]
))
check("states used by the gradient audit remain normalized", all(
    row["state_norm_error"] < 1e-12 for row in gradient_check["rows"]
))

# Same-Dicke-state, same-sector mixer control.
check("same-sector control contains the six prespecified cases",
      {(row["seed"], row["m"]) for row in same_sector["rows"]}
      == {(seed, m) for seed in (12, 17) for m in (1, 2, 3)})
check("all support and complete unit-weight endpoints converge", all(
    row["controls"][name]["stability"]["joint_criterion_passed"]
    for row in same_sector["rows"]
    for name in ("support_unweighted", "complete_unweighted")
))
check("all RMS-normalized weighted endpoints converge", all(
    row["rms_normalized_weighted"]["stability"]["joint_criterion_passed"]
    for row in mixer_scale["rows"]
))
check("RMS-normalized edge weights have unit RMS", all(
    close(row["normalized_weight_rms"], 1.0, 1e-12)
    for row in mixer_scale["rows"]
))
raw_support = mixer_audit["summaries"][
    "problem_weighted_raw_minus_support_unweighted"
]
norm_support = mixer_audit["summaries"][
    "problem_weighted_rms_normalized_minus_support_unweighted"
]
norm_complete = mixer_audit["summaries"][
    "problem_weighted_rms_normalized_minus_complete_unweighted"
]
check("raw weighted and support-unweighted energy wins split three-three",
      raw_support["weighted_lower_energy"] == raw_support["control_lower_energy"] == 3)
check("raw weighted P(opt) wins only four of six support controls",
      raw_support["weighted_larger_p_opt"] == 4
      and raw_support["control_larger_p_opt"] == 2)
check("normalized weighted and support-unweighted energy wins split three-three",
      norm_support["weighted_lower_energy"] == norm_support["control_lower_energy"] == 3)
check("normalized weighted and support-unweighted P(opt) wins split three-three",
      norm_support["weighted_larger_p_opt"] == norm_support["control_larger_p_opt"] == 3)
check("normalized median P(opt) ratio is below one and heterogeneous",
      # The SHGO endpoint is stable at the precision reported in the paper;
      # low-order BLAS differences move the unrounded ratio by about 3e-9.
      close(norm_support["p_opt_ratio"]["median"], 0.9753102729, 1e-8)
      and norm_support["p_opt_ratio"]["minimum"] < 0.31
      and norm_support["p_opt_ratio"]["maximum"] > 1.98)
check("complete unit-weight control has lower normalized energy in four of six",
      norm_complete["weighted_lower_energy"] == 2
      and norm_complete["control_lower_energy"] == 4)
check("seed 12 support is complete and seed 17 has 62 of 66 edges", all(
    (close(row["cooperation_edge_density"], 1.0, 1e-12) if row["seed"] == 12
     else close(row["cooperation_edge_density"], 62 / 66, 1e-12))
    for row in mixer_audit["rows"]
))
check("finite beta window exposes scale sensitivity in two cases",
      mixer_audit["scale_window_sensitivity"]
      ["cases_whose_weighted_energy_changes_by_more_than_1e-8"] == 2
      and mixer_audit["scale_window_sensitivity"]
      ["cases_whose_raw_optimum_maps_outside_normalized_beta_window"]
      == [{"seed": 17, "m": 1}, {"seed": 17, "m": 2}])
check("new mixer gradients and state evolvers pass", mixer_gradient["all_checks_passed"])
check("main text reports the negative same-sector control", all(fragment in main_tex for fragment in (
    "same-sector control", "Energy then still splits three--three",
    "median ratio $0.975$", "only two independent task seeds"
)))
check("active manuscript excludes retired positive QWOA framing",
      "fig_advantage_vs_m.pdf" not in supp_tex
      and "fig_qwoa_vs_qaoa.pdf" not in supp_tex
      and "ansatz we advocate" not in main_tex
      and "cardinality-preserving mixer of" not in supp_tex
      and "a cardinality prior, not an identified mixer advantage" in main_tex)


for name, passed, detail in checks:
    print(f"{'PASS' if passed else 'FAIL'}  {name}" + (f" ({detail})" if detail else ""))
failed = [name for name, passed, _ in checks if not passed]
print(f"\n{len(checks) - len(failed)}/{len(checks)} audit checks passed")
if failed:
    sys.exit(1)
