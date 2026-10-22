#!/usr/bin/env python3
"""Consolidate the raw and RMS-normalized same-sector mixer controls."""
from __future__ import annotations

import json

import numpy as np

from _paths import RESULTS_DIR


RAW = RESULTS_DIR / "same_sector_mixer_control.json"
NORMALIZED = RESULTS_DIR / "mixer_scale_sensitivity.json"
OUTPUT = RESULTS_DIR / "same_sector_mixer_audit.json"
EXPECTED_CASES = {(seed, m) for seed in (12, 17) for m in (1, 2, 3)}


def _summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "n": len(array),
        "median": float(np.median(array)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def main() -> None:
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    normalized = json.loads(NORMALIZED.read_text(encoding="utf-8"))
    raw_by_case = {(int(row["seed"]), int(row["m"])): row for row in raw["rows"]}
    norm_by_case = {
        (int(row["seed"]), int(row["m"])): row for row in normalized["rows"]
    }
    if set(raw_by_case) != EXPECTED_CASES or set(norm_by_case) != EXPECTED_CASES:
        raise AssertionError("same-sector audit does not contain the six expected cases")

    rows = []
    for seed, m in sorted(EXPECTED_CASES):
        raw_row = raw_by_case[(seed, m)]
        norm_row = norm_by_case[(seed, m)]
        weighted = raw_row["controls"]["problem_weighted"]
        support = raw_row["controls"]["support_unweighted"]
        complete = raw_row["controls"]["complete_unweighted"]
        weighted_norm = norm_row["rms_normalized_weighted"]
        weighted_converged = bool(
            weighted["stability"]["joint_criterion_passed_after_adaptive_audit"]
        )
        if not all((
            weighted_converged,
            support["stability"]["joint_criterion_passed"],
            complete["stability"]["joint_criterion_passed"],
            weighted_norm["stability"]["joint_criterion_passed"],
        )):
            raise AssertionError(f"nonconverged control at seed={seed}, m={m}")
        if abs(float(norm_row["normalized_weight_rms"]) - 1.0) > 1e-12:
            raise AssertionError("RMS normalization failed")
        if not all(raw_row["matrix_checks"].values()):
            # Hermiticity errors are floats equal to zero, so test them separately.
            booleans = [
                value for value in raw_row["matrix_checks"].values()
                if isinstance(value, bool)
            ]
            errors = [
                value for value in raw_row["matrix_checks"].values()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            ]
            if not all(booleans) or max(errors, default=0.0) > 1e-12:
                raise AssertionError("mixer matrix audit failed")

        def metrics(block: dict) -> dict:
            return {
                "expected_energy": float(block["metrics"]["expected_energy"]),
                "p_opt": float(block["metrics"]["p_opt"]),
                "parameters": [float(value) for value in block["parameters"]],
            }

        entries = {
            "problem_weighted_raw": metrics(weighted),
            "problem_weighted_rms_normalized": metrics(weighted_norm),
            "support_unweighted": metrics(support),
            "complete_unweighted": metrics(complete),
        }
        comparisons = {}
        for weighted_label in (
            "problem_weighted_raw", "problem_weighted_rms_normalized"
        ):
            for control_label in ("support_unweighted", "complete_unweighted"):
                left, right = entries[weighted_label], entries[control_label]
                comparisons[f"{weighted_label}_minus_{control_label}"] = {
                    "expected_energy_difference": (
                        left["expected_energy"] - right["expected_energy"]
                    ),
                    "p_opt_difference": left["p_opt"] - right["p_opt"],
                    "p_opt_ratio": left["p_opt"] / max(right["p_opt"], 1e-300),
                }
        rows.append({
            "seed": seed,
            "m": m,
            "k_opt": int(raw_row["k_opt"]),
            "sector_dimension": int(raw_row["sector_dimension"]),
            "cooperation_edge_density": float(
                raw_row["mixer_metadata"]["problem_weighted"]["edge_density"]
            ),
            "raw_weight_rms": float(norm_row["raw_weight_rms"]),
            "all_endpoints_converged": True,
            "scale_mapping": {
                "raw_weighted_beta": entries["problem_weighted_raw"]["parameters"][1],
                "beta_for_same_unitary_after_rms_normalization": (
                    entries["problem_weighted_raw"]["parameters"][1]
                    * float(norm_row["raw_weight_rms"])
                ),
                "same_unitary_beta_inside_normalized_window": bool(
                    entries["problem_weighted_raw"]["parameters"][1]
                    * float(norm_row["raw_weight_rms"])
                    <= np.pi + 1e-12
                ),
            },
            "mixers": entries,
            "comparisons": comparisons,
        })

    summaries = {}
    for weighted_label in (
        "problem_weighted_raw", "problem_weighted_rms_normalized"
    ):
        for control_label in ("support_unweighted", "complete_unweighted"):
            key = f"{weighted_label}_minus_{control_label}"
            energy = [row["comparisons"][key]["expected_energy_difference"] for row in rows]
            probability = [row["comparisons"][key]["p_opt_difference"] for row in rows]
            ratios = [row["comparisons"][key]["p_opt_ratio"] for row in rows]
            summaries[key] = {
                "n": len(rows),
                "weighted_lower_energy": int(sum(value < 0.0 for value in energy)),
                "control_lower_energy": int(sum(value > 0.0 for value in energy)),
                "weighted_larger_p_opt": int(sum(value > 0.0 for value in probability)),
                "control_larger_p_opt": int(sum(value < 0.0 for value in probability)),
                "energy_difference": _summary(energy),
                "p_opt_ratio": _summary(ratios),
            }

    scale_energy_changes = [
        row["mixers"]["problem_weighted_rms_normalized"]["expected_energy"]
        - row["mixers"]["problem_weighted_raw"]["expected_energy"]
        for row in rows
    ]
    scale_probability_ratios = [
        row["mixers"]["problem_weighted_rms_normalized"]["p_opt"]
        / max(row["mixers"]["problem_weighted_raw"]["p_opt"], 1e-300)
        for row in rows
    ]
    outside_normalized_window = [
        row for row in rows
        if not row["scale_mapping"]["same_unitary_beta_inside_normalized_window"]
    ]
    output = {
        "description": "Consolidated same-sector XY-mixer audit",
        "inputs": [
            str(RAW.relative_to(RESULTS_DIR.parent)),
            str(NORMALIZED.relative_to(RESULTS_DIR.parent)),
        ],
        "design": {
            "cases": "prespecified seeds 12 and 17 paired over m=1,2,3",
            "fixed": (
                "instance, k_opt sector, uniform Dicke initial state, exact cost, p=1, "
                "optimizer protocol, and parameter windows"
            ),
            "primary_raw_contrast": (
                "shared-task edge weights versus unit weights on identical support"
            ),
            "scale_sensitivity": (
                "shared-task weights divided by their RMS versus the same unit-weight support"
            ),
            "secondary_topology_contrast": (
                "weighted support versus complete-graph unit-weight XY"
            ),
            "inference": (
                "descriptive only; the six rows contain only two independent instance seeds"
            ),
        },
        "rows": rows,
        "summaries": summaries,
        "scale_window_sensitivity": {
            "cases_whose_raw_optimum_maps_outside_normalized_beta_window": [
                {"seed": row["seed"], "m": row["m"]}
                for row in outside_normalized_window
            ],
            "cases_whose_weighted_energy_changes_by_more_than_1e-8": int(
                sum(abs(value) > 1e-8 for value in scale_energy_changes)
            ),
            "weighted_energy_change_normalized_minus_raw": _summary(scale_energy_changes),
            "normalized_over_raw_p_opt_ratio": _summary(scale_probability_ratios),
            "interpretation": (
                "A finite common beta window is not scale invariant. The normalized analysis "
                "therefore tests relative weight heterogeneity rather than raw coupling magnitude."
            ),
        },
        "scientific_interpretation": {
            "same_support_result": (
                "Weighted and unit-weight mixers each have lower expected energy in three of six "
                "cases, both with raw and RMS-normalized weights."
            ),
            "sampling_result": (
                "Raw weights have larger P(opt) in four of six support-matched cases, but after "
                "RMS normalization the split is three to three and the median ratio is below one."
            ),
            "topology_result": (
                "The complete unweighted control has lower energy in four of six cases after RMS "
                "normalization; P(opt) again splits three to three."
            ),
            "decision": (
                "The supplied sector explains the earlier QWOA/QAOA sampling contrast, but the "
                "problem-informed edge weights and near-complete geometry do not provide a robust "
                "p=1 mixer advantage on this pilot. Any positive effect is case-, metric-, and "
                "scale-window-dependent. No p>1 or population-level inference is licensed."
            ),
        },
    }
    OUTPUT.write_text(json.dumps(output, indent=1) + "\n", encoding="utf-8")
    primary = summaries[
        "problem_weighted_rms_normalized_minus_support_unweighted"
    ]
    print(
        "RMS-normalized weighted vs support-unweighted: "
        f"energy wins {primary['weighted_lower_energy']}-{primary['control_lower_energy']}, "
        f"P(opt) wins {primary['weighted_larger_p_opt']}-{primary['control_larger_p_opt']}, "
        f"median P ratio={primary['p_opt_ratio']['median']:.6g}"
    )
    print(f"-> {OUTPUT}")


if __name__ == "__main__":
    main()
