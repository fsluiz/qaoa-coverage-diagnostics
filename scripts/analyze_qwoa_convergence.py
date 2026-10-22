#!/usr/bin/env python3
"""Consolidate the convergence-first QWOA/QAOA pilot and its adaptive p=1 audit."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from _paths import RESULTS_DIR


PILOT = RESULTS_DIR / "qwoa_convergence_pilot.json"
HIGH_RESOLUTION = RESULTS_DIR / "qwoa_p1_high_resolution.json"
OUTPUT = RESULTS_DIR / "qwoa_convergence_audit.json"
MATCH_TOL = 1e-8


def summary(values: list[float]) -> dict | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {
        "n": len(values),
        "median": float(np.median(array)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def main() -> None:
    pilot = json.loads(PILOT.read_text(encoding="utf-8"))
    high = json.loads(HIGH_RESOLUTION.read_text(encoding="utf-8"))
    high_by_case = {(int(row["seed"]), int(row["m"])): row for row in high["rows"]}

    p1_rows = []
    depth_status = {str(p): {"cases": 0, "jointly_converged": 0} for p in (1, 2, 3)}
    for case in pilot["rows"]:
        for p in (1, 2, 3):
            depth = case["depths"][str(p)]
            qwoa_converged = bool(depth["qwoa"]["stability"]["joint_criterion_passed"])
            qaoa_converged = bool(depth["qaoa"]["stability"]["joint_criterion_passed"])
            adaptive = None
            if p == 1 and not qwoa_converged:
                adaptive = high_by_case.get((int(case["seed"]), int(case["m"])))
                if adaptive is None:
                    raise AssertionError("missing adaptive p=1 QWOA replication")
                matches = [
                    endpoint
                    for endpoint in depth["qwoa"]["refinements"]
                    if abs(
                        float(endpoint["final_expected_energy"])
                        - float(adaptive["expected_energy"])
                    )
                    <= MATCH_TOL
                    and abs(float(endpoint["metrics"]["p_opt"]) - float(adaptive["p_opt"]))
                    <= MATCH_TOL
                ]
                qwoa_converged = len(matches) >= 1 and bool(adaptive["success"])
                if not qwoa_converged:
                    raise AssertionError("adaptive p=1 result did not replicate a pilot endpoint")
            depth_status[str(p)]["cases"] += 1
            depth_status[str(p)]["jointly_converged"] += int(
                qwoa_converged and qaoa_converged
            )
            if p == 1:
                qwoa = depth["qwoa"]["metrics"]
                qaoa = depth["qaoa"]["metrics"]
                p1_rows.append({
                    "seed": int(case["seed"]),
                    "m": int(case["m"]),
                    "qwoa_converged_after_adaptive_check": qwoa_converged,
                    "qaoa_converged": qaoa_converged,
                    "adaptive_check_used": adaptive is not None,
                    "qwoa_expected_energy": float(qwoa["expected_energy"]),
                    "qaoa_expected_energy": float(qaoa["expected_energy"]),
                    "qwoa_p_opt": float(qwoa["p_opt"]),
                    "qaoa_p_opt_unconditional": float(qaoa["p_opt"]),
                    "qaoa_p_k_opt": float(qaoa["p_k_opt"]),
                    "qaoa_p_opt_given_k_opt": float(qaoa["p_opt_given_k_opt"]),
                    "unconditional_p_opt_ratio": float(
                        depth["p_opt_ratio_qwoa_over_qaoa_unconditional"]
                    ),
                    "conditional_p_opt_ratio": float(
                        depth["p_opt_ratio_qwoa_over_qaoa_conditional_k_opt"]
                    ),
                })

    if depth_status["1"]["jointly_converged"] != 6:
        raise AssertionError("p=1 should converge in all six cases after adaptive audit")
    if depth_status["2"]["jointly_converged"] or depth_status["3"]["jointly_converged"]:
        raise AssertionError("unexpected p>1 convergence status")

    raw_ratios = [row["unconditional_p_opt_ratio"] for row in p1_rows]
    conditional_ratios = [row["conditional_p_opt_ratio"] for row in p1_rows]
    qaoa_sector_mass = [row["qaoa_p_k_opt"] for row in p1_rows]
    by_m = {}
    for m in (1, 2, 3):
        selected = [row for row in p1_rows if row["m"] == m]
        by_m[str(m)] = {
            "n": len(selected),
            "unconditional_p_opt_ratio": summary(
                [row["unconditional_p_opt_ratio"] for row in selected]
            ),
            "conditional_p_opt_ratio": summary(
                [row["conditional_p_opt_ratio"] for row in selected]
            ),
        }

    output = {
        "description": "Audited conclusion of the prespecified convergence-first QWOA/QAOA pilot",
        "inputs": [str(PILOT.relative_to(RESULTS_DIR.parent)), str(HIGH_RESOLUTION.relative_to(RESULTS_DIR.parent))],
        "depth_convergence": depth_status,
        "p1": {
            "rows": p1_rows,
            "unconditional_p_opt_ratio": summary(raw_ratios),
            "conditional_p_opt_ratio": summary(conditional_ratios),
            "qaoa_probability_of_k_opt": summary(qaoa_sector_mass),
            "qwoa_unconditional_p_opt_wins": int(sum(value > 1.0 for value in raw_ratios)),
            "qwoa_conditional_p_opt_wins": int(
                sum(value > 1.0 for value in conditional_ratios)
            ),
            "qaoa_lower_expected_energy": int(
                sum(row["qaoa_expected_energy"] < row["qwoa_expected_energy"] for row in p1_rows)
            ),
            "by_multiplicity_descriptive_only": by_m,
        },
        "scientific_interpretation": {
            "primary_objective": (
                "QAOA has lower converged expected energy in all six p=1 cases."
            ),
            "absolute_sampling_metric": (
                "QWOA has larger absolute P(opt) in all six p=1 cases, but it is initialized "
                "with oracle knowledge of k_opt while QAOA explores all cardinalities."
            ),
            "cardinality_diagnostic": (
                "After conditioning QAOA on k_opt, QWOA wins four of six cases and loses both "
                "m=1 cases; the ratio spans more than two orders of magnitude. Conditioning "
                "does not make the ansatzes fully controlled because QAOA angles were optimized "
                "on the full-register expectation."
            ),
            "depth_result": (
                "No p=2 or p=3 case reached paired optimizer convergence; their point estimates "
                "are best-known values only and cannot support a depth-scaling claim."
            ),
            "decision": (
                "The existing claim of a robust QWOA mixer advantage is not supported. Treat "
                "the comparison as a sector-prior/objective-mismatch diagnostic. A mixer-only "
                "claim would require a same-sector control such as an unweighted XY mixer from "
                "the same Dicke state, plus converged p>1 optimization and compiled costs for a "
                "faithful oracle."
            ),
        },
    }
    OUTPUT.write_text(json.dumps(output, indent=1) + "\n", encoding="utf-8")
    print(f"p=1 jointly converged: {depth_status['1']['jointly_converged']}/6")
    print(f"p=2 jointly converged: {depth_status['2']['jointly_converged']}/6")
    print(f"p=3 jointly converged: {depth_status['3']['jointly_converged']}/6")
    print(
        "p=1 median ratios: "
        f"unconditional={np.median(raw_ratios):.6g}, "
        f"conditional={np.median(conditional_ratios):.6g}"
    )
    print(f"-> {OUTPUT}")


if __name__ == "__main__":
    main()
