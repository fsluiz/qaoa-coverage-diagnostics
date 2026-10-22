#!/usr/bin/env python3
"""Stress-test the preliminary p=1 QWOA/QAOA angle optimiser.

The broad run uses a 9x9 grid and four local starts.  This independent protocol
uses a 13x13 grid and eight starts on seed 3, which contains representative
QWOA-win, small-gap, and reversal cases across m=1,2,3.  Disagreement is evidence
that neither search protocol has established the global p=1 maximum.

Writes results/qwoa_optimizer_stability.json.
"""
from __future__ import annotations

import json

import numpy as np

import benchmark_qwoa_qaoa as bm
from _paths import RESULTS_DIR
from run_qwoa_expanded import run_case


SEED = 3
GRID_SIZE = 13
STARTS = 8


def main() -> None:
    primary = json.loads((RESULTS_DIR / "qwoa_expanded_p1.json").read_text(encoding="utf-8"))
    primary_rows = {
        (int(row["seed"]), int(row["m"])): row for row in primary["rows"]
    }
    instance = bm.build_instance(min_task_degree=3, rng_seed=SEED)
    A = np.asarray(instance["A_q"], dtype=int)
    comparisons = []
    for m in (1, 2, 3):
        reference = primary_rows[(SEED, m)]
        stress = run_case(A, SEED, m, GRID_SIZE, STARTS)
        row = {
            "seed": SEED,
            "m": m,
            "primary_grid_size": int(primary["meta"]["grid_size"]),
            "primary_local_starts": int(primary["meta"]["local_starts"]),
            "stress_grid_size": GRID_SIZE,
            "stress_local_starts": STARTS,
            "primary": {
                "qwoa_p_opt": reference["qwoa"]["p_opt"],
                "qaoa_p_opt": reference["qaoa"]["p_opt"],
                "qwoa_over_qaoa": reference["absolute_p_opt_ratio_qwoa_over_qaoa"],
            },
            "stress": {
                "qwoa_p_opt": stress["qwoa"]["p_opt"],
                "qaoa_p_opt": stress["qaoa"]["p_opt"],
                "qwoa_over_qaoa": stress["absolute_p_opt_ratio_qwoa_over_qaoa"],
            },
        }
        row["relative_change_qwoa"] = (
            row["stress"]["qwoa_p_opt"] / row["primary"]["qwoa_p_opt"] - 1.0
        )
        row["relative_change_qaoa"] = (
            row["stress"]["qaoa_p_opt"] / row["primary"]["qaoa_p_opt"] - 1.0
        )
        row["same_winner"] = bool(
            (row["primary"]["qwoa_p_opt"] > row["primary"]["qaoa_p_opt"])
            == (row["stress"]["qwoa_p_opt"] > row["stress"]["qaoa_p_opt"])
        )
        comparisons.append(row)

    max_relative_disagreement = max(
        abs(row[key])
        for row in comparisons
        for key in ("relative_change_qwoa", "relative_change_qaoa")
    )
    converged = max_relative_disagreement <= 0.01 and all(row["same_winner"] for row in comparisons)
    out = {
        "description": "Independent angular-search stress test of the preliminary broad p=1 ensemble",
        "comparisons": comparisons,
        "convergence_criterion": "all probabilities agree within 1% relative and every paired winner is unchanged",
        "max_relative_probability_disagreement": max_relative_disagreement,
        "winner_changed_in_any_case": any(not row["same_winner"] for row in comparisons),
        "optimizer_convergence_established": converged,
        "decision": (
            "Do not use the broad p=1 point estimates as converged manuscript values; "
            "retain them as a preliminary breadth scan and implement a global/convergence study."
        ),
    }
    path = RESULTS_DIR / "qwoa_optimizer_stability.json"
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    for row in comparisons:
        print(
            f"m={row['m']}: ratio {row['primary']['qwoa_over_qaoa']:.3f} -> "
            f"{row['stress']['qwoa_over_qaoa']:.3f}; same winner={row['same_winner']}"
        )
    print(f"convergence established: {converged}")
    print(f"-> {path}")


if __name__ == "__main__":
    main()

