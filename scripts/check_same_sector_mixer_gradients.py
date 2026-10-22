#!/usr/bin/env python3
"""Finite-difference checks for the new same-sector XY mixer controls."""
from __future__ import annotations

import itertools
import json

import numpy as np

import benchmark_qwoa_qaoa as bm
import run_qwoa_convergence as convergence
from _paths import RESULTS_DIR


SEED = 12
M = 2
P = 2
STEP = 1e-6


def central_difference(value_fn, parameters: np.ndarray) -> np.ndarray:
    gradient = np.empty_like(parameters)
    for index in range(len(parameters)):
        plus, minus = parameters.copy(), parameters.copy()
        plus[index] += STEP
        minus[index] -= STEP
        gradient[index] = (value_fn(plus) - value_fn(minus)) / (2.0 * STEP)
    return gradient


def audit(label: str, sector: bm.QWOASector, parameters: np.ndarray) -> dict:
    state, value, analytic = convergence.qwoa_state_value_grad(sector, P, parameters)
    numeric = central_difference(
        lambda point: convergence.qwoa_state_value_grad(sector, P, point)[1],
        parameters,
    )
    reference = sector.evolve(P, parameters)
    error = float(np.max(np.abs(analytic - numeric)))
    return {
        "mixer": label,
        "parameters": parameters.tolist(),
        "value": float(value),
        "scaled_maximum_gradient_error": error / (
            1.0 + float(np.max(np.abs(numeric)))
        ),
        "state_norm_error": abs(float(np.vdot(state, state).real) - 1.0),
        "maximum_probability_difference_from_reference_evolver": float(
            np.max(np.abs(np.abs(state) ** 2 - reference))
        ),
    }


def main() -> None:
    instance = bm.build_instance(min_task_degree=3, rng_seed=SEED)
    A = np.asarray(instance["A_q"], dtype=int)
    n = A.shape[0]
    bm.LAM = float(n + 1)
    pc = bm.precompute(A, M)
    k_opt, _ = bm.classical_optimum(pc)
    raw = bm.cooperation_edges(A)
    rms = float(np.sqrt(np.mean(np.asarray(list(raw.values())) ** 2)))
    edge_sets = {
        "support_unweighted": {edge: 1.0 for edge in raw},
        "complete_unweighted": {
            edge: 1.0 for edge in itertools.combinations(range(n), 2)
        },
        "problem_weighted_rms_normalized": {
            edge: weight / rms for edge, weight in raw.items()
        },
    }
    rng = np.random.default_rng(20260916)
    bounds = convergence.parameter_bounds("qwoa", P, n + 1)
    rows = []
    for label, edges in edge_sets.items():
        parameters = bounds[:, 0] + rng.random(2 * P) * (
            bounds[:, 1] - bounds[:, 0]
        )
        rows.append(audit(label, bm.QWOASector(A, edges, k_opt, pc), parameters))
    output = {
        "description": "Central-difference audit of same-sector mixer gradients",
        "seed": SEED,
        "m": M,
        "p": P,
        "central_difference_step": STEP,
        "rows": rows,
        "all_checks_passed": all(
            row["scaled_maximum_gradient_error"] < 1e-7
            and row["state_norm_error"] < 1e-12
            and row["maximum_probability_difference_from_reference_evolver"] < 1e-12
            for row in rows
        ),
    }
    path = RESULTS_DIR / "same_sector_mixer_gradient_check.json"
    path.write_text(json.dumps(output, indent=1) + "\n", encoding="utf-8")
    for row in rows:
        print(
            f"{row['mixer']}: scaled gradient error="
            f"{row['scaled_maximum_gradient_error']:.3e}"
        )
    print(f"-> {path}")


if __name__ == "__main__":
    main()
