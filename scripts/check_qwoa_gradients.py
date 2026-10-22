#!/usr/bin/env python3
"""Finite-difference validation of the adjoint gradients used in the QWOA audit."""
from __future__ import annotations

import json

import numpy as np

import benchmark_qwoa_qaoa as bm
import run_qwoa_convergence as convergence
from _paths import RESULTS_DIR


INSTANCE_SEED = 12
M = 2
P = 2
PARAMETER_SEED = 4
STEP = 1e-6


def central_difference(value_fn, parameters: np.ndarray) -> np.ndarray:
    gradient = np.empty_like(parameters)
    for index in range(len(parameters)):
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += STEP
        minus[index] -= STEP
        gradient[index] = (value_fn(plus) - value_fn(minus)) / (2.0 * STEP)
    return gradient


def audit(name, state_value_grad, reference_probs, bounds, rng) -> dict:
    parameters = bounds[:, 0] + rng.random(2 * P) * (bounds[:, 1] - bounds[:, 0])
    state, value, analytic = state_value_grad(parameters)
    numeric = central_difference(lambda x: state_value_grad(x)[1], parameters)
    reference = reference_probs(parameters)
    absolute_error = float(np.max(np.abs(analytic - numeric)))
    relative_error = absolute_error / (1.0 + float(np.max(np.abs(numeric))))
    return {
        "ansatz": name,
        "parameters": parameters.tolist(),
        "value": float(value),
        "analytic_gradient": analytic.tolist(),
        "central_difference_gradient": numeric.tolist(),
        "maximum_absolute_gradient_error": absolute_error,
        "scaled_maximum_gradient_error": relative_error,
        "state_norm_error": abs(float(np.vdot(state, state).real) - 1.0),
        "maximum_probability_difference_from_reference_evolver": float(
            np.max(np.abs(np.abs(state) ** 2 - reference))
        ),
    }


def main() -> None:
    instance = bm.build_instance(min_task_degree=3, rng_seed=INSTANCE_SEED)
    A = np.asarray(instance["A_q"], dtype=int)
    n = A.shape[0]
    bm.LAM = float(n + 1)
    pc = bm.precompute(A, M)
    k_opt, _ = bm.classical_optimum(pc)
    sector = bm.QWOASector(A, bm.cooperation_edges(A), k_opt, pc)
    rng = np.random.default_rng(PARAMETER_SEED)
    rows = [
        audit(
            "qaoa",
            lambda x: convergence.qaoa_state_value_grad(pc, n, P, x),
            lambda x: bm.qaoa_run(pc, n, P, x),
            convergence.parameter_bounds("qaoa", P, n + 1),
            rng,
        ),
        audit(
            "qwoa",
            lambda x: convergence.qwoa_state_value_grad(sector, P, x),
            lambda x: sector.evolve(P, x),
            convergence.parameter_bounds("qwoa", P, n + 1),
            rng,
        ),
    ]
    output = {
        "description": "Central-difference validation of exact adjoint gradients",
        "instance_seed": INSTANCE_SEED,
        "m": M,
        "p": P,
        "parameter_seed": PARAMETER_SEED,
        "central_difference_step": STEP,
        "rows": rows,
        "all_scaled_gradient_errors_below_1e-7": all(
            row["scaled_maximum_gradient_error"] < 1e-7 for row in rows
        ),
    }
    path = RESULTS_DIR / "qwoa_gradient_check.json"
    path.write_text(json.dumps(output, indent=1) + "\n", encoding="utf-8")
    for row in rows:
        print(
            f"{row['ansatz']}: scaled gradient error="
            f"{row['scaled_maximum_gradient_error']:.3e}, "
            f"probability error={row['maximum_probability_difference_from_reference_evolver']:.3e}"
        )
    print(f"-> {path}")


if __name__ == "__main__":
    main()
