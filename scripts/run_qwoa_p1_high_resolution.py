#!/usr/bin/env python3
"""Reproduce the adaptive high-resolution p=1 QWOA convergence checks.

Only the two cases that failed the prespecified 1024/2048-point agreement test
are evaluated. The case selection is fixed by the pilot protocol, not by the
direction or size of an algorithmic comparison.
"""
from __future__ import annotations

import json
import time

import numpy as np
from scipy.optimize import shgo

import benchmark_qwoa_qaoa as bm
import run_qwoa_convergence as convergence
from _paths import RESULTS_DIR


CASES = ((12, 2), (17, 3))
SAMPLES = 4096
ITERATIONS = 3
MATCH_TOLERANCE = 1e-7
PILOT = RESULTS_DIR / "qwoa_convergence_pilot.json"
OUTPUT = RESULTS_DIR / "qwoa_p1_high_resolution.json"


def run_case(seed: int, multiplicity: int, pilot: dict) -> dict:
    instance = bm.build_instance(min_task_degree=3, rng_seed=seed)
    incidence = np.asarray(instance["A_q"], dtype=int)
    n = incidence.shape[0]
    penalty = n + 1
    bm.LAM = float(penalty)
    precomputed = bm.precompute(incidence, multiplicity)
    k_opt, _ = bm.classical_optimum(precomputed)
    sector = bm.QWOASector(
        incidence, bm.cooperation_edges(incidence), k_opt, precomputed
    )
    bounds = convergence.parameter_bounds("qwoa", 1, penalty)
    scipy_bounds = [tuple(row) for row in bounds]

    def state_value_gradient(parameters: np.ndarray):
        return convergence.qwoa_state_value_grad(sector, 1, parameters)

    def objective(parameters: np.ndarray) -> float:
        return float(state_value_gradient(np.asarray(parameters, dtype=float))[1])

    started = time.time()
    result = shgo(
        objective,
        scipy_bounds,
        n=SAMPLES,
        iters=ITERATIONS,
        sampling_method="sobol",
        minimizer_kwargs={
            "method": "L-BFGS-B",
            "jac": lambda point: state_value_gradient(
                np.asarray(point, dtype=float)
            )[2],
            "bounds": scipy_bounds,
            "options": {
                "maxiter": convergence.LOCAL_MAXITER,
                "maxfun": convergence.LOCAL_MAXFUN,
                "ftol": 1e-13,
                "gtol": 1e-9,
                "maxls": 30,
            },
        },
    )
    if not result.success or result.x is None:
        raise RuntimeError(
            f"seed={seed}, m={multiplicity}: SHGO failed: {result.message}"
        )
    parameters = np.asarray(result.x, dtype=float)
    probabilities = sector.evolve(1, parameters)
    p_opt = float(convergence.qwoa_metrics(probabilities, sector)["p_opt"])

    pilot_case = next(
        row
        for row in pilot["rows"]
        if int(row["seed"]) == seed and int(row["m"]) == multiplicity
    )
    endpoints = pilot_case["depths"]["1"]["qwoa"]["refinements"]
    matches = [
        row
        for row in endpoints
        if abs(float(row["final_expected_energy"]) - float(result.fun))
        <= MATCH_TOLERANCE
        and abs(float(row["metrics"]["p_opt"]) - p_opt) <= MATCH_TOLERANCE
    ]
    if not matches:
        raise AssertionError(
            f"seed={seed}, m={multiplicity}: high-resolution endpoint does not "
            "replicate either prespecified pilot resolution"
        )
    closest = min(
        matches,
        key=lambda row: abs(float(row["final_expected_energy"]) - float(result.fun)),
    )
    return {
        "seed": seed,
        "m": multiplicity,
        "n": SAMPLES,
        "iterations": ITERATIONS,
        "expected_energy": float(result.fun),
        "parameters": parameters.tolist(),
        "p_opt": p_opt,
        "nfev": int(result.nfev),
        "n_local_minima": int(len(result.funl)),
        "runtime_s": round(time.time() - started, 3),
        "success": bool(result.success),
        "message": str(result.message),
        "replicates_pilot_level": str(closest["repeat"]),
    }


def main() -> None:
    pilot = json.loads(PILOT.read_text(encoding="utf-8"))
    rows = []
    for seed, multiplicity in CASES:
        print(f"seed={seed}, m={multiplicity}: starting adaptive SHGO", flush=True)
        row = run_case(seed, multiplicity, pilot)
        rows.append(row)
        print(
            f"seed={seed}, m={multiplicity}: E={row['expected_energy']:.12f}, "
            f"P(opt)={row['p_opt']:.12g}, replicate={row['replicates_pilot_level']}",
            flush=True,
        )
    payload = {
        "description": (
            "Adaptive high-resolution SHGO replication for the two p=1 QWOA "
            "cases whose initial 1024/2048-point searches disagreed"
        ),
        "protocol": {
            "samples": SAMPLES,
            "iterations": ITERATIONS,
            "objective": "expectation of cardinality + 13 * number of violated tasks",
            "optimizer": (
                "SHGO Sobol sampling with exact-gradient L-BFGS-B local refinement"
            ),
            "selection_rule": (
                "only cases failing the prespecified two-resolution p=1 "
                "convergence check"
            ),
            "interpretation": "replication audit, not a new performance-selected ensemble",
        },
        "rows": rows,
    }
    OUTPUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"-> {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
