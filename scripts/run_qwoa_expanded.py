#!/usr/bin/env python3
"""Broad, unselected p=1 QWOA/QAOA ensemble.

This is a breadth study, not a depth study.  Each task seed produces one strict
triple-cover incidence matrix (every task degree is at least three), and the same
matrix is analysed at m=1,2,3.  Both ansatzes receive the same one-sided diagonal
oracle, the same penalty grid, and direct oracle-assisted optimisation of P(opt).

The optimisation protocol is deterministic: a coarse two-angle grid followed by
L-BFGS-B from its best points.  Direct P(opt) optimisation uses the known optimum
and is therefore an upper-bound experiment, not an implementable variational
workflow.

Default run:

    python3 scripts/run_qwoa_expanded.py --seeds 0:24

The JSON is checkpointed after every (seed, m) row and can be resumed.
Writes results/qwoa_expanded_p1.json by default.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import binomtest, wilcoxon

import benchmark_qwoa_qaoa as bm
from _paths import RESULTS_DIR


PENALTY_GRID = [13.0, 25.0, 50.0]
EPS_PROXY = 0.006


def parse_int_spec(spec: str) -> list[int]:
    if ":" in spec:
        parts = [int(v) if v else None for v in spec.split(":")]
        if len(parts) == 2:
            start, stop = parts
            step = 1
        elif len(parts) == 3:
            start, stop, step = parts
        else:
            raise ValueError(spec)
        return list(range(0 if start is None else start, stop, 1 if step is None else step))
    return [int(v) for v in spec.split(",") if v.strip()]


def optimise_probability(score, grid_size: int, starts: int) -> tuple[np.ndarray, float, int]:
    """Maximise a two-angle probability using a deterministic grid and polish."""
    gammas = np.linspace(-np.pi, np.pi, grid_size, endpoint=False)
    betas = np.linspace(0.0, np.pi, grid_size)
    grid = []
    evaluations = 0
    for gamma in gammas:
        for beta in betas:
            value = float(score(np.asarray([gamma, beta])))
            evaluations += 1
            grid.append((value, np.asarray([gamma, beta], dtype=float)))
    grid.sort(key=lambda item: item[0], reverse=True)
    best_probability, best_parameters = grid[0]
    for _, initial in grid[:starts]:
        result = minimize(
            lambda parameters: -float(score(parameters)),
            initial,
            method="L-BFGS-B",
            bounds=[(-np.pi, np.pi), (0.0, np.pi)],
            options={"maxiter": 100, "ftol": 1e-12, "gtol": 1e-9},
        )
        evaluations += int(result.nfev)
        probability = -float(result.fun)
        if probability > best_probability:
            best_probability = probability
            best_parameters = np.asarray(result.x, dtype=float)
    return best_parameters, best_probability, evaluations


def bootstrap_ci(values: np.ndarray, statistic, rng: np.random.Generator, n_boot: int = 10000) -> list[float]:
    draws = np.empty(n_boot, dtype=float)
    n = len(values)
    for i in range(n_boot):
        draws[i] = statistic(values[rng.integers(0, n, size=n)])
    return [float(v) for v in np.quantile(draws, [0.025, 0.975])]


def summarise(rows: list[dict]) -> dict:
    rng = np.random.default_rng(20260915)
    summary = {}
    for m in sorted(set(int(row["m"]) for row in rows)):
        selected = [row for row in rows if int(row["m"]) == m]
        qwoa = np.asarray([row["qwoa"]["p_opt"] for row in selected], dtype=float)
        qaoa = np.asarray([row["qaoa"]["p_opt"] for row in selected], dtype=float)
        log_ratio = np.log10(np.maximum(qwoa, 1e-300)) - np.log10(np.maximum(qaoa, 1e-300))
        wins = int(np.sum(qwoa > qaoa))
        losses = int(np.sum(qwoa < qaoa))
        non_ties = wins + losses
        try:
            wilcoxon_p = float(wilcoxon(log_ratio, alternative="two-sided").pvalue)
        except ValueError:
            wilcoxon_p = None
        summary[str(m)] = {
            "n_instances": len(selected),
            "qwoa_p_opt_mean": float(qwoa.mean()),
            "qwoa_p_opt_median": float(np.median(qwoa)),
            "qwoa_p_opt_median_bootstrap95": bootstrap_ci(qwoa, np.median, rng),
            "qaoa_p_opt_mean": float(qaoa.mean()),
            "qaoa_p_opt_median": float(np.median(qaoa)),
            "qaoa_p_opt_median_bootstrap95": bootstrap_ci(qaoa, np.median, rng),
            "paired_log10_qwoa_over_qaoa_mean": float(log_ratio.mean()),
            "paired_log10_qwoa_over_qaoa_median": float(np.median(log_ratio)),
            "paired_log10_ratio_median_bootstrap95": bootstrap_ci(log_ratio, np.median, rng),
            "qwoa_wins": wins,
            "qaoa_wins": losses,
            "ties": len(selected) - non_ties,
            "two_sided_sign_test_p": (
                float(binomtest(wins, non_ties, 0.5).pvalue) if non_ties else None
            ),
            "paired_wilcoxon_log_ratio_p": wilcoxon_p,
            "median_qwoa_sector_concentration": float(
                np.median([row["qwoa"]["concentration_vs_sector_uniform"] for row in selected])
            ),
            "median_qaoa_all_register_concentration": float(
                np.median([row["qaoa"]["concentration_vs_all_register_uniform"] for row in selected])
            ),
            "median_hardware_proxy_ratio": float(
                np.median([row["hardware_proxy_qwoa_over_qaoa"] for row in selected])
            ),
        }
    return summary


def run_case(A: np.ndarray, seed: int, m: int, grid_size: int, starts: int) -> dict:
    n = A.shape[0]
    bm.LAM = PENALTY_GRID[0]
    reference = bm.precompute(A, m)
    k_opt, opt_states = bm.classical_optimum(reference)
    if k_opt is None:
        raise RuntimeError(f"seed={seed}, m={m}: strict instance is infeasible")
    opt_states = np.asarray(opt_states, dtype=int)
    opt_mask = np.zeros(1 << n, dtype=bool)
    opt_mask[opt_states] = True

    edges = bm.cooperation_edges(A)
    n_edges_xy = len(edges)
    n_ZZ = int(sum(
        1 for i, j in itertools.combinations(range(n), 2)
        if np.any((A[i] == 1) & (A[j] == 1))
    ))

    # Diagonalise the fixed mixer once.  Only the diagonal oracle changes with lambda.
    sector = bm.QWOASector(A, edges, k_opt, reference)
    qwoa_candidates = []
    qaoa_candidates = []
    for lam in PENALTY_GRID:
        bm.LAM = lam
        pc = bm.precompute(A, m)
        sector.Esec = pc["energy"][sector.states]

        def qwoa_score(parameters):
            return float(sector.evolve(1, parameters)[sector.feas_sec].sum())

        qwoa_parameters, qwoa_probability, qwoa_nfev = optimise_probability(
            qwoa_score, grid_size, starts
        )
        qwoa_candidates.append({
            "lambda": lam,
            "gamma": float(qwoa_parameters[0]),
            "beta": float(qwoa_parameters[1]),
            "p_opt": qwoa_probability,
            "optimizer_evaluations": qwoa_nfev,
        })

        def qaoa_score(parameters):
            return float(bm.qaoa_run(pc, n, 1, parameters)[opt_mask].sum())

        qaoa_parameters, qaoa_probability, qaoa_nfev = optimise_probability(
            qaoa_score, grid_size, starts
        )
        qaoa_candidates.append({
            "lambda": lam,
            "gamma": float(qaoa_parameters[0]),
            "beta": float(qaoa_parameters[1]),
            "p_opt": qaoa_probability,
            "optimizer_evaluations": qaoa_nfev,
        })

    bm.LAM = PENALTY_GRID[0]
    qwoa = dict(max(qwoa_candidates, key=lambda row: row["p_opt"]))
    qaoa = dict(max(qaoa_candidates, key=lambda row: row["p_opt"]))
    qwoa_cx = bm.cx_counts(n_ZZ, n_edges_xy, k_opt, n, 1, "QWOA")
    qaoa_cx = bm.cx_counts(n_ZZ, n_edges_xy, k_opt, n, 1, "QAOA")
    p_uniform_sector = len(opt_states) / math.comb(n, k_opt)
    p_uniform_all = len(opt_states) / (1 << n)
    qwoa.update(
        cx_proxy=qwoa_cx,
        sector_dimension=math.comb(n, k_opt),
        uniform_sector_baseline=p_uniform_sector,
        concentration_vs_sector_uniform=qwoa["p_opt"] / p_uniform_sector,
        all_penalty_candidates=qwoa_candidates,
    )
    qaoa.update(
        cx_proxy=qaoa_cx,
        full_dimension=1 << n,
        uniform_all_register_baseline=p_uniform_all,
        concentration_vs_all_register_uniform=qaoa["p_opt"] / p_uniform_all,
        all_penalty_candidates=qaoa_candidates,
    )
    return {
        "seed": seed,
        "m": m,
        "n": n,
        "n_tasks": int(A.shape[1]),
        "minimum_task_degree": int(A.sum(axis=0).min()),
        "mean_task_degree": float(A.sum(axis=0).mean()),
        "n_ZZ": n_ZZ,
        "n_XY_edges_including_connectivity_bridges": n_edges_xy,
        "k_opt": int(k_opt),
        "n_opt": int(len(opt_states)),
        "qwoa": qwoa,
        "qaoa": qaoa,
        "absolute_p_opt_ratio_qwoa_over_qaoa": qwoa["p_opt"] / max(qaoa["p_opt"], 1e-300),
        "hardware_proxy_qwoa_over_qaoa": (
            (1.0 - EPS_PROXY) ** qwoa_cx * qwoa["p_opt"]
            / max((1.0 - EPS_PROXY) ** qaoa_cx * qaoa["p_opt"], 1e-300)
        ),
    }


def checkpoint(path: Path, meta: dict, rows: list[dict], started: float) -> None:
    out = {
        "meta": {**meta, "runtime_s": round(time.time() - started, 3)},
        "rows": rows,
        "summary": summarise(rows) if rows else {},
    }
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0:24", help="comma list or Python-style start:stop[:step]")
    parser.add_argument("--m-values", default="1,2,3")
    parser.add_argument("--grid-size", type=int, default=9)
    parser.add_argument("--starts", type=int, default=4)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "qwoa_expanded_p1.json")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    seeds = parse_int_spec(args.seeds)
    m_values = parse_int_spec(args.m_values)
    started = time.time()

    rows = []
    if args.out.exists() and not args.no_resume:
        rows = load_rows = json.loads(args.out.read_text(encoding="utf-8")).get("rows", [])
        print(f"resume: loaded {len(load_rows)} rows from {args.out}")
    completed = {(int(row["seed"]), int(row["m"])) for row in rows}
    meta = {
        "description": "Preliminary unselected, strict-coverage, oracle-assisted p=1 breadth ensemble",
        "scientific_status": "preliminary; optimizer convergence must be established before manuscript use",
        "requested_seeds": seeds,
        "m_values": m_values,
        "p": 1,
        "minimum_task_degree": 3,
        "penalty_grid_for_both_ansatzes": PENALTY_GRID,
        "optimizer": "deterministic coarse grid plus L-BFGS-B; direct P(opt)",
        "grid_size": args.grid_size,
        "local_starts": args.starts,
        "eps_hardware_proxy": EPS_PROXY,
        "qualifications": [
            "P(opt) optimisation knows the exact optimum and is an oracle-assisted upper bound.",
            "This broad ensemble tests p=1 only and does not establish depth scaling.",
            "The one-sided diagonal oracle is shared but is not a demonstrated 2-local hardware encoding.",
            "CX values are analytic resource proxies, not backend-transpiled gate counts.",
            "A denser-grid stress test on seed 3 changed individual optima; use qwoa_optimizer_stability.json.",
        ],
    }

    for seed in seeds:
        instance = bm.build_instance(min_task_degree=3, rng_seed=seed)
        A = np.asarray(instance["A_q"], dtype=int)
        print(
            f"seed={seed}: R={instance['R_chosen']:.0f} km, "
            f"degree min/mean={A.sum(axis=0).min()}/{A.sum(axis=0).mean():.2f}"
        )
        for m in m_values:
            if (seed, m) in completed:
                print(f"  m={m}: already complete")
                continue
            before = time.time()
            row = run_case(A, seed, m, args.grid_size, args.starts)
            row["range_km"] = float(instance["R_chosen"])
            rows.append(row)
            rows.sort(key=lambda item: (int(item["seed"]), int(item["m"])))
            checkpoint(args.out, meta, rows, started)
            print(
                f"  m={m}: k*={row['k_opt']}, P_QWOA={row['qwoa']['p_opt']:.5g}, "
                f"P_QAOA={row['qaoa']['p_opt']:.5g}, ratio="
                f"{row['absolute_p_opt_ratio_qwoa_over_qaoa']:.3g}, "
                f"{time.time() - before:.1f}s"
            )
    checkpoint(args.out, meta, rows, started)
    print(f"-> {args.out} ({len(rows)} rows, {time.time() - started:.1f}s)")


if __name__ == "__main__":
    main()
