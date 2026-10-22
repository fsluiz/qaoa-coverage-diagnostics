#!/usr/bin/env python3
"""Same-sector control for the problem-weighted QWOA mixer.

The earlier QWOA/QAOA comparison changed the initial state, accessible Hilbert
space, and mixer at the same time.  This experiment holds the first two fixed:
every ansatz starts in the same Dicke state in the oracle-supplied ``k=k_opt``
sector and uses the same exact one-sided coverage cost.  Only the XY mixer is
changed.

Primary contrast
----------------
``problem_weighted`` uses the number of shared tasks as the cooperation-edge
weight.  ``support_unweighted`` keeps exactly the same edge support and replaces
every nonzero weight by one.  This isolates edge weighting while retaining the
same topology.

Secondary contrast
------------------
``complete_unweighted`` uses unit weights on the complete graph.  It removes
the remaining problem-informed topology.  When the cooperation support is
already complete, as it is for seed 12, this control is algebraically identical
to ``support_unweighted`` and its result is reused rather than recomputed.

The six cases and p=1 protocol were fixed by the preceding convergence audit.
The weighted endpoints are read from that audit's raw pilot; the two controls
are optimized with the identical dense-grid, two-resolution SHGO, exact-gradient
polish, and adaptive third-resolution rule.  P(opt) is evaluation-only.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

import benchmark_qwoa_qaoa as bm
import run_qwoa_convergence as convergence
from _paths import RESULTS_DIR


PILOT = RESULTS_DIR / "qwoa_convergence_pilot.json"
AUDIT = RESULTS_DIR / "qwoa_convergence_audit.json"
DEFAULT_OUTPUT = RESULTS_DIR / "same_sector_mixer_control.json"
P = 1


def _parse_int_spec(spec: str) -> list[int]:
    return convergence.parse_int_spec(spec)


def _complete_edges(n: int) -> dict[tuple[int, int], float]:
    return {edge: 1.0 for edge in itertools.combinations(range(n), 2)}


def _edge_records(edges: dict[tuple[int, int], float]) -> list[dict]:
    return [
        {"vertices": [int(left), int(right)], "weight": float(weight)}
        for (left, right), weight in sorted(edges.items())
    ]


def _mixer_metadata(
    label: str,
    edges: dict[tuple[int, int], float],
    n: int,
    reused_from: str | None = None,
) -> dict:
    values = np.asarray(list(edges.values()), dtype=float)
    return {
        "label": label,
        "n_edges": len(edges),
        "complete_graph_edges": n * (n - 1) // 2,
        "edge_density": float(len(edges) / (n * (n - 1) / 2)),
        "minimum_edge_weight": float(values.min()),
        "maximum_edge_weight": float(values.max()),
        "mean_edge_weight": float(values.mean()),
        "rms_edge_weight": float(np.sqrt(np.mean(values**2))),
        "edges": _edge_records(edges),
        "reused_from": reused_from,
    }


def _load_weighted_reference(seed: int, m: int) -> dict:
    pilot = json.loads(PILOT.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    raw = next(
        row for row in pilot["rows"]
        if int(row["seed"]) == seed and int(row["m"]) == m
    )["depths"]["1"]["qwoa"]
    audited = next(
        row for row in audit["p1"]["rows"]
        if int(row["seed"]) == seed and int(row["m"]) == m
    )
    if not audited["qwoa_converged_after_adaptive_check"]:
        raise AssertionError("weighted reference is not optimizer-converged")
    return {
        "source": str(PILOT.relative_to(RESULTS_DIR.parent)),
        "audit_source": str(AUDIT.relative_to(RESULTS_DIR.parent)),
        "parameters": raw["parameters"],
        "expected_energy": float(raw["metrics"]["expected_energy"]),
        "metrics": raw["metrics"],
        "stability": {
            **raw["stability"],
            "joint_criterion_passed_after_adaptive_audit": True,
            "adaptive_check_used": bool(audited["adaptive_check_used"]),
        },
        "method": raw["method"],
        "parameter_bounds": raw["parameter_bounds"],
    }


def _optimize_control(sector: bm.QWOASector, penalty: int) -> dict:
    probability = lambda parameters: sector.evolve(P, parameters)
    objective = lambda parameters: float(probability(parameters) @ sector.Esec)
    value_gradient = lambda parameters: convergence.qwoa_state_value_grad(
        sector, P, parameters
    )[1:]
    bounds = convergence.parameter_bounds("qwoa", P, penalty)
    search = convergence.p1_landscape_search(
        objective, value_gradient, bounds, include_adaptive=True
    )
    search["method"] = (
        "dense_grid_plus_two_resolution_simplicial_homology_global_optimization_"
        "with_adaptive_third_resolution"
    )
    search["parameter_bounds"] = bounds.tolist()
    return convergence.attach_run_metrics(
        search, probability, lambda probs: convergence.qwoa_metrics(probs, sector)
    )


def _matrix_checks(
    weighted: bm.QWOASector,
    support: bm.QWOASector,
    complete: bm.QWOASector,
    support_edges: dict[tuple[int, int], float],
    complete_edges: dict[tuple[int, int], float],
) -> dict:
    same_states = bool(
        np.array_equal(weighted.states, support.states)
        and np.array_equal(weighted.states, complete.states)
    )
    same_cost = bool(
        np.array_equal(weighted.Esec, support.Esec)
        and np.array_equal(weighted.Esec, complete.Esec)
    )
    expected_support = set(support_edges)
    expected_complete = set(complete_edges)
    return {
        "same_sector_basis": same_states,
        "same_sector_cost": same_cost,
        "same_uniform_dicke_initial_state": True,
        "support_unweighted_all_weights_one": all(
            weight == 1.0 for weight in support_edges.values()
        ),
        "complete_unweighted_all_weights_one": all(
            weight == 1.0 for weight in complete_edges.values()
        ),
        "support_edges_subset_of_complete": expected_support <= expected_complete,
        "weighted_mixer_hermitian_error": float(
            np.max(np.abs(
                weighted.evecs @ np.diag(weighted.evals) @ weighted.evecs.conj().T
                - (weighted.evecs @ np.diag(weighted.evals) @ weighted.evecs.conj().T).conj().T
            ))
        ),
        "support_mixer_hermitian_error": float(
            np.max(np.abs(
                support.evecs @ np.diag(support.evals) @ support.evecs.conj().T
                - (support.evecs @ np.diag(support.evals) @ support.evecs.conj().T).conj().T
            ))
        ),
        "complete_mixer_hermitian_error": float(
            np.max(np.abs(
                complete.evecs @ np.diag(complete.evals) @ complete.evecs.conj().T
                - (complete.evecs @ np.diag(complete.evals) @ complete.evecs.conj().T).conj().T
            ))
        ),
    }


def run_case(seed: int, m: int) -> dict:
    instance = bm.build_instance(min_task_degree=3, rng_seed=seed)
    A = np.asarray(instance["A_q"], dtype=int)
    n = A.shape[0]
    penalty = n + 1
    bm.LAM = float(penalty)
    pc = bm.precompute(A, m)
    k_opt, opt_states = bm.classical_optimum(pc)
    if k_opt is None:
        raise RuntimeError(f"seed={seed}, m={m}: strict instance is infeasible")

    weighted_edges = bm.cooperation_edges(A)
    support_edges = {edge: 1.0 for edge in weighted_edges}
    complete_edges = _complete_edges(n)
    weighted_sector = bm.QWOASector(A, weighted_edges, k_opt, pc)
    support_sector = bm.QWOASector(A, support_edges, k_opt, pc)
    support_search = _optimize_control(support_sector, penalty)

    if set(support_edges) == set(complete_edges):
        complete_sector = support_sector
        complete_search = deepcopy(support_search)
        complete_search["reused_from"] = "support_unweighted"
    else:
        complete_sector = bm.QWOASector(A, complete_edges, k_opt, pc)
        complete_search = _optimize_control(complete_sector, penalty)
        complete_search["reused_from"] = None

    weighted_reference = _load_weighted_reference(seed, m)
    controls = {
        "problem_weighted": weighted_reference,
        "support_unweighted": support_search,
        "complete_unweighted": complete_search,
    }
    comparisons = {}
    for control_name in ("support_unweighted", "complete_unweighted"):
        control = controls[control_name]
        comparisons[f"problem_weighted_minus_{control_name}"] = {
            "expected_energy_difference": float(
                weighted_reference["metrics"]["expected_energy"]
                - control["metrics"]["expected_energy"]
            ),
            "p_opt_difference": float(
                weighted_reference["metrics"]["p_opt"]
                - control["metrics"]["p_opt"]
            ),
            "p_opt_ratio": float(
                weighted_reference["metrics"]["p_opt"]
                / max(control["metrics"]["p_opt"], 1e-300)
            ),
        }

    return {
        "seed": seed,
        "m": m,
        "n": n,
        "n_tasks": int(A.shape[1]),
        "penalty": penalty,
        "k_opt": int(k_opt),
        "n_opt": int(len(opt_states)),
        "sector_dimension": int(weighted_sector.dim),
        "uniform_sector_p_opt": float(len(opt_states) / math.comb(n, k_opt)),
        "mixer_metadata": {
            "problem_weighted": _mixer_metadata(
                "shared-task-weighted cooperation graph", weighted_edges, n
            ),
            "support_unweighted": _mixer_metadata(
                "same cooperation support with unit weights", support_edges, n
            ),
            "complete_unweighted": _mixer_metadata(
                "complete graph with unit weights",
                complete_edges,
                n,
                "support_unweighted" if set(support_edges) == set(complete_edges) else None,
            ),
        },
        "matrix_checks": _matrix_checks(
            weighted_sector,
            support_sector,
            complete_sector,
            support_edges,
            complete_edges,
        ),
        "controls": controls,
        "comparisons": comparisons,
    }


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for control_name in ("support_unweighted", "complete_unweighted"):
        converged = [
            row for row in rows
            if row["controls"][control_name]["stability"]["joint_criterion_passed"]
        ]
        key = f"problem_weighted_minus_{control_name}"
        energy = np.asarray(
            [row["comparisons"][key]["expected_energy_difference"] for row in converged],
            dtype=float,
        )
        probability = np.asarray(
            [row["comparisons"][key]["p_opt_difference"] for row in converged],
            dtype=float,
        )
        ratios = np.asarray(
            [row["comparisons"][key]["p_opt_ratio"] for row in converged],
            dtype=float,
        )
        summary[control_name] = {
            "n_cases": len(rows),
            "n_control_converged": len(converged),
            "n_both_converged": len(converged),
            "problem_weighted_lower_energy": int(np.sum(energy < 0.0)),
            "control_lower_energy": int(np.sum(energy > 0.0)),
            "ties_within_1e-8": int(np.sum(np.abs(energy) <= 1e-8)),
            "problem_weighted_larger_p_opt": int(np.sum(probability > 0.0)),
            "control_larger_p_opt": int(np.sum(probability < 0.0)),
            "median_weighted_minus_control_energy": (
                float(np.median(energy)) if len(energy) else None
            ),
            "energy_difference_range": (
                [float(energy.min()), float(energy.max())] if len(energy) else None
            ),
            "median_weighted_over_control_p_opt_ratio": (
                float(np.median(ratios)) if len(ratios) else None
            ),
            "p_opt_ratio_range": (
                [float(ratios.min()), float(ratios.max())] if len(ratios) else None
            ),
            "inference_status": (
                "descriptive_only: six multiplicity rows arise from two instance seeds"
            ),
        }
    return summary


def checkpoint(
    output_path: Path,
    rows: list[dict],
    requested: dict,
    started: float,
    prior_runtime_s: float,
) -> None:
    payload = {
        "meta": {
            "description": "Same-Dicke-state, same-sector p=1 XY-mixer control",
            "scientific_status": "prespecified control following the QWOA convergence audit",
            "primary_contrast": (
                "shared-task-weighted cooperation graph versus the identical edge support "
                "with all weights equal to one"
            ),
            "secondary_contrast": (
                "shared-task-weighted cooperation graph versus the complete unit-weight graph"
            ),
            "fixed_elements": [
                "instance", "k_opt sector", "uniform Dicke initial state",
                "exact one-sided cost", "p=1", "optimizer protocol", "parameter bounds",
            ],
            "optimized_metric": "expected exact cost",
            "evaluation_only_metric": "probability of an optimal feasible string",
            "beta_window": [0.0, math.pi],
            "gamma_window": [0.0, 2.0 * math.pi / 13.0],
            "mixer_scale_note": (
                "Raw shared-task counts are the treatment; the unit-weight control sets every "
                "retained coupling to one. Beta is optimized over the same declared window."
            ),
            "requested": requested,
            "runtime_s": round(prior_runtime_s + time.time() - started, 3),
        },
        "rows": sorted(rows, key=lambda row: (row["seed"], row["m"])),
        "summary": summarize(rows),
    }
    output_path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="12,17")
    parser.add_argument("--m-values", default="1,2,3")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    seeds = _parse_int_spec(args.seeds)
    m_values = _parse_int_spec(args.m_values)
    requested = {"seeds": seeds, "m_values": m_values, "p": P}

    started = time.time()
    rows = []
    prior_runtime_s = 0.0
    if args.out.exists() and not args.no_resume:
        previous = json.loads(args.out.read_text(encoding="utf-8"))
        rows = previous.get("rows", [])
        prior_runtime_s = float(previous.get("meta", {}).get("runtime_s", 0.0))
        print(f"resume: loaded {len(rows)} complete cases from {args.out}", flush=True)
    completed = {(int(row["seed"]), int(row["m"])) for row in rows}
    for seed, m in itertools.product(seeds, m_values):
        if (seed, m) in completed:
            print(f"seed={seed}, m={m}: already complete", flush=True)
            continue
        before = time.time()
        print(f"seed={seed}, m={m}: starting same-sector controls", flush=True)
        row = run_case(seed, m)
        rows.append(row)
        checkpoint(args.out, rows, requested, started, prior_runtime_s)
        print(
            f"seed={seed}, m={m}: done in {time.time() - before:.1f}s; "
            f"support_converged={row['controls']['support_unweighted']['stability']['joint_criterion_passed']}; "
            f"complete_converged={row['controls']['complete_unweighted']['stability']['joint_criterion_passed']}",
            flush=True,
        )
    checkpoint(args.out, rows, requested, started, prior_runtime_s)
    print(f"-> {args.out} ({len(rows)} cases, {time.time() - started:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
