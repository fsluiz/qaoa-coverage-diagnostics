#!/usr/bin/env python3
"""RMS-normalized sensitivity analysis for the same-sector mixer control.

Raw shared-task weights have a larger global scale than unit edge weights.  A
free, unbounded beta would absorb that scale exactly, but the declared finite
window beta in [0, pi] need not.  This script divides the problem-informed
weights by their edge-weight RMS and repeats the identical p=1 convergence
protocol.  The primary support-matched unit control already has RMS one and is
read from ``same_sector_mixer_control.json``.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

import benchmark_qwoa_qaoa as bm
import run_same_sector_mixer_control as control
from _paths import RESULTS_DIR


CONTROL = RESULTS_DIR / "same_sector_mixer_control.json"
DEFAULT_OUTPUT = RESULTS_DIR / "mixer_scale_sensitivity.json"


def _source_rows() -> dict[tuple[int, int], dict]:
    payload = json.loads(CONTROL.read_text(encoding="utf-8"))
    return {
        (int(row["seed"]), int(row["m"])): row
        for row in payload["rows"]
    }


def run_case(seed: int, m: int, source: dict) -> dict:
    instance = bm.build_instance(min_task_degree=3, rng_seed=seed)
    A = np.asarray(instance["A_q"], dtype=int)
    n = A.shape[0]
    penalty = n + 1
    bm.LAM = float(penalty)
    pc = bm.precompute(A, m)
    k_opt, _ = bm.classical_optimum(pc)
    raw_edges = bm.cooperation_edges(A)
    values = np.asarray(list(raw_edges.values()), dtype=float)
    rms = float(np.sqrt(np.mean(values**2)))
    normalized_edges = {edge: weight / rms for edge, weight in raw_edges.items()}
    normalized_rms = float(np.sqrt(np.mean(np.asarray(list(normalized_edges.values())) ** 2)))
    sector = bm.QWOASector(A, normalized_edges, k_opt, pc)
    search = control._optimize_control(sector, penalty)

    support = source["controls"]["support_unweighted"]
    complete = source["controls"]["complete_unweighted"]
    comparisons = {}
    for name, comparator in (("support_unweighted", support), ("complete_unweighted", complete)):
        comparisons[f"rms_normalized_weighted_minus_{name}"] = {
            "expected_energy_difference": float(
                search["metrics"]["expected_energy"]
                - comparator["metrics"]["expected_energy"]
            ),
            "p_opt_difference": float(
                search["metrics"]["p_opt"] - comparator["metrics"]["p_opt"]
            ),
            "p_opt_ratio": float(
                search["metrics"]["p_opt"]
                / max(comparator["metrics"]["p_opt"], 1e-300)
            ),
        }
    return {
        "seed": seed,
        "m": m,
        "n": n,
        "k_opt": int(k_opt),
        "sector_dimension": int(sector.dim),
        "raw_weight_rms": rms,
        "normalized_weight_rms": normalized_rms,
        "normalized_weights": control._edge_records(normalized_edges),
        "rms_normalized_weighted": search,
        "source_controls": {
            "source": str(CONTROL.relative_to(RESULTS_DIR.parent)),
            "support_unweighted": {
                "parameters": support["parameters"],
                "metrics": support["metrics"],
                "stability": support["stability"],
            },
            "complete_unweighted": {
                "parameters": complete["parameters"],
                "metrics": complete["metrics"],
                "stability": complete["stability"],
            },
        },
        "comparisons": comparisons,
    }


def summarize(rows: list[dict]) -> dict:
    output = {}
    for comparator in ("support_unweighted", "complete_unweighted"):
        converged = [
            row for row in rows
            if row["rms_normalized_weighted"]["stability"]["joint_criterion_passed"]
            and row["source_controls"][comparator]["stability"]["joint_criterion_passed"]
        ]
        key = f"rms_normalized_weighted_minus_{comparator}"
        energy = np.asarray(
            [row["comparisons"][key]["expected_energy_difference"] for row in converged]
        )
        probability = np.asarray(
            [row["comparisons"][key]["p_opt_difference"] for row in converged]
        )
        ratios = np.asarray(
            [row["comparisons"][key]["p_opt_ratio"] for row in converged]
        )
        output[comparator] = {
            "n_cases": len(rows),
            "n_both_converged": len(converged),
            "normalized_weighted_lower_energy": int(np.sum(energy < 0.0)),
            "control_lower_energy": int(np.sum(energy > 0.0)),
            "normalized_weighted_larger_p_opt": int(np.sum(probability > 0.0)),
            "control_larger_p_opt": int(np.sum(probability < 0.0)),
            "median_normalized_weighted_minus_control_energy": (
                float(np.median(energy)) if len(energy) else None
            ),
            "energy_difference_range": (
                [float(energy.min()), float(energy.max())] if len(energy) else None
            ),
            "median_normalized_weighted_over_control_p_opt_ratio": (
                float(np.median(ratios)) if len(ratios) else None
            ),
            "p_opt_ratio_range": (
                [float(ratios.min()), float(ratios.max())] if len(ratios) else None
            ),
            "inference_status": (
                "descriptive_only: six multiplicity rows arise from two instance seeds"
            ),
        }
    return output


def checkpoint(
    output: Path,
    rows: list[dict],
    requested: dict,
    started: float,
    prior_runtime_s: float,
) -> None:
    payload = {
        "meta": {
            "description": "RMS-normalized same-sector mixer sensitivity analysis",
            "reason": (
                "remove the arbitrary global scale of shared-task edge weights while "
                "preserving their relative heterogeneity"
            ),
            "normalization": "w_ij divided by sqrt(mean over retained edges of w_ij^2)",
            "fixed_beta_window": [0.0, float(np.pi)],
            "optimizer_protocol": (
                "identical p=1 grid, SHGO resolutions, adaptive replication, and exact gradients"
            ),
            "requested": requested,
            "runtime_s": round(prior_runtime_s + time.time() - started, 3),
        },
        "rows": sorted(rows, key=lambda row: (row["seed"], row["m"])),
        "summary": summarize(rows),
    }
    output.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="12,17")
    parser.add_argument("--m-values", default="1,2,3")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    seeds = control._parse_int_spec(args.seeds)
    m_values = control._parse_int_spec(args.m_values)
    requested = {"seeds": seeds, "m_values": m_values, "p": 1}
    source_rows = _source_rows()

    started = time.time()
    rows = []
    prior_runtime_s = 0.0
    if args.out.exists() and not args.no_resume:
        previous = json.loads(args.out.read_text(encoding="utf-8"))
        rows = previous.get("rows", [])
        prior_runtime_s = float(previous.get("meta", {}).get("runtime_s", 0.0))
        print(f"resume: loaded {len(rows)} cases from {args.out}", flush=True)
    completed = {(int(row["seed"]), int(row["m"])) for row in rows}
    for seed, m in itertools.product(seeds, m_values):
        if (seed, m) in completed:
            print(f"seed={seed}, m={m}: already complete", flush=True)
            continue
        if (seed, m) not in source_rows:
            raise KeyError(f"missing source control for seed={seed}, m={m}")
        before = time.time()
        print(f"seed={seed}, m={m}: starting RMS-normalized weighted mixer", flush=True)
        row = run_case(seed, m, source_rows[(seed, m)])
        rows.append(row)
        checkpoint(args.out, rows, requested, started, prior_runtime_s)
        print(
            f"seed={seed}, m={m}: done in {time.time() - before:.1f}s; "
            f"converged={row['rms_normalized_weighted']['stability']['joint_criterion_passed']}",
            flush=True,
        )
    checkpoint(args.out, rows, requested, started, prior_runtime_s)
    print(f"-> {args.out} ({len(rows)} cases, {time.time() - started:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
