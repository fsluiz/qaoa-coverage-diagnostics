#!/usr/bin/env python3
"""Twenty-seed extension of the p=1 QWOA/QAOA audit and same-sector mixer control.

The v3 pilot used two task seeds (12, 17), so its six multiplicity rows were descriptive
only.  This script repeats the complete p=1 protocol on twenty seeds chosen by the same
prespecified rule, extended: ``default_rng(20260915).choice(24, 20, replace=False)``,
which contains the two pilot seeds.  For every (seed, m) with m = 1, 2, 3 it optimises,
with the identical dense-grid + two-resolution SHGO + adaptive-third-resolution protocol
of ``run_qwoa_convergence.p1_landscape_search``:

  qaoa            full-register QAOA, transverse-field mixer
  weighted        QWOA, raw shared-task-weighted XY mixer, k* Dicke state
  weighted_rms    the same mixer divided by its edge-weight RMS
  support         unit weights on the weighted mixer's edge support
  complete        unit weights on the complete graph

All five minimise the exact expected one-sided cost; P(opt) is evaluation-only.  The
scale-equivalence diagnostic ``beta_raw * RMS <= pi`` is recorded for every row: when it
holds, the raw optimum is reachable inside the normalised mixer's beta window and the two
endpoints must coincide.

Writes ``results/qwoa_20seeds.json``.  Pure NumPy/SciPy, parallel over (seed, m).  An
existing output is resumed, which with all 60 rows present only recomputes the summary;
``--fresh`` recomputes every row.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from scipy.stats import binomtest, wilcoxon

import benchmark_qwoa_qaoa as bm
import run_qwoa_convergence as convergence
from _paths import RESULTS_DIR

SELECTION_SEED = 20260915
N_SEEDS = 20
SEEDS = np.random.default_rng(SELECTION_SEED).choice(24, size=N_SEEDS, replace=False).tolist()
M_VALUES = (1, 2, 3)
P = 1
OUT = RESULTS_DIR / "qwoa_20seeds.json"


def _qwoa_search(sector: bm.QWOASector, penalty: int) -> dict:
    probability = lambda x: sector.evolve(P, x)
    objective = lambda x: float(probability(x) @ sector.Esec)
    value_grad = lambda x: convergence.qwoa_state_value_grad(sector, P, x)[1:]
    bounds = convergence.parameter_bounds("qwoa", P, penalty)
    search = convergence.p1_landscape_search(objective, value_grad, bounds,
                                             include_adaptive=True)
    search = convergence.attach_run_metrics(
        search, probability, lambda pr: convergence.qwoa_metrics(pr, sector))
    return _compact(search)


def _qaoa_search(pc: dict, n: int, penalty: int, opt_states, k_opt: int) -> dict:
    probability = lambda x: bm.qaoa_run(pc, n, P, x)
    objective = lambda x: float(probability(x) @ pc["energy"])
    value_grad = lambda x: convergence.qaoa_state_value_grad(pc, n, P, x)[1:]
    bounds = convergence.parameter_bounds("qaoa", P, penalty)
    search = convergence.p1_landscape_search(objective, value_grad, bounds,
                                             include_adaptive=True)
    search = convergence.attach_run_metrics(
        search, probability,
        lambda pr: convergence.qaoa_metrics(pr, pc, opt_states, k_opt))
    return _compact(search)


def _compact(search: dict) -> dict:
    """Keep what the paper needs; drop landscape dumps."""
    return {
        "parameters": search["parameters"],
        "expected_energy": search["metrics"]["expected_energy"],
        "metrics": search["metrics"],
        "stability": search["stability"],
        "shgo_levels": search["shgo_levels"],
        "endpoints": [
            {"repeat": r["repeat"], "parameters": r["parameters"],
             "expected_energy": r["metrics"]["expected_energy"],
             "p_opt": r["metrics"]["p_opt"]}
            for r in search["refinements"]
        ],
    }


def run_case(args: tuple[int, int]) -> dict:
    seed, m = args
    t0 = time.time()
    instance = bm.build_instance(min_task_degree=3, rng_seed=seed)
    A = np.asarray(instance["A_q"], dtype=int)
    n = A.shape[0]
    penalty = n + 1
    bm.LAM = float(penalty)
    pc = bm.precompute(A, m)
    k_opt, opt_states = bm.classical_optimum(pc)
    if k_opt is None:
        return {"seed": seed, "m": m, "status": "infeasible"}
    opt_states = np.asarray(opt_states, dtype=int)

    weighted_edges = bm.cooperation_edges(A)
    w = np.asarray(list(weighted_edges.values()), dtype=float)
    rms = float(np.sqrt(np.mean(w ** 2)))
    edges = {
        "weighted": weighted_edges,
        "weighted_rms": {e: v / rms for e, v in weighted_edges.items()},
        "support": {e: 1.0 for e in weighted_edges},
        "complete": {e: 1.0 for e in itertools.combinations(range(n), 2)},
    }
    searches = {}
    for name, edge_set in edges.items():
        if name == "complete" and set(edge_set) == set(edges["support"]):
            searches[name] = dict(searches["support"], reused_from="support")
            continue
        sector = bm.QWOASector(A, edge_set, k_opt, pc)
        searches[name] = _qwoa_search(sector, penalty)
    searches["qaoa"] = _qaoa_search(pc, n, penalty, opt_states, k_opt)

    beta_raw = float(searches["weighted"]["parameters"][1])
    return {
        "seed": seed, "m": m, "status": "ok", "n": n, "n_tasks": int(A.shape[1]),
        "range_km": float(instance["R_chosen"]), "penalty": penalty,
        "k_opt": int(k_opt), "n_opt": int(len(opt_states)),
        "sector_dimension": math.comb(n, k_opt),
        "uniform_sector_p_opt": len(opt_states) / math.comb(n, k_opt),
        "raw_weight_rms": rms,
        "n_support_edges": len(edges["support"]),
        "beta_raw_times_rms": beta_raw * rms,
        "raw_optimum_inside_normalised_window": bool(beta_raw * rms <= math.pi),
        "searches": searches,
        "runtime_s": round(time.time() - t0, 2),
    }


def _pair_stats(rows: list[dict], a: str, b: str) -> dict:
    """Paired comparison of ansatz a against control b over converged rows."""
    use = [r for r in rows
           if r["searches"][a]["stability"]["joint_criterion_passed"]
           and r["searches"][b]["stability"]["joint_criterion_passed"]]
    dE = np.array([r["searches"][a]["expected_energy"] - r["searches"][b]["expected_energy"]
                   for r in use])
    ratio = np.array([r["searches"][a]["metrics"]["p_opt"]
                      / max(r["searches"][b]["metrics"]["p_opt"], 1e-300) for r in use])
    lr = np.log(ratio)
    out = {
        "n_pairs": len(use),
        "a_lower_energy": int(np.sum(dE < -1e-8)),
        "b_lower_energy": int(np.sum(dE > 1e-8)),
        "energy_ties": int(np.sum(np.abs(dE) <= 1e-8)),
        "a_larger_p_opt": int(np.sum(lr > 1e-8)),
        "b_larger_p_opt": int(np.sum(lr < -1e-8)),
        "p_opt_ties": int(np.sum(np.abs(lr) <= 1e-8)),
        "median_energy_difference": float(np.median(dE)) if len(dE) else None,
        "median_p_opt_ratio": float(np.exp(np.median(lr))) if len(lr) else None,
        "p_opt_ratio_iqr": ([float(np.exp(np.percentile(lr, 25))),
                             float(np.exp(np.percentile(lr, 75)))] if len(lr) else None),
        "p_opt_ratio_range": ([float(ratio.min()), float(ratio.max())] if len(lr) else None),
    }
    nz = lr[np.abs(lr) > 1e-8]
    if len(nz):
        out["sign_test_p_opt_two_sided"] = float(
            binomtest(int(np.sum(nz > 0)), len(nz), 0.5).pvalue)
        out["wilcoxon_log_ratio_two_sided"] = float(wilcoxon(nz).pvalue)
    nzE = dE[np.abs(dE) > 1e-8]
    if len(nzE):
        out["sign_test_energy_two_sided"] = float(
            binomtest(int(np.sum(nzE < 0)), len(nzE), 0.5).pvalue)
        out["wilcoxon_energy_two_sided"] = float(wilcoxon(nzE).pvalue)
    return out


def _seed_cluster_stats(rows: list[dict], a: str, b: str) -> dict:
    """Seed-level paired test: the three multiplicities of one seed are not independent,
    so each seed contributes its mean energy difference and mean log P(opt) ratio."""
    dE, lr = [], []
    for seed in sorted({r["seed"] for r in rows}):
        use = [r for r in rows if r["seed"] == seed
               and r["searches"][a]["stability"]["joint_criterion_passed"]
               and r["searches"][b]["stability"]["joint_criterion_passed"]]
        if not use:
            continue
        dE.append(np.mean([r["searches"][a]["expected_energy"]
                           - r["searches"][b]["expected_energy"] for r in use]))
        lr.append(np.mean([np.log(r["searches"][a]["metrics"]["p_opt"]
                                  / r["searches"][b]["metrics"]["p_opt"]) for r in use]))
    dE, lr = np.asarray(dE), np.asarray(lr)
    return {
        "n_seeds": len(dE),
        "a_lower_energy_seeds": int(np.sum(dE < 0)),
        "a_larger_p_opt_seeds": int(np.sum(lr > 0)),
        "median_seed_energy_difference": float(np.median(dE)),
        "median_seed_p_opt_ratio": float(np.exp(np.median(lr))),
        "sign_test_energy_two_sided": float(binomtest(int(np.sum(dE < 0)), len(dE), 0.5).pvalue),
        "wilcoxon_energy_two_sided": float(wilcoxon(dE).pvalue),
        "sign_test_p_opt_two_sided": float(binomtest(int(np.sum(lr > 0)), len(lr), 0.5).pvalue),
        "wilcoxon_p_opt_two_sided": float(wilcoxon(lr).pvalue),
    }


def _seed_bootstrap_median_ratio(rows: list[dict], a: str, b: str, n_boot: int = 10000,
                                 seed: int = 20260930) -> list[float]:
    """Bootstrap 95% interval of the median seed-level P(opt) ratio, resampling seeds."""
    lr = []
    for s_ in sorted({r["seed"] for r in rows}):
        use = [r for r in rows if r["seed"] == s_
               and r["searches"][a]["stability"]["joint_criterion_passed"]
               and r["searches"][b]["stability"]["joint_criterion_passed"]]
        if use:
            lr.append(np.mean([np.log(r["searches"][a]["metrics"]["p_opt"]
                                      / r["searches"][b]["metrics"]["p_opt"]) for r in use]))
    lr = np.asarray(lr)
    rng = np.random.default_rng(seed)
    med = [np.median(rng.choice(lr, len(lr), replace=True)) for _ in range(n_boot)]
    return [float(np.exp(np.percentile(med, 2.5))), float(np.exp(np.percentile(med, 97.5)))]


def _holm(pvals: dict) -> dict:
    """Holm-adjusted p values for a family of tests given as {name: p}."""
    names = sorted(pvals, key=pvals.get)
    out, run = {}, 0.0
    for i, k in enumerate(names):
        run = max(run, min(1.0, (len(names) - i) * pvals[k]))
        out[k] = run
    return out


def summarise(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["status"] == "ok"]
    conv = {name: sum(r["searches"][name]["stability"]["joint_criterion_passed"] for r in ok)
            for name in ("qaoa", "weighted", "weighted_rms", "support", "complete")}
    pk = np.array([r["searches"]["qaoa"]["metrics"]["p_k_opt"] for r in ok])
    unc = np.array([r["searches"]["weighted"]["metrics"]["p_opt"]
                    / max(r["searches"]["qaoa"]["metrics"]["p_opt"], 1e-300) for r in ok])
    cond = np.array([r["searches"]["weighted"]["metrics"]["p_opt"]
                     / max(r["searches"]["qaoa"]["metrics"]["p_opt_given_k_opt"], 1e-300)
                     for r in ok])
    summary = {
        "n_rows": len(rows), "n_ok": len(ok),
        "n_independent_seeds": len({r["seed"] for r in ok}),
        "jointly_converged": conv,
        "scale_equivalent_rows": int(sum(r["raw_optimum_inside_normalised_window"] for r in ok)),
        "qwoa_vs_qaoa": {
            "qaoa_lower_energy": int(sum(r["searches"]["qaoa"]["expected_energy"]
                                         < r["searches"]["weighted"]["expected_energy"]
                                         for r in ok)),
            "median_unconditional_ratio": float(np.median(unc)),
            "unconditional_ratio_range": [float(unc.min()), float(unc.max())],
            "median_conditional_ratio": float(np.median(cond)),
            "conditional_ratio_range": [float(cond.min()), float(cond.max())],
            "qwoa_conditional_wins": int(np.sum(cond > 1)),
            "median_qaoa_p_k_opt": float(np.median(pk)),
            "qaoa_p_k_opt_range": [float(pk.min()), float(pk.max())],
        },
        "weighted_vs_support": _pair_stats(ok, "weighted", "support"),
        "weighted_rms_vs_support": _pair_stats(ok, "weighted_rms", "support"),
        "weighted_rms_vs_complete": _pair_stats(ok, "weighted_rms", "complete"),
        "weighted_vs_complete": _pair_stats(ok, "weighted", "complete"),
        "seed_clustered": {
            f"{a}_vs_{b}": _seed_cluster_stats(ok, a, b)
            for a, b in (("weighted_rms", "support"), ("weighted", "support"),
                         ("weighted_rms", "complete"), ("weighted", "complete"))
        },
        "by_m": {},
    }
    for m in M_VALUES:
        sub = [r for r in ok if r["m"] == m]
        summary["by_m"][str(m)] = {
            "weighted_rms_vs_support": _pair_stats(sub, "weighted_rms", "support"),
            "weighted_vs_support": _pair_stats(sub, "weighted", "support"),
        }
    # the per-m row-level tests form a family of 2 contrasts x 3 m x 2 metrics
    fam = {f"m{m}_{c}_{t}": summary["by_m"][str(m)][c][k]
           for m in M_VALUES for c in ("weighted_rms_vs_support", "weighted_vs_support")
           for t, k in (("energy", "wilcoxon_energy_two_sided"),
                        ("p_opt", "wilcoxon_log_ratio_two_sided"))}
    summary["by_m_holm"] = {"raw": fam, "holm": _holm(fam)}
    # sensitivity: the two pilot seeds fixed the protocol, so drop them
    pilots = {12, 17}
    rest = [r for r in ok if r["seed"] not in pilots]
    summary["excluding_pilot_seeds"] = {
        "pilot_seeds": sorted(pilots),
        "weighted_rms_vs_support": _seed_cluster_stats(rest, "weighted_rms", "support"),
        "median_ratio_ci95": _seed_bootstrap_median_ratio(rest, "weighted_rms", "support"),
    }
    summary["seed_clustered"]["weighted_rms_vs_support"]["median_ratio_ci95"] = \
        _seed_bootstrap_median_ratio(ok, "weighted_rms", "support")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--fresh", action="store_true",
                    help="ignore an existing output and recompute every row (~5 h CPU)")
    a = ap.parse_args()
    rows: list[dict] = []
    if a.out.exists() and not a.fresh:
        rows = json.loads(a.out.read_text())["rows"]
    done = {(r["seed"], r["m"]) for r in rows}
    todo = [(s, m) for s in SEEDS for m in M_VALUES if (s, m) not in done]
    print(f"seeds={SEEDS}; {len(done)} done, {len(todo)} to run on {a.workers} workers",
          flush=True)
    t0 = time.time()

    def dump():
        payload = {
            "meta": {
                "description": "20-seed p=1 QWOA/QAOA audit and same-sector mixer control",
                "selection_rule": "default_rng(20260915).choice(24, 20, replace=False)",
                "seeds": SEEDS, "m_values": list(M_VALUES), "p": P,
                "objective": "expected |x| + (n+1) * #violated tasks (strict m-cover)",
                "p_opt_role": "evaluation only",
                "beta_window": [0.0, math.pi],
                "protocol": "run_qwoa_convergence.p1_landscape_search, include_adaptive=True",
            },
            "rows": sorted(rows, key=lambda r: (r["seed"], r["m"])),
        }
        if rows:
            payload["summary"] = summarise(rows)
        a.out.write_text(json.dumps(payload, indent=1) + "\n")

    with Pool(a.workers) as pool:
        for row in pool.imap_unordered(run_case, todo):
            rows.append(row)
            dump()
            print(f"seed={row['seed']} m={row['m']} {row['status']} "
                  f"({row.get('runtime_s')}s; {len(rows)}/{len(SEEDS) * 3}; "
                  f"{time.time() - t0:.0f}s elapsed)", flush=True)
    dump()
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
