#!/usr/bin/env python3
"""
PATH B — referee-hardened QWOA-vs-QAOA comparison.

Addresses referee-alpha's three demands that Path A could not (data not stored):
  (1) MULTI-SEED ensemble  -> mean +/- std over independent task instances
                              (n=12 bases, m=3, task scatter varied by RNG_SEED).
  (2) lambda-OPTIMIZED QAOA -> sweep the penalty lambda for QAOA (it is lambda-
                              sensitive) and give it its best shot at each p.
  (3) EQUAL 2-QUBIT-GATE BUDGET -> compare QWOA(p<=4) vs QAOA at MATCHED CX count
                              (QAOA gets up to ~10 layers), not equal p; and report
                              the hardware-adjusted F_est * P(opt) with the paper's
                              own model F_est = (1-eps)^{n_2Q}.

Reuses the exact statevector engine from benchmark_qwoa_qaoa.py (pure NumPy/SciPy).
Writes path_b_results.json incrementally (one instance at a time) so partial runs
are usable.  Author: assistant for F. S. Luiz (IFGW/UNICAMP).
"""
from __future__ import annotations
import json, time, itertools, math
import numpy as np
from scipy.optimize import minimize

import benchmark_qwoa_qaoa as bm
from benchmark_qwoa_qaoa import (
    build_instance, precompute, classical_optimum, greedy_mcover,
    cooperation_edges, cx_counts, qaoa_run, qaoa_metrics, qwoa_metrics, QWOASector,
)
from _paths import RESULTS_DIR

# ---- ensemble / sweep settings ---------------------------------------------
# Eight independent task realizations that all yield an INTERIOR optimum
# (n=12, m=3, k_opt=9, sector dim 220) -> clean mean+/-std at fixed k_opt.
ENSEMBLE_SEEDS = [13, 14, 16, 17, 18, 19, 20, 22]
M = 3                                                # coverage multiplicity (resilient)
LAM_REF = 25.0                                        # QWOA + classical reference penalty
LAM_GRID = [10.0, 25.0, 50.0]                         # QAOA penalty sweep
P_QWOA = [1, 2, 3, 4]
P_QAOA = [1, 2, 3, 4, 6, 8, 10]                        # up to 10 layers for CX matching
N_STARTS = 5
LBFGS_MAXITER = 100
POLISH_MAXITER = 250
EPS_EFF = 0.006                                        # first-order per-2Q-gate error


def optimise(fn, p, rng, warm=None):
    bnds = [(-np.pi, np.pi)] * p + [(0.0, np.pi)] * p
    starts = [np.concatenate([rng.uniform(-np.pi, np.pi, p),
                              rng.uniform(0.0, np.pi / 2, p)]) for _ in range(N_STARTS)]
    if warm is not None and len(warm) == 2 * p:
        starts.append(np.asarray(warm, float))
    best_val, best_par = np.inf, None
    for x0 in starts:
        r = minimize(fn, x0, method="L-BFGS-B", bounds=bnds,
                     options=dict(maxiter=LBFGS_MAXITER, ftol=1e-9, gtol=1e-8))
        if r.fun < best_val:
            best_val, best_par = float(r.fun), r.x
    r = minimize(fn, best_par, method="Nelder-Mead",
                 options=dict(maxiter=POLISH_MAXITER, xatol=1e-5, fatol=1e-8))
    if r.fun < best_val:
        best_val, best_par = float(r.fun), r.x
    return best_par, best_val


def _extend(prev, p):
    g, b = prev[:p - 1], prev[p - 1:]
    return np.concatenate([g, g[-1:], b, b[-1:]])


def run_instance(seed):
    bm.RNG_SEED = seed
    bm.LAM = LAM_REF
    inst = build_instance()
    A_q = np.asarray(inst["A_q"], dtype=int)
    n = A_q.shape[0]
    edges = cooperation_edges(A_q); n_edges_xy = len(edges)
    n_ZZ = sum(1 for b1, b2 in itertools.combinations(range(n), 2)
               if int(np.sum((A_q[b1] == 1) & (A_q[b2] == 1))) > 0)
    pc_ref = precompute(A_q, M)
    k_opt, opt_states = classical_optimum(pc_ref)
    k_greedy, _ = greedy_mcover(A_q, M)
    from math import comb
    p_rnd = len(opt_states) / comb(n, k_opt)          # in-sector random baseline

    rng = np.random.RandomState(0)

    # ---- QWOA-direct at k_opt: p-sweep, pick best P(opt) ----
    sec = QWOASector(A_q, edges, k_opt, pc_ref)
    qwoa_rows, prev = [], None
    for p in P_QWOA:
        warm = _extend(prev, p) if prev is not None else None
        fn = lambda par, _s=sec, _p=p: float((_s.evolve(_p, par) * _s.Esec).sum())
        par, _ = optimise(fn, p, rng, warm); prev = par
        mt = qwoa_metrics(sec.evolve(p, par), sec, k_opt)
        cx = cx_counts(n_ZZ, n_edges_xy, k_opt, n, p, "QWOA")
        qwoa_rows.append(dict(p=p, p_opt=mt["p_opt"], cx=cx))
    qwoa_best = max(qwoa_rows, key=lambda r: r["p_opt"])

    # ---- QAOA: lambda-sweep x p-sweep, measured vs the TRUE optimum ----
    qaoa_rows = []
    for lam in LAM_GRID:
        bm.LAM = lam
        pc_l = precompute(A_q, M)
        prev = None
        for p in P_QAOA:
            warm = _extend(prev, p) if prev is not None else None
            fn = lambda par, _p=p, _E=pc_l["energy"]: float((qaoa_run(pc_l, n, _p, par) * _E).sum())
            par, _ = optimise(fn, p, rng, warm); prev = par
            mt = qaoa_metrics(qaoa_run(pc_l, n, p, par), pc_ref, opt_states)
            cx = cx_counts(n_ZZ, n_edges_xy, k_opt, n, p, "QAOA")
            qaoa_rows.append(dict(lam=lam, p=p, p_opt=mt["p_opt"], cx=cx,
                                  exp_k_feas=mt["exp_k_feas"]))
    bm.LAM = LAM_REF
    qaoa_best = max(qaoa_rows, key=lambda r: r["p_opt"])                      # any p, any lam
    budget = qwoa_best["cx"]
    eqcx = [r for r in qaoa_rows if r["cx"] <= budget]
    qaoa_eqcx = max(eqcx, key=lambda r: r["p_opt"]) if eqcx else qaoa_best    # at equal CX

    def fp(row): return ((1 - EPS_EFF) ** row["cx"]) * row["p_opt"]

    return dict(
        seed=seed, n=n, m=M, R_km=inst["R_chosen"], n_ZZ=n_ZZ, k_opt=k_opt,
        k_greedy=k_greedy, n_opt=len(opt_states), p_rnd=p_rnd,
        qwoa_best=qwoa_best, qaoa_best=qaoa_best, qaoa_eqcx=qaoa_eqcx,
        qwoa_over_rnd=qwoa_best["p_opt"] / p_rnd,
        noiseless_ratio=qwoa_best["p_opt"] / max(qaoa_best["p_opt"], 1e-300),
        eqcx_ratio=qwoa_best["p_opt"] / max(qaoa_eqcx["p_opt"], 1e-300),
        Fest_qwoa=(1 - EPS_EFF) ** qwoa_best["cx"],
        Fest_qaoa_eqcx=(1 - EPS_EFF) ** qaoa_eqcx["cx"],
        FP_qwoa=fp(qwoa_best), FP_qaoa_eqcx=fp(qaoa_eqcx),
        FP_ratio=fp(qwoa_best) / max(fp(qaoa_eqcx), 1e-300),
    )


def summarize(rows):
    def ms(key):
        v = np.array([r[key] for r in rows], float)
        return float(v.mean()), float(v.std())
    keys = ["qwoa_over_rnd", "noiseless_ratio", "eqcx_ratio", "FP_ratio"]
    return {k: dict(mean=ms(k)[0], std=ms(k)[1]) for k in keys}


def main():
    t0 = time.time()
    out = dict(meta=dict(ensemble_seeds=ENSEMBLE_SEEDS, n_qubits=bm.N_QUBITS, m=M,
                         lam_ref=LAM_REF, lam_grid=LAM_GRID, P_QWOA=P_QWOA,
                         P_QAOA=P_QAOA, N_STARTS=N_STARTS, eps_eff=EPS_EFF),
               instances=[])
    print(f"PATH B  n={bm.N_QUBITS} m={M}  seeds={ENSEMBLE_SEEDS}")
    print(f"{'seed':>4} {'R':>5} {'nZZ':>4} {'kopt':>4} | "
          f"{'QWOA_Popt':>10} {'QAOA_best':>10} {'QAOA_eqCX':>10} | "
          f"{'noiseless':>9} {'eqCX':>7} {'F*P':>6}")
    for s in ENSEMBLE_SEEDS:
        r = run_instance(s)
        out["instances"].append(r)
        out["summary"] = summarize(out["instances"])
        out["meta"]["runtime_s"] = round(time.time() - t0, 1)
        json.dump(out, open(RESULTS_DIR / "path_b_results.json", "w"), indent=1)
        print(f"{s:>4} {r['R_km']:>5.0f} {r['n_ZZ']:>4} {r['k_opt']:>4} | "
              f"{r['qwoa_best']['p_opt']:>10.3e} {r['qaoa_best']['p_opt']:>10.3e} "
              f"{r['qaoa_eqcx']['p_opt']:>10.3e} | "
              f"{r['noiseless_ratio']:>8.0f}x {r['eqcx_ratio']:>6.1f}x {r['FP_ratio']:>5.2f}x")
    sm = out["summary"]
    print("\nENSEMBLE (mean +/- std over %d instances):" % len(out["instances"]))
    print(f"  QWOA/random           : {sm['qwoa_over_rnd']['mean']:.1f} +/- {sm['qwoa_over_rnd']['std']:.1f} x")
    print(f"  noiseless QWOA/QAOA    : {sm['noiseless_ratio']['mean']:.0f} +/- {sm['noiseless_ratio']['std']:.0f} x")
    print(f"  EQUAL-CX QWOA/QAOA     : {sm['eqcx_ratio']['mean']:.1f} +/- {sm['eqcx_ratio']['std']:.1f} x")
    print(f"  hardware F*P QWOA/QAOA : {sm['FP_ratio']['mean']:.2f} +/- {sm['FP_ratio']['std']:.2f} x")
    print(f"\nDone in {out['meta']['runtime_s']}s -> path_b_results.json")


if __name__ == "__main__":
    main()
