#!/usr/bin/env python3
"""
Strong-optimizer 3-way comparison on the greedy-OVERSHOOT instance
(overshoot_instance.json, n=14, m=3, greedy=12, k_opt=9):

  (1) QAOA            -- full 2^n space, X-mixer + |+>^n.
  (2) QWOA-direct     -- fixed-Hamming-weight sector, COLD multistart at EACH k,
                         NO classical greedy input, NO cross-sector transfer.
  (3) QWOA-descent    -- same sectors, but WARM-STARTED down the ladder
                         (best angles at k carried to k-1), seeded by greedy's k.

Goal: (a) clean QWOA-vs-QAOA head-to-head at k_opt under a strong optimizer, and
(b) settle whether the warm-started descent genuinely beats cold QWOA (the 12x gap
seen earlier may have been Nelder-Mead variance).

Stronger optimizer than benchmark_qwoa_qaoa.py:
  * L-BFGS-B (2-point FD gradient) -- exploits the smoothness of <E>(angles);
  * N_STARTS random restarts + INTERP warm-start (p-1 -> p) + (descent) cross-k seed;
  * final Nelder-Mead polish on the best start.

Reuses the exact simulation engine from benchmark_qwoa_qaoa.py.
Pure NumPy/SciPy.  Author: assistant for F. S. Luiz (IFGW/UNICAMP).
"""
from __future__ import annotations
import json
import time
import itertools

import numpy as np
from scipy.optimize import minimize

from benchmark_qwoa_qaoa import (
    precompute, classical_optimum, greedy_mcover, ilp_mcover,
    cooperation_edges, cx_counts, qaoa_run, qaoa_metrics, qwoa_metrics,
    QWOASector,
)
from _paths import DATA_DIR, RESULTS_DIR

# ---- strong-optimizer settings ---------------------------------------------
P_LIST = [1, 2, 3, 4]
N_STARTS = 12            # random restarts per (ansatz, p, sector)  [was 8]
LBFGS_MAXITER = 200      # L-BFGS-B iterations per start
POLISH_MAXITER = 1500    # Nelder-Mead polish on the best start    [was 200]


def _bounds(p):
    return [(-np.pi, np.pi)] * p + [(0.0, np.pi)] * p   # gammas, betas


def _interp_extend(prev, p):
    """Extend best (p-1)-params to length-2p by repeating the last layer."""
    g, b = prev[: p - 1], prev[p - 1:]
    return np.concatenate([g, g[-1:], b, b[-1:]])


def optimise_strong(fn, p, rng, warm_starts=None):
    """L-BFGS-B multistart + Nelder-Mead polish. `warm_starts`: list of seed
    vectors (each len 2p) to add to the random restarts. Returns (best_par, best_val)."""
    bnds = _bounds(p)
    starts = [np.concatenate([rng.uniform(-np.pi, np.pi, p),
                              rng.uniform(0.0, np.pi / 2, p)])
              for _ in range(N_STARTS)]
    for w in (warm_starts or []):
        if w is not None and len(w) == 2 * p:
            starts.append(np.asarray(w, float))
    best_val, best_par = np.inf, None
    for x0 in starts:
        res = minimize(fn, x0, method="L-BFGS-B", bounds=bnds,
                       options=dict(maxiter=LBFGS_MAXITER, ftol=1e-10, gtol=1e-8))
        if res.fun < best_val:
            best_val, best_par = float(res.fun), res.x
    # polish the global best
    res = minimize(fn, best_par, method="Nelder-Mead",
                   options=dict(maxiter=POLISH_MAXITER, xatol=1e-5, fatol=1e-8))
    if res.fun < best_val:
        best_val, best_par = float(res.fun), res.x
    return best_par, best_val


def main():
    t0 = time.time()
    h = json.load(open(DATA_DIR / "overshoot_instance.json"))
    A_q = np.array(h["A_q"], dtype=int)
    m = int(h["m"]); n = A_q.shape[0]
    edges = cooperation_edges(A_q); n_edges_xy = len(edges)
    n_ZZ = int(np.sum([1 for (b1, b2) in itertools.combinations(range(n), 2)
                       if int(np.sum((A_q[b1] == 1) & (A_q[b2] == 1))) > 0]))

    pc = precompute(A_q, m)
    k_opt, opt_states = classical_optimum(pc)
    k_greedy, _ = greedy_mcover(A_q, m)
    k_ilp, _, _ = ilp_mcover(A_q, m)
    print(f"Instance n={n}, {A_q.shape[1]} tasks, m={m}, n_ZZ={n_ZZ}, XY-edges={n_edges_xy}")
    print(f"classical: greedy={k_greedy}, ILP={k_ilp}, brute k_opt={k_opt} "
          f"(#opt={len(opt_states)}), OVERSHOOT={k_greedy-k_opt}")
    print(f"settings: N_STARTS={N_STARTS}, L-BFGS maxiter={LBFGS_MAXITER}, "
          f"polish NM maxiter={POLISH_MAXITER}, P_LIST={P_LIST}\n")

    rng = np.random.RandomState(0)
    sec_cache: dict[int, QWOASector] = {}
    def sector(k):
        if k not in sec_cache:
            sec_cache[k] = QWOASector(A_q, edges, k, pc)
        return sec_cache[k]

    res = dict(meta=dict(source="overshoot_instance.json", n=n, m=m, n_ZZ=n_ZZ,
                         n_edges_xy=n_edges_xy, k_greedy=k_greedy, k_ilp=k_ilp,
                         k_opt=k_opt, n_opt=len(opt_states), P_LIST=P_LIST,
                         N_STARTS=N_STARTS, lbfgs_maxiter=LBFGS_MAXITER,
                         polish_maxiter=POLISH_MAXITER),
               qaoa=[], qwoa_direct={}, qwoa_descent={})

    # ----- (1) QAOA: full-space p-sweep (INTERP warm only) -------------------
    print("=" * 70 + f"\n (1) QAOA  (full 2^{n} space)\n" + "=" * 70)
    prev = None
    for p in P_LIST:
        warm = [_interp_extend(prev, p)] if prev is not None else None
        fn = (lambda par, _p=p: float((qaoa_run(pc, n, _p, par) * pc["energy"]).sum()))
        par, _ = optimise_strong(fn, p, rng, warm); prev = par
        mt = qaoa_metrics(qaoa_run(pc, n, p, par), pc, opt_states)
        cx = cx_counts(n_ZZ, n_edges_xy, k_opt, n, p, "QAOA"); mt.update(p=p, cx=cx)
        res["qaoa"].append(mt)
        print(f"  p={p}: P(opt)={mt['p_opt']:.3e}  P(feas)={mt['p_feas']:.3f}  "
              f"<k|feas>={mt['exp_k_feas']:.2f}  CX={cx}")

    # ----- (2) QWOA-direct: COLD multistart at each k -----------------------
    print("\n" + "=" * 70 + "\n (2) QWOA-direct  (cold, no greedy, no cross-k transfer)\n" + "=" * 70)
    for k in range(k_greedy, k_opt - 1, -1):
        sec = sector(k); prev = None; rows = []
        for p in P_LIST:
            warm = [_interp_extend(prev, p)] if prev is not None else None
            fn = (lambda par, _s=sec, _p=p: float((_s.evolve(_p, par) * _s.Esec).sum()))
            par, _ = optimise_strong(fn, p, rng, warm); prev = par
            mt = qwoa_metrics(sec.evolve(p, par), sec, k_opt)
            cx = cx_counts(n_ZZ, n_edges_xy, k, n, p, "QWOA")
            mt.update(p=p, cx=cx, par=par.tolist()); rows.append(mt)
        res["qwoa_direct"][str(k)] = dict(dim=sec.dim, n_feas=int(sec.feas_sec.sum()), p_sweep=rows)
        best = max(rows, key=lambda r: r["p_feas"])
        print(f"  k={k} (dim={sec.dim}, #feas={int(sec.feas_sec.sum())}): "
              f"best P(feas)={best['p_feas']:.3f} P(opt)={best['p_opt']:.3e} @p={best['p']}")

    # ----- (3) QWOA-descent: WARM-STARTED down the ladder -------------------
    print("\n" + "=" * 70 + "\n (3) QWOA-descent  (warm-started k_greedy -> k_opt)\n" + "=" * 70)
    prev_k_par = {}                       # best params at previous (higher) k, by p
    for k in range(k_greedy, k_opt - 1, -1):
        sec = sector(k); prev = None; rows = []; best_par_p = {}
        for p in P_LIST:
            warm = []
            if prev is not None:
                warm.append(_interp_extend(prev, p))         # INTERP within sector
            if p in prev_k_par:
                warm.append(prev_k_par[p])                   # transfer from k+1, same p
            fn = (lambda par, _s=sec, _p=p: float((_s.evolve(_p, par) * _s.Esec).sum()))
            par, _ = optimise_strong(fn, p, rng, warm or None); prev = par
            best_par_p[p] = par
            mt = qwoa_metrics(sec.evolve(p, par), sec, k_opt)
            cx = cx_counts(n_ZZ, n_edges_xy, k, n, p, "QWOA")
            mt.update(p=p, cx=cx); rows.append(mt)
        prev_k_par = best_par_p
        res["qwoa_descent"][str(k)] = dict(dim=sec.dim, n_feas=int(sec.feas_sec.sum()), p_sweep=rows)
        best = max(rows, key=lambda r: r["p_feas"])
        print(f"  k={k} (dim={sec.dim}, #feas={int(sec.feas_sec.sum())}): "
              f"best P(feas)={best['p_feas']:.3f} P(opt)={best['p_opt']:.3e} @p={best['p']}")

    # ----- headline 3-way at k_opt -----------------------------------------
    def best_popt(rows): return max(r["p_opt"] for r in rows)
    qaoa_best = best_popt(res["qaoa"])
    direct_best = best_popt(res["qwoa_direct"][str(k_opt)]["p_sweep"])
    desc_best = best_popt(res["qwoa_descent"][str(k_opt)]["p_sweep"])
    print("\n" + "=" * 70 + f"\n HEADLINE: best P(opt) at k_opt={k_opt}\n" + "=" * 70)
    print(f"  QAOA          : {qaoa_best:.3e}")
    print(f"  QWOA-direct   : {direct_best:.3e}   (x{direct_best/qaoa_best:.0f} vs QAOA)")
    print(f"  QWOA-descent  : {desc_best:.3e}   (x{desc_best/qaoa_best:.0f} vs QAOA; "
          f"x{desc_best/direct_best:.2f} vs direct)")
    res["headline"] = dict(k_opt=k_opt, qaoa=qaoa_best, qwoa_direct=direct_best,
                           qwoa_descent=desc_best)

    res["meta"]["runtime_s"] = round(time.time() - t0, 1)
    json.dump(res, open(RESULTS_DIR / "compare_strong_results.json", "w"), indent=2)
    print(f"\nDone in {res['meta']['runtime_s']}s -> compare_strong_results.json")


if __name__ == "__main__":
    main()
