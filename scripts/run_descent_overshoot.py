#!/usr/bin/env python3
"""
QWOA k-descent on the greedy-OVERSHOOT instance (overshoot_instance.json).

Demonstrates the non-trivial descent that the auto-tuned benchmark instance
could not show (there greedy == k_opt): classical greedy gives k_greedy, QWOA
descends the fixed-Hamming-weight sectors k_greedy -> k_greedy-1 -> ... until it
hits the optimum k_opt, locating a feasible (m-fold) cover at each rung.

Also reruns the QWOA-vs-QAOA head-to-head AT the optimum sector on this harder
instance.  Reuses the exact engine from benchmark_qwoa_qaoa.py.

Pure NumPy/SciPy.  Author: assistant for F. S. Luiz (IFGW/UNICAMP).
"""
from __future__ import annotations
import json
import time

import numpy as np

from benchmark_qwoa_qaoa import (
    P_LIST, precompute, classical_optimum, greedy_mcover, ilp_mcover,
    cooperation_edges, cx_counts, qaoa_run, qaoa_metrics, qwoa_metrics,
    QWOASector, optimise,
)
from _paths import DATA_DIR, RESULTS_DIR


def main():
    t0 = time.time()
    h = json.load(open(DATA_DIR / "overshoot_instance.json"))
    A_q = np.array(h["A_q"], dtype=int)
    m = int(h["m"])
    n = A_q.shape[0]
    edges = cooperation_edges(A_q)
    n_edges_xy = len(edges)
    import itertools
    n_ZZ = int(np.sum([1 for (b1, b2) in itertools.combinations(range(n), 2)
                       if int(np.sum((A_q[b1] == 1) & (A_q[b2] == 1))) > 0]))

    pc = precompute(A_q, m)
    k_opt, opt_states = classical_optimum(pc)
    k_greedy, greedy_sel = greedy_mcover(A_q, m)
    k_ilp, ilp_sel, _ = ilp_mcover(A_q, m)

    print(f"Overshoot instance: n={n} qubits, {A_q.shape[1]} tasks, m={m}, "
          f"n_ZZ={n_ZZ}, XY-edges={n_edges_xy}")
    print(f"  classical: greedy={k_greedy}, ILP={k_ilp}, brute k_opt={k_opt} "
          f"(#opt={len(opt_states)})  | OVERSHOOT = {k_greedy - k_opt}")

    rng = np.random.RandomState(0)
    sec_cache: dict[int, QWOASector] = {}
    def get_sector(k):
        if k not in sec_cache:
            sec_cache[k] = QWOASector(A_q, edges, k, pc)
        return sec_cache[k]

    results: dict = dict(
        meta=dict(source="overshoot_instance.json", n_qubits=n,
                  n_tasks=int(A_q.shape[1]), m=m, n_ZZ=n_ZZ,
                  n_edges_xy=n_edges_xy, p_list=P_LIST,
                  k_greedy=k_greedy, k_ilp=k_ilp, k_opt=k_opt,
                  overshoot=k_greedy - k_opt, n_opt=len(opt_states)),
        descent=[], qaoa_at_opt=[], qwoa_at_opt=[],
    )

    # ---- QWOA k-descent: k_greedy -> k_opt, warm-started down the ladder ----
    print(f"\n{'='*64}\n QWOA k-descent (warm-started, best over p in {P_LIST})\n{'='*64}")
    print(f"{'k':>3} {'dimC':>6} {'#feas':>6} {'feas/dim':>8} "
          f"{'P(feas)':>9} {'P(opt)':>9} {'CX':>5}")
    warm = None
    for k in range(k_greedy, k_opt - 1, -1):
        sec = get_sector(k)
        dim, nf = sec.dim, int(sec.feas_sec.sum())
        best_pf, best_probs, best_par, best_p = -1.0, None, None, None
        wp = warm
        for p in P_LIST:
            fn = lambda par: float((sec.evolve(p, par) * sec.Esec).sum())
            par, _ = optimise(fn, p, rng, warm=wp)
            wp = par                                   # INTERP warm-start within sector
            probs = sec.evolve(p, par)
            pf = float(probs[sec.feas_sec].sum())
            if pf > best_pf:
                best_pf, best_probs, best_par, best_p = pf, probs, par, p
        warm = best_par[:2 * 1] if best_par is not None else None  # seed next k from p=1 slice
        mt = qwoa_metrics(best_probs, sec, k_opt)
        cx = cx_counts(n_ZZ, n_edges_xy, k, n, best_p, "QWOA")
        results["descent"].append(dict(
            k=k, dim=dim, n_feasible=nf, p_best=best_p,
            p_feas=mt["p_feas"], p_opt=mt["p_opt"], cx=cx))
        print(f"{k:>3} {dim:>6} {nf:>6} {nf/dim:>8.3f} "
              f"{mt['p_feas']:>9.3f} {mt['p_opt']:>9.3e} {cx:>5}")

    # ---- QWOA vs QAOA head-to-head at the optimum sector --------------------
    print(f"\n{'='*64}\n QWOA vs QAOA at k_opt={k_opt}\n{'='*64}")
    sec = get_sector(k_opt)
    print(f"  P_random(opt) in 2^{n} = {len(opt_states)/(1<<n):.2e}; "
          f"in C({n},{k_opt})={sec.dim} = {len(opt_states)/sec.dim:.2e}")
    prev = None
    for p in P_LIST:
        fn = lambda par: float((qaoa_run(pc, n, p, par) * pc["energy"]).sum())
        par, _ = optimise(fn, p, rng, warm=prev); prev = par
        probs = qaoa_run(pc, n, p, par)
        mt = qaoa_metrics(probs, pc, opt_states)
        cx = cx_counts(n_ZZ, n_edges_xy, k_opt, n, p, "QAOA")
        mt.update(p=p, cx=cx); results["qaoa_at_opt"].append(mt)
        print(f"  QAOA p={p}: P(opt)={mt['p_opt']:.3e}  P(feas)={mt['p_feas']:.3f}  "
              f"<k|feas>={mt['exp_k_feas']:.2f}  CX={cx}")
    prev = None
    for p in P_LIST:
        fn = lambda par: float((sec.evolve(p, par) * sec.Esec).sum())
        par, _ = optimise(fn, p, rng, warm=prev); prev = par
        probs = sec.evolve(p, par)
        mt = qwoa_metrics(probs, sec, k_opt)
        cx = cx_counts(n_ZZ, n_edges_xy, k_opt, n, p, "QWOA")
        mt.update(p=p, cx=cx); results["qwoa_at_opt"].append(mt)
        print(f"  QWOA p={p}: P(opt)={mt['p_opt']:.3e}  P(feas)={mt['p_feas']:.3f}  "
              f"<k|feas>={mt['exp_k_feas']:.2f}  CX={cx}")

    results["meta"]["runtime_s"] = round(time.time() - t0, 1)
    json.dump(results, open(RESULTS_DIR / "overshoot_descent_results.json", "w"), indent=2)
    print(f"\nDone in {results['meta']['runtime_s']}s -> overshoot_descent_results.json")


if __name__ == "__main__":
    main()
