#!/usr/bin/env python3
"""Quantify how much of each coverage instance is fixed by unit propagation.

Central measurement for the executability-vs-triviality trade-off: a task covered by a
single base forces that base into every feasible cover.  Propagating those forcings (and
the tasks they cover) repeatedly leaves a residual problem; if the residual is empty the
optimum is obtained without search.  We report, per instance and along the R sweep:

  forced        number of bases fixed by propagation
  k_opt         ILP optimum
  frac_forced   forced / k_opt
  residual_t    tasks still uncovered after propagation
  residual_b    free bases that can still cover a residual task
  n_ZZ          two-qubit interaction count of the QAOA circuit

Writes results/triviality.json.
"""
from __future__ import annotations

import json
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from _paths import DATA_DIR, RESULTS_DIR
from run_m_ladder import build_instance, coverage


def propagate(A: np.ndarray, target: np.ndarray) -> dict:
    """Unit propagation for m-cover: if a task still needs r more covers and exactly r
    unforced bases can supply them, all r are forced.  Iterate to a fixed point.

    `need` is recomputed from the current forced set at the top of every pass, so a task
    already satisfied by earlier forcings never forces anything further.
    """
    n_t, n_b = A.shape
    forced = np.zeros(n_b, int)
    while True:
        need = np.maximum(0, target.astype(int) - (A @ forced))
        changed = False
        for t in range(n_t):
            if need[t] <= 0:
                continue
            cand = np.where((A[t] == 1) & (forced == 0))[0]
            if len(cand) == need[t]:
                forced[cand] = 1
                need = np.maximum(0, target.astype(int) - (A @ forced))
                changed = True
        if not changed:
            break
    need = np.maximum(0, target.astype(int) - (A @ forced))
    alive = need > 0
    res_t = int(alive.sum())
    res_b = int(((A[alive].sum(0) > 0) & (forced == 0)).sum()) if res_t else 0
    # Live invariant: every forced base is the unique supplier of some still-needed cover,
    # so dropping it must leave a target uncoverable.  This is what makes forced a subset of
    # every feasible cover, and hence n_forced <= k_opt.
    for b in np.where(forced == 1)[0]:
        keep = np.ones(n_b, int); keep[b] = 0             # every base except b
        assert ((A * keep[None, :]).sum(1) < target).any(), \
            f"base {b} was forced although the instance is coverable without it"
    return dict(forced=int(forced.sum()), residual_t=res_t, residual_b=res_b,
                forced_is_cover=bool((A @ forced >= target).all()))


def n_zz_of(A: np.ndarray) -> int:
    C = (A.T @ A > 0).astype(int); np.fill_diagonal(C, 0)
    return int(C.sum() // 2)


def ilp(A: np.ndarray, target: np.ndarray):
    """Returns (optimum, seconds, status).  The wall time matters here: the block
    enumeration cost sum_i 2^|V_i| is often mistaken for the cost of solving the instance,
    and the only way to show it is not is to report what solving actually costs."""
    t0 = time.perf_counter()
    r = milp(c=np.ones(A.shape[1]),
             constraints=LinearConstraint(A, lb=target, ub=np.inf),
             integrality=np.ones(A.shape[1]), bounds=Bounds(0, 1),
             options={"time_limit": 120})
    el = time.perf_counter() - t0
    k = None if r.x is None else int(round(r.x.sum()))
    return k, round(el, 5), int(r.status)


def row(name, A, m):
    target = np.minimum(m, A.sum(1))
    p = propagate(A, target)
    k, ilp_seconds, ilp_status = ilp(A, target)
    out = dict(name=name, n_b=int(A.shape[1]), n_t=int(A.shape[0]), m=m,
               n_ZZ=n_zz_of(A), k_opt=k, mean_deg=round(float(A.sum(1).mean()), 3),
               deg1_tasks=int((A.sum(1) == 1).sum()),
               ilp_seconds=ilp_seconds, ilp_status=ilp_status, **p)
    out["frac_forced"] = None if not k else round(p["forced"] / k, 3)
    return out


if __name__ == "__main__":
    rows = []
    for tag, f in [("Rung 1 (PEF-24, R=200)", "frozen_rung1.json"),
                   ("Rung 2 (Arco-79, R=100)", "frozen_arco.json"),
                   ("overshoot (n=14, m=3)", "frozen_overshoot.json")]:
        d = json.load(open(DATA_DIR / f))
        rows.append(row(tag, np.asarray(d["A"], int), d["m"]))

    # the R sweep on the canonical PEF instance (same generator as Table I)
    bl, bo, tl, to, seed, n_b = build_instance()
    sweep = []
    for R in [200, 250, 300, 350, 400, 450, 500, 550, 600]:
        A = coverage(bl, bo, tl, to, float(R)).astype(int)
        r = row(f"PEF-24 R={R}", A, 1); r["R_km"] = R
        sweep.append(r)

    print(f"{'instance':26} {'n_ZZ':>5} {'k_opt':>6} {'forced':>7} {'frac':>6} "
          f"{'res_t':>6} {'res_b':>6}")
    for r in rows + sweep:
        print(f"{r['name']:26} {r['n_ZZ']:>5} {str(r['k_opt']):>6} {r['forced']:>7} "
              f"{str(r['frac_forced']):>6} {r['residual_t']:>6} {r['residual_b']:>6}")
    json.dump(dict(instances=rows, pef_R_sweep=sweep, task_seed=int(seed)),
              open(RESULTS_DIR / "triviality.json", "w"), indent=1)
    print("\n-> results/triviality.json")
