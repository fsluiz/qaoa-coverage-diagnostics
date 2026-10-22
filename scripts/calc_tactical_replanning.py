#!/usr/bin/env python3
"""When a post is lost and the cover must be reorganised, is the classical solver still ahead?

"Reorganising after a failure" hides three different problems, and they do not have the
same difficulty.  This script separates them and times each one.

  reactive     a post is destroyed; re-solve the cover on what is left, either from
               scratch or with a hysteresis penalty that discourages churning posts that
               are still working.  This is what the tactical loop actually runs.
  preventive   buy redundancy up front: every target covered m times, so any m-1 losses
               are absorbed with no re-planning at all.
  robust       two-stage with recourse: choose a deployment now, let an adversary destroy
               up to Gamma of the posts actually emplaced, then repair at a premium.
               min-max-min, and the only formulation here that is not a plain ILP.

Writes results/tactical_replanning.json.
"""
from __future__ import annotations

import itertools
import json
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from _paths import DATA_DIR, RESULTS_DIR
from calc_dimension_sweep import dmat
from calc_ilp_geographic import load_municipalities

SEED = 3
BIG = 10**6


def cover_ilp(A, cost=None, forbid=None, m=1, warm=None):
    """Minimum-cost set cover with m-fold coverage; posts in `forbid` are unavailable."""
    n = A.shape[1]
    c = np.ones(n) if cost is None else np.asarray(cost, float)
    ub = np.ones(n)
    if forbid is not None and len(forbid):
        ub[list(forbid)] = 0.0
    # No clamping: a target reachable by fewer than m available posts makes the m-fold
    # requirement genuinely infeasible, and saying so is the point.
    need = np.full(A.shape[0], float(m))
    feasible = bool((A[:, ub > 0].sum(1) >= need).all())
    if not feasible:
        return dict(status=-1, k=None, x=None, seconds=0.0)
    t0 = time.perf_counter()
    r = milp(c=c, constraints=LinearConstraint(A.astype(float), lb=need, ub=np.inf),
             integrality=np.ones(n), bounds=Bounds(np.zeros(n), ub),
             options={"time_limit": 300, "mip_rel_gap": 0.0})
    el = time.perf_counter() - t0
    if r.status != 0 or r.x is None:
        return dict(status=int(r.status), k=None, x=None, seconds=round(el, 5))
    x = (r.x > 0.5).astype(int)
    assert (A @ x >= need).all(), "returned solution does not meet the coverage requirement"
    return dict(status=0, k=int(x.sum()), x=x, seconds=round(el, 5), obj=float(r.fun))


def reactive(A, rng, mode, n_fail, hysteresis):
    """Destroy posts one at a time, re-solving after each loss."""
    base = cover_ilp(A)
    cur, lost, rows = base["x"], [], []
    for step in range(1, n_fail + 1):
        live = [b for b in range(A.shape[1]) if b not in lost]
        if mode == "random":
            victim = int(rng.choice([b for b in live if cur is not None and cur[b]] or live))
        else:                                        # targeted: take out the busiest post
            sel = [b for b in live if cur is not None and cur[b]]
            victim = int(max(sel or live, key=lambda b: A[:, b].sum()))
        lost.append(victim)
        # hysteresis: re-using a post already emplaced is cheaper than moving to a new one
        cost = np.ones(A.shape[1])
        if hysteresis and cur is not None:
            cost = np.where(cur > 0, 1.0, 1.0 + hysteresis)
        r = cover_ilp(A, cost=cost, forbid=lost)
        churn = None if r["x"] is None or cur is None else int((r["x"] != cur).sum())
        rows.append(dict(step=step, lost=len(lost), victim=victim, k=r["k"],
                         churn=churn, seconds=r["seconds"], feasible=r["status"] == 0))
        if r["status"] != 0:
            break
        cur = r["x"]
    return base, rows


def robust_ccg(A, gamma, lam, max_iter=30):
    """Resilience of a static minimum cover to an adversary of budget `gamma`.

    NOT a two-stage robust optimisation: the master step re-solves the plain minimum cover
    and never incorporates the scenarios already discovered, so the first-stage plan is the
    unprotected optimum rather than a hedged one.  What the separation step measures is
    therefore how dearly -- or whether at all -- that fixed plan can be repaired after the
    adversary removes up to `gamma` of the posts it emplaced.  That is the quantity the
    manuscript uses; a genuine min-max-min solve would need the master to carry a repair-cost
    constraint per scenario.
    """
    n = A.shape[1]
    scenarios: list[tuple[int, ...]] = [()]
    sub_ilps = 0
    t0 = time.perf_counter()
    for it in range(1, max_iter + 1):
        # master: cheapest x that survives every scenario found so far, repairs included
        best = None
        cand = cover_ilp(A)
        if cand["status"] != 0:
            return dict(gamma=gamma, status=-1, iterations=it, sub_ilps=sub_ilps,
                        seconds=round(time.perf_counter() - t0, 4))
        # enumerate repairs for the current x over all scenarios (the separation step)
        x = cand["x"]
        sel = np.where(x > 0)[0]
        worst, worst_S = -1, None
        for S in itertools.combinations(sel.tolist(), min(gamma, len(sel))):
            sub_ilps += 1
            surviving = x.copy()
            surviving[list(S)] = 0
            residual = A[A @ surviving < 1]          # targets left uncovered by the survivors
            if residual.shape[0] == 0:
                rep = 0
            else:
                rr = cover_ilp(residual, forbid=set(S) | set(np.where(x > 0)[0].tolist()))
                if rr["status"] != 0:
                    # no surviving or unemplaced post reaches some target: not a dear
                    # repair but an impossible one
                    return dict(gamma=gamma, status=1, unrepairable=True,
                                k_first=int(x.sum()), destroyed=list(S),
                                iterations=it, sub_ilps=sub_ilps,
                                seconds=round(time.perf_counter() - t0, 4))
                rep = rr["k"]
            if rep > worst:
                worst, worst_S = rep, S
        best = dict(k_first=int(x.sum()), worst_repair=int(worst),
                    total=float(x.sum() + lam * worst))
        if worst_S in scenarios or gamma == 0 or worst == 0:
            break
        scenarios.append(worst_S)
    return dict(gamma=gamma, status=0, iterations=it, sub_ilps=sub_ilps,
                seconds=round(time.perf_counter() - t0, 4), **best)


def fragility(A):
    """Targets with a single reachable post admit neither redundancy nor repair."""
    deg = A.sum(1)
    sole = {int(np.where(A[t] == 1)[0][0]) for t in np.where(deg == 1)[0]}
    return dict(min_reachable=int(deg.min()), max_m_possible=int(deg.min()),
                singly_reachable=int((deg == 1).sum()),
                singly_reachable_frac=round(float((deg == 1).mean()), 4),
                irreplaceable_posts=len(sole))


def retrofit(d, R_km, m=2):
    """How many extra posts, sited at real municipal centroids, buy m-fold coverage?

    The existing posts are free (already emplaced); every candidate site costs one.  The
    answer is the price of being able to reorganise at all after a loss.
    """
    mla, mlo, _ = load_municipalities("centroid")
    bla = np.asarray(d["base_lats"], float); blo = np.asarray(d["base_lons"], float)
    tla = np.asarray(d["task_lats"], float); tlo = np.asarray(d["task_lons"], float)
    A_new = (dmat(tla, tlo, mla, mlo) <= R_km).astype(int)
    A_all = np.hstack([np.asarray(d["A"], int), A_new])
    n_old = len(bla)
    reach = A_all.sum(1)
    if (reach < m).any():
        return dict(m=m, feasible=False, unreachable=int((reach < m).sum()),
                    n_candidates=int(A_new.shape[1]))
    cost = np.concatenate([np.zeros(n_old), np.ones(A_new.shape[1])])
    r = cover_ilp(A_all, cost=cost, m=m)
    if r["status"] != 0:
        return dict(m=m, feasible=False, n_candidates=int(A_new.shape[1]))
    new_posts = int(r["x"][n_old:].sum())
    return dict(m=m, feasible=True, n_candidates=int(A_new.shape[1]),
                new_posts=new_posts, posts_used_total=int(r["x"].sum()),
                seconds=r["seconds"])


def main() -> None:
    out = {"seed": SEED, "instances": {}}
    for tag, fn in (("rung1", "frozen_rung1.json"), ("arco", "frozen_arco.json")):
        d = json.load(open(DATA_DIR / fn))
        A = np.asarray(d["A"], int)
        rec = {"n_b": int(A.shape[1]), "n_t": int(A.shape[0])}
        print(f"\n===== {tag}: {d['name']}  ({A.shape[1]} posts, {A.shape[0]} targets) =====")

        base = cover_ilp(A)
        print(f"initial cover: k = {base['k']} in {base['seconds'] * 1e3:.1f} ms")
        rec["k_opt"] = base["k"]
        rec["seconds_initial"] = base["seconds"]

        print("\n-- reactive: destroy posts one at a time, re-solve after each loss --")
        print(f"{'mode':>10}{'hyst':>6}{'step':>5}{'k':>5}{'churn':>7}{'ms':>8}{'feasible':>10}")
        rec["reactive"] = {}
        for mode in ("random", "targeted"):
            for hyst in (0.0, 0.5):
                rng = np.random.default_rng(SEED)
                _, rows = reactive(A, rng, mode, n_fail=min(8, A.shape[1] - 1), hysteresis=hyst)
                rec["reactive"][f"{mode}_h{hyst}"] = rows
                for r in rows:
                    print(f"{mode:>10}{hyst:>6.1f}{r['step']:>5}{str(r['k']):>5}"
                          f"{str(r['churn']):>7}{r['seconds'] * 1e3:>8.1f}"
                          f"{str(r['feasible']):>10}")

        print("\n-- preventive: m-fold coverage, redundancy bought up front --")
        print(f"{'m':>3}{'k':>6}{'premium':>9}{'ms':>8}{'feasible':>10}")
        rec["preventive"] = []
        for m in (1, 2, 3):
            r = cover_ilp(A, m=m)
            prem = None if r["k"] is None else r["k"] / base["k"]
            rec["preventive"].append(dict(m=m, k=r["k"], premium=prem, seconds=r["seconds"],
                                          feasible=r["status"] == 0))
            print(f"{m:>3}{str(r['k']):>6}{('%.2fx' % prem) if prem else '--':>9}"
                  f"{r['seconds'] * 1e3:>8.1f}{str(r['status'] == 0):>10}")

        rec["fragility"] = fragility(A)
        f = rec["fragility"]
        print(f"\n-- fragility: {f['singly_reachable']}/{A.shape[0]} targets reachable by a "
              f"single post; {f['irreplaceable_posts']} posts are irreplaceable --")
        print(f"   the largest m with genuine m-fold coverage is m = {f['max_m_possible']}")

        print("\n-- retrofit: extra posts at real municipal centroids to buy m-fold cover --")
        rec["retrofit"] = []
        for m in (2, 3):
            rr = retrofit(d, float(d["R_km"]), m=m)
            rec["retrofit"].append(rr)
            if rr["feasible"]:
                print(f"   m={m}: {rr['new_posts']} new posts needed "
                      f"(from {rr['n_candidates']} candidate sites), {rr['seconds'] * 1e3:.0f} ms")
            else:
                print(f"   m={m}: impossible even with all {rr['n_candidates']} candidate sites"
                      + (f"; {rr['unreachable']} targets still short" if 'unreachable' in rr else ""))

        print("\n-- robust: two-stage with recourse, adversary destroys up to Gamma posts --")
        print(f"{'Gamma':>6}{'k_1st':>7}{'worst repair':>14}{'sub-ILPs':>10}{'seconds':>9}")
        rec["robust"] = []
        for g in (0, 1, 2, 3):
            r = robust_ccg(A, g, lam=2.0)
            rec["robust"].append(r)
            print(f"{g:>6}{str(r.get('k_first')):>7}{str(r.get('worst_repair')):>14}"
                  f"{r['sub_ilps']:>10}{r['seconds']:>9.2f}", flush=True)
        out["instances"][tag] = rec

    json.dump(out, open(RESULTS_DIR / "tactical_replanning.json", "w"), indent=1, default=str)
    print("\n-> results/tactical_replanning.json")


if __name__ == "__main__":
    main()
