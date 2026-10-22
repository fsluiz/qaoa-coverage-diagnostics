#!/usr/bin/env python3
"""When full coverage is no longer possible, how much can re-planning still save?

Two gaps left by `calc_tactical_replanning.py`.

First, its only failure mode removes whole posts.  The v3 notebooks also model `area_jam`,
which degrades individual links inside a rectangle while every post stays alive - the one
mode that can fragment coverage without costing a single asset.

Second, and more important, it only ever asks for a *complete* cover, so once a target
becomes unreachable it reports infeasible and stops.  Operationally the mission does not
stop: the question becomes how much of the frontier can still be watched on the budget
already in the field.  That is maximum coverage, not minimum cover - a different problem
with a different relaxation - and the quantity that matters is the gain from re-optimising
over doing nothing:

    passive    keep the surviving posts of the old plan, accept whatever they still cover
    active     re-solve for maximum coverage on the same budget, with the casualty barred

Writes results/graceful_degradation.json.
"""
from __future__ import annotations

import json
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from _paths import DATA_DIR, RESULTS_DIR

SEED = 3


def max_coverage(A, budget, forbid=None, relax=False):
    """Maximise the number of covered targets on a fixed budget of posts.

    Variables are [x_b (posts) | z_t (targets covered)].  z_t <= sum_b A[t,b] x_b, and
    sum_b x_b <= budget.
    """
    n_t, n_b = A.shape
    n = n_b + n_t
    c = np.concatenate([np.zeros(n_b), -np.ones(n_t)])       # maximise sum z
    rows = [np.concatenate([np.ones(n_b), np.zeros(n_t)])]
    lb, ub_c = [0.0], [float(budget)]
    for t in range(n_t):
        r = np.zeros(n)
        r[:n_b] = -A[t].astype(float)
        r[n_b + t] = 1.0
        rows.append(r); lb.append(-np.inf); ub_c.append(0.0)   # z_t - sum A x <= 0
    M = np.vstack(rows)
    ub = np.ones(n)
    if forbid is not None and len(forbid):
        ub[list(forbid)] = 0.0
    t0 = time.perf_counter()
    if relax:
        r = linprog(c, A_ub=M, b_ub=np.array(ub_c), bounds=list(zip(np.zeros(n), ub)),
                    method="highs")
        el = time.perf_counter() - t0
        return dict(status=int(r.status), covered=None if r.status else -float(r.fun),
                    seconds=round(el, 5), integral=None)
    r = milp(c=c, constraints=LinearConstraint(M, lb=np.array(lb), ub=np.array(ub_c)),
             integrality=np.concatenate([np.ones(n_b), np.zeros(n_t)]),
             bounds=Bounds(np.zeros(n), ub), options={"time_limit": 600, "mip_rel_gap": 0.0})
    el = time.perf_counter() - t0
    if r.status != 0 or r.x is None:
        return dict(status=int(r.status), covered=None, seconds=round(el, 5))
    x = (r.x[:n_b] > 0.5).astype(int)
    cov = int((A @ x >= 1).sum())
    assert x.sum() <= budget + 1e-6, "budget violated"
    return dict(status=0, covered=cov, x=x, seconds=round(el, 5))


def greedy_coverage(A, budget, forbid=None):
    """The classical (1 - 1/e) baseline, for the price comparison."""
    t0 = time.perf_counter()
    avail = np.ones(A.shape[1], bool)
    if forbid is not None and len(forbid):
        avail[list(forbid)] = False
    done = np.zeros(A.shape[0], bool)
    picked = []
    for _ in range(budget):
        gain = ((A == 1) & ~done[:, None]).sum(0) * avail
        b = int(gain.argmax())
        if gain[b] == 0:
            break
        picked.append(b); avail[b] = False; done |= A[:, b] == 1
    return dict(covered=int(done.sum()), k=len(picked),
                seconds=round(time.perf_counter() - t0, 5))


def jam(A, tla, tlo, bla, blo, box, p, rng):
    """Degrade links with an endpoint inside the jamming rectangle, each with probability p."""
    la0, la1, lo0, lo1 = box
    inside_t = (tla >= la0) & (tla <= la1) & (tlo >= lo0) & (tlo <= lo1)
    inside_b = (bla >= la0) & (bla <= la1) & (blo >= lo0) & (blo <= lo1)
    hit = inside_t[:, None] | inside_b[None, :]
    drop = hit & (rng.random(A.shape) < p)
    return np.where(drop, 0, A), int((drop & (A == 1)).sum())


def main() -> None:
    out = {"seed": SEED, "instances": {}}
    for tag, fn in (("rung1", "frozen_rung1.json"), ("arco", "frozen_arco.json")):
        d = json.load(open(DATA_DIR / fn))
        A = np.asarray(d["A"], int)
        tla = np.asarray(d["task_lats"], float); tlo = np.asarray(d["task_lons"], float)
        bla = np.asarray(d["base_lats"], float); blo = np.asarray(d["base_lons"], float)
        k = int(d["k_opt"]); n_t = A.shape[0]
        rec = {"n_b": int(A.shape[1]), "n_t": n_t, "k_opt": k}
        print(f"\n===== {tag}: {d['name']} =====")
        print(f"{A.shape[1]} posts, {n_t} targets, full cover uses k = {k}")

        full = max_coverage(A, k)
        assert full["covered"] == n_t, "the budget k should still cover everything intact"

        # Warm NumPy/Python dispatch before recording a sub-millisecond heuristic.
        # A cold first call can otherwise dominate the quantity being measured and
        # even reverse the descriptive ILP/greedy timing order on a busy machine.
        greedy_coverage(A, k)

        # ---- does re-optimising beat doing nothing, once a post is lost? -------------
        print("\n-- lose one post: passive (keep the survivors) vs active (re-optimise) --")
        print(f"{'lost':>5}{'passive':>9}{'active':>8}{'gain':>6}{'greedy':>8}"
              f"{'ILP ms':>8}{'greedy ms':>10}{'LP relax':>10}")
        rows = []
        for b in np.where(full["x"] > 0)[0]:
            surv = full["x"].copy(); surv[b] = 0
            passive = int((A @ surv >= 1).sum())
            act = max_coverage(A, k, forbid=[b])
            greedy_runs = [greedy_coverage(A, k, forbid=[b]) for _ in range(5)]
            if len({run["covered"] for run in greedy_runs}) != 1:
                raise AssertionError("deterministic greedy coverage changed across timing repeats")
            gre = dict(greedy_runs[0])
            gre["seconds"] = round(
                float(np.median([run["seconds"] for run in greedy_runs])), 5
            )
            rel = max_coverage(A, k, forbid=[b], relax=True)
            rows.append(dict(lost=int(b), passive=passive, active=act["covered"],
                             gain=act["covered"] - passive, greedy=gre["covered"],
                             ilp_seconds=act["seconds"], greedy_seconds=gre["seconds"],
                             lp_relax=None if rel["covered"] is None else round(rel["covered"], 3)))
            r = rows[-1]
            print(f"{b:>5}{r['passive']:>9}{r['active']:>8}{r['gain']:>+6}{r['greedy']:>8}"
                  f"{r['ilp_seconds'] * 1e3:>8.1f}{r['greedy_seconds'] * 1e3:>10.2f}"
                  f"{str(r['lp_relax']):>10}")
        rec["single_loss"] = rows
        g = [r["gain"] for r in rows]
        print(f"  re-optimising recovers {min(g)} to {max(g)} targets "
              f"(mean {np.mean(g):.1f}) that passive loss would have forfeited")
        print(f"  greedy matches the ILP on {sum(r['greedy'] == r['active'] for r in rows)}"
              f"/{len(rows)} losses; ILP is {np.mean([r['ilp_seconds'] for r in rows]) / np.mean([r['greedy_seconds'] for r in rows]):.1f}x "
              f"the cost of greedy")

        # ---- how much budget buys back full coverage? --------------------------------
        print("\n-- coverage against budget, after the worst single loss --")
        worst = min(rows, key=lambda r: r["active"])["lost"]
        curve = []
        for extra in range(0, 6):
            r = max_coverage(A, k + extra, forbid=[worst])
            curve.append(dict(budget=k + extra, covered=r["covered"], seconds=r["seconds"]))
            print(f"   budget {k + extra:>3}: {r['covered']:>3}/{n_t} targets "
                  f"({100 * r['covered'] / n_t:>5.1f}%)  {r['seconds'] * 1e3:.1f} ms")
        rec["budget_curve"] = {"lost": int(worst), "rows": curve}

        # ---- area jamming: links die, every post survives ---------------------------
        print("\n-- area jamming: links degraded inside a rectangle, no post lost --")
        print(f"{'box%':>6}{'p':>5}{'links cut':>11}{'full cover?':>13}{'max cov':>9}{'ms':>7}")
        cla, clo = float(np.median(tla)), float(np.median(tlo))
        rec["jam"] = []
        for frac in (0.25, 0.5):
            h = frac * (tla.max() - tla.min()) / 2, frac * (tlo.max() - tlo.min()) / 2
            box = (cla - h[0], cla + h[0], clo - h[1], clo + h[1])
            for p in (0.5, 1.0):
                rng = np.random.default_rng(SEED)
                Aj, cut = jam(A, tla, tlo, bla, blo, box, p, rng)
                reach = (Aj.sum(1) >= 1).all()
                r = max_coverage(Aj, k)
                rec["jam"].append(dict(box_frac=frac, p=p, links_cut=cut,
                                       full_cover_possible=bool(reach),
                                       covered=r["covered"], seconds=r["seconds"]))
                print(f"{frac:>6.2f}{p:>5.1f}{cut:>11}{str(bool(reach)):>13}"
                      f"{str(r['covered']):>9}{r['seconds'] * 1e3:>7.1f}")
        out["instances"][tag] = rec

    json.dump(out, open(RESULTS_DIR / "graceful_degradation.json", "w"), indent=1, default=str)
    print("\n-> results/graceful_degradation.json")


if __name__ == "__main__":
    main()
