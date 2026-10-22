#!/usr/bin/env python3
"""Irregular two-dimensional geometry at genuinely complete coverage.

`calc_dimension_sweep.py` answers the refinement question in three geometries, but its
poisson2d arm discards 5-8% of the targets: `build()` returns `A[A.sum(1) >= 1]`, so a
target no post can reach is dropped rather than covered.  Operationally that is the very
thing the refinement question forbids - an unreachable waypoint is an unwatched stretch -
so the blocks and treewidths reported there are measured on a mutilated instance.

Here the post intensity is instead grown until every target is covered, and only the
resulting full-coverage instance is analysed.  Three experiments:

  full      one seed per radius, the honest counterpart of the sweep's poisson2d rows
  seeds     three seeds per radius, to see whether a loose LP bound is reproducible
  scale     one seed, radius pushed down until the solver starts to feel it

Writes results/coverage_preserving_2d.json.
"""
from __future__ import annotations

import json
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from _paths import RESULTS_DIR
from calc_block_decomposition import decompose
from calc_dimension_sweep import dmat
from calc_ilp_geographic import load_municipalities

SEED = 3
TIME_LIMIT = 1200
MAX_POSTS = 40000


def region():
    la, lo, _ = load_municipalities("centroid")
    box = (la.min(), la.max(), lo.min(), lo.max())
    area = (box[1] - box[0]) * 111.0 * (box[3] - box[2]) * 111.0 * np.cos(
        np.radians((box[0] + box[1]) / 2))
    return box, float(area)


def full_cover_instance(R, seed, box, area):
    """Grow the post count until no target is left uncovered, then drop idle posts."""
    la0, la1, lo0, lo1 = box
    n_t = max(int(area / (R * R)), 10)
    n_b = n_t
    while True:
        rng = np.random.default_rng(seed)
        b = (rng.uniform(la0, la1, n_b), rng.uniform(lo0, lo1, n_b))
        t = (rng.uniform(la0, la1, n_t), rng.uniform(lo0, lo1, n_t))
        A = (dmat(t[0], t[1], b[0], b[1]) <= R).astype(np.int64)
        if (A.sum(1) >= 1).all():
            break
        n_b = int(n_b * 1.25) + 1
        if n_b > MAX_POSTS:
            raise SystemExit(f"R={R}: no full coverage below {MAX_POSTS} posts")
    return A[:, A.sum(0) > 0], n_b


def solve(A):
    Af = A.astype(float)
    lp = linprog(np.ones(Af.shape[1]), A_ub=-Af, b_ub=-np.ones(Af.shape[0]),
                 bounds=(0, 1), method="highs")
    assert lp.status == 0, f"LP relaxation failed: {lp.message}"
    t0 = time.perf_counter()
    r = milp(c=np.ones(Af.shape[1]),
             constraints=LinearConstraint(Af, lb=np.ones(Af.shape[0]), ub=np.inf),
             integrality=np.ones(Af.shape[1]), bounds=Bounds(0, 1),
             options={"time_limit": TIME_LIMIT, "mip_rel_gap": 0.0})
    seconds = time.perf_counter() - t0
    assert r.status == 0 and r.x is not None, f"ILP not proven optimal: status {r.status}"
    sel = np.where(r.x > 0.5)[0]
    assert (A[:, sel].sum(1) >= 1).all(), "the returned ILP solution is not a cover"
    assert len(sel) == int(round(r.x.sum())), "fractional entries in an integral solution"
    return float(lp.fun), len(sel), seconds


def row(R, seed, box, area):
    A, n_posted = full_cover_instance(R, seed, box, area)
    d = decompose(A)
    lp, k, seconds = solve(A)
    ceil_lp = int(np.ceil(lp - 1e-9))
    return dict(R_km=R, seed=seed, n_b=int(A.shape[1]), n_t=int(A.shape[0]),
                n_posted=n_posted, n_ZZ=d["n_ZZ"], n_components=d["n_components"],
                largest_block=d["largest_block"], direct_sum=d["direct_sum"],
                lp=round(lp, 4), ceil_lp=ceil_lp, k_opt=k,
                bound_slack=k - ceil_lp, covered=1.0, seconds=round(seconds, 4))


def show(tag, rows):
    print(f"\n=== {tag} ===")
    print(f"{'R':>5}{'seed':>5}{'n_B':>7}{'n_T':>6}{'n_ZZ':>9}{'blk':>5}{'big':>7}"
          f"{'LP':>10}{'ceil':>6}{'k_opt':>6}{'slack':>6}{'t (s)':>9}")
    for r in rows:
        print(f"{r['R_km']:>5}{r['seed']:>5}{r['n_b']:>7}{r['n_t']:>6}{r['n_ZZ']:>9}"
              f"{r['n_components']:>5}{r['largest_block']:>7}{r['lp']:>10.2f}"
              f"{r['ceil_lp']:>6}{r['k_opt']:>6}{r['bound_slack']:>6}{r['seconds']:>9.3f}",
              flush=True)


def main() -> None:
    box, area = region()
    print(f"bounding box {area / 1e6:.2f} million km^2, seed {SEED}")
    out = {"seed": SEED, "bbox_area_km2": round(area, 1), "experiments": {}}

    full = [row(R, SEED, box, area) for R in (400, 300, 200, 150, 120)]
    show("full", full)

    seeds = [row(R, s, box, area) for R in (200, 150, 120) for s in (3, 4, 5)]
    show("seeds", seeds)

    # Stops at 60 km: below that the search for a full-coverage post set rebuilds a
    # distance matrix of order 10^8 entries per attempt, and the trend over the 22x
    # span in n_B already reported here is unambiguous.
    scale = [row(R, SEED, box, area) for R in (200, 150, 120, 100, 80, 60)]
    show("scale", scale)

    out["experiments"] = {"full": full, "seeds": seeds, "scale": scale}
    loose = [r for r in seeds if r["bound_slack"] > 0]
    out["seeds_with_loose_lp_bound"] = len(loose)
    out["seeds_tested"] = len(seeds)
    json.dump(out, open(RESULTS_DIR / "coverage_preserving_2d.json", "w"), indent=1)
    print(f"\nLP bound strictly below the optimum in {len(loose)}/{len(seeds)} seeded instances")
    print("-> results/coverage_preserving_2d.json")


if __name__ == "__main__":
    main()
