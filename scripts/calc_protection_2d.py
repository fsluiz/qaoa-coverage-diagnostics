#!/usr/bin/env python3
"""The protection variant: what happens when the posts must themselves be surveilled.

Pure coverage asks only that every target be reached.  Operationally a post is also an
asset, and the variant the field cares about adds, for every post b,

    x_b + sum_{b' in N(b)} x_b'  >=  1        (domination on the post graph)

or, when a post cannot vouch for itself,

    sum_{b' in N(b)} x_b'        >=  1        (total domination)

Two questions are settled here.  First, structure: penalising domination makes every closed
neighbourhood N[b] a clique, so the QUBO interaction graph becomes G_ZZ union G_B^2, which
adds edges between posts sharing no target.  Does the block decomposition survive that?
Second, cost: does protection actually bind, or is an optimal cover already a dominating set?

Feasibility differs sharply between the two constraints.  Ordinary domination is always
feasible - selecting every post satisfies it - so an isolated outpost is forced, not fatal.
Total domination is infeasible as soon as one post has no neighbour within R_p, which is the
isolated-vertex threshold of the post graph.

Writes results/protection_2d.json.
"""
from __future__ import annotations

import json
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from _paths import RESULTS_DIR
from calc_block_decomposition import interaction_graph
from calc_dimension_sweep import dmat, frontier_polyline, lattice, walk
from calc_ilp_geographic import load_municipalities

SEED = 3
TIME_LIMIT = 120     # a row that will not close in two minutes is itself the measurement


def post_graph(bla, blo, R_p):
    G = (dmat(bla, blo, bla, blo) <= R_p).astype(np.int64)
    np.fill_diagonal(G, 0)
    return G


def qubo_graph(A, G):
    """G_ZZ union G_B^2: domination penalties clique every closed neighbourhood N[b]."""
    Gc = G.copy()
    np.fill_diagonal(Gc, 1)                     # closed neighbourhood
    G2 = ((Gc @ Gc) > 0).astype(np.int64)       # b, b' both in some N[c]  =>  coupled
    np.fill_diagonal(G2, 0)
    U = ((interaction_graph(A) + G2) > 0).astype(np.int64)
    np.fill_diagonal(U, 0)
    return U


def components(C):
    n, lab = connected_components(csr_matrix(C), directed=False)
    sizes = np.bincount(lab)
    return int(n), int(sizes.max())


def cover_ilp(A, extra=None):
    """Minimum cover, optionally with extra >= 1 rows appended (the protection layer)."""
    M = A.astype(float) if extra is None else np.vstack([A.astype(float), extra.astype(float)])
    n = M.shape[1]
    lp = linprog(np.ones(n), A_ub=-M, b_ub=-np.ones(M.shape[0]), bounds=(0, 1), method="highs")
    t0 = time.perf_counter()
    r = milp(c=np.ones(n), constraints=LinearConstraint(M, lb=np.ones(M.shape[0]), ub=np.inf),
             integrality=np.ones(n), bounds=Bounds(0, 1),
             options={"time_limit": TIME_LIMIT, "mip_rel_gap": 0.0})
    el = time.perf_counter() - t0
    if r.status != 0 or r.x is None:
        # status 1 = the time limit was reached without a proof of optimality
        inc = None if r.x is None else int(round(r.x.sum()))
        return dict(status=int(r.status), k_opt=None, incumbent=inc, timed_out=True,
                    lp=None if lp.status else round(float(lp.fun), 4), seconds=round(el, 4))
    sel = np.where(r.x > 0.5)[0]
    assert (M[:, sel].sum(1) >= 1).all(), "returned solution violates its own constraints"
    return dict(status=0, k_opt=int(len(sel)), lp=round(float(lp.fun), 4),
                seconds=round(el, 4), x=sel)


def build_with_posts(kind, R, bbox, poly, rng):
    """Mirrors calc_dimension_sweep.build(), but also returns the surviving post coordinates,
    which the protection layer needs and which build() discards."""
    if kind == "frontier":
        b = walk(*poly, R); t = walk(*poly, 25.0)
    elif kind == "grid2d":
        b = lattice(*bbox, R); t = lattice(*bbox, R, offset=R / 222.0)
    else:
        la0, la1, lo0, lo1 = bbox
        area = (la1 - la0) * 111.0 * (lo1 - lo0) * 111.0 * np.cos(np.radians((la0 + la1) / 2))
        n = max(int(area / (R * R)), 10)
        b = (rng.uniform(la0, la1, n), rng.uniform(lo0, lo1, n))
        t = (rng.uniform(la0, la1, n), rng.uniform(lo0, lo1, n))
    A = (dmat(t[0], t[1], b[0], b[1]) <= R).astype(np.int64)
    keep = A.sum(0) > 0
    A = A[:, keep]
    cov = float((A.sum(1) >= 1).mean())
    return A[A.sum(1) >= 1], cov, b[0][keep], b[1][keep]


def spacing(bla, blo):
    """Median nearest-neighbour distance between posts: the natural unit for R_p."""
    D = dmat(bla, blo, bla, blo)
    np.fill_diagonal(D, np.inf)
    return float(np.median(D.min(1)))


def analyse(kind, R, ratio, bbox, poly, rng):
    A, cov, bla, blo = build_with_posts(kind, R, bbox, poly, rng)
    delta = spacing(bla, blo)
    R_p = ratio * delta
    G = post_graph(bla, blo, R_p)

    zz_comp, zz_big = components(interaction_graph(A))
    qb_comp, qb_big = components(qubo_graph(A, G))

    base = cover_ilp(A)
    dom = cover_ilp(A, extra=(G + np.eye(len(G), dtype=np.int64)))     # x_b + sum_N(b) >= 1
    isolated = int((G.sum(1) == 0).sum())
    tot = (dict(status=-1, k_opt=None, lp=None, seconds=0.0, infeasible_reason="isolated post")
           if isolated else cover_ilp(A, extra=G))                    # sum_N(b) >= 1

    dominates = bool((G[:, base["x"]].sum(1) + np.isin(np.arange(len(G)), base["x"])).min() >= 1)
    return dict(kind=kind, R_km=R, R_p_over_delta=ratio, delta_km=round(delta, 2),
                R_p_km=round(R_p, 2), n_b=int(A.shape[1]), n_t=int(A.shape[0]),
                covered=round(cov, 4),
                blocks_cover=zz_comp, largest_cover=zz_big,
                blocks_protected=qb_comp, largest_protected=qb_big,
                k_cover=base["k_opt"], k_dominated=dom["k_opt"],
                k_total_dominated=tot["k_opt"], total_dom_status=tot["status"],
                dom_timed_out=dom.get("timed_out", False),
                dom_incumbent=dom.get("incumbent"), dom_lp=dom.get("lp"),
                isolated_posts=isolated,
                optimal_cover_already_dominates=dominates,
                seconds_cover=base["seconds"], seconds_dominated=dom["seconds"])


def main() -> None:
    tla, tlo, _ = load_municipalities("centroid")
    bbox = (tla.min(), tla.max(), tlo.min(), tlo.max())
    poly = frontier_polyline()
    rows = []
    print("R_p is quoted as a multiple of the median nearest-neighbour post spacing: setting")
    print("* marks a domination ILP that hit the %d s limit without proving optimality." % TIME_LIMIT)
    print("R_p = R would land exactly on the lattice pitch, where the comparison is a coin toss.\n")
    print(f"{'regime':>10}{'R':>5}{'Rp/d':>6}{'n_B':>6}{'blk_cov':>8}{'blk_prot':>9}"
          f"{'k_cov':>6}{'k_dom':>8}{'k_tot':>6}{'iso':>5}{'cover dominates?':>18}")
    for kind, R in (("grid2d", 150.0), ("poisson2d", 150.0)):
        for ratio in (0.9, 1.2, 1.5, 2.0, 3.0):
            rng = np.random.default_rng(SEED)
            r = analyse(kind, R, ratio, bbox, poly, rng)
            rows.append(r)
            kd = (str(r['k_dominated']) if r['k_dominated'] is not None
                  else f"<={r['dom_incumbent']}*" if r['dom_incumbent'] is not None else "--")
            print(f"{kind:>10}{int(R):>5}{ratio:>6.1f}{r['n_b']:>6}{r['blocks_cover']:>8}"
                  f"{r['blocks_protected']:>9}{str(r['k_cover']):>6}{kd:>8}"
                  f"{str(r['k_total_dominated']):>6}{r['isolated_posts']:>5}"
                  f"{str(r['optimal_cover_already_dominates']):>18}", flush=True)
    json.dump({"seed": SEED, "rows": rows}, open(RESULTS_DIR / "protection_2d.json", "w"), indent=1)
    print("\n-> results/protection_2d.json")


if __name__ == "__main__":
    main()
