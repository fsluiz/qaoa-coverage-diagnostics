#!/usr/bin/env python3
"""Does the block decomposition survive refinement?  Three geometries, one protocol.

Refining surveillance is not free: an unreachable target is an unwatched stretch, so shrinking
R must be compensated by emplacing posts.  This script holds coverage at 100% while R shrinks,
in three geometries, and asks in each whether the instance stays classically easy and by what
mechanism:

  frontier  - posts and targets along the PEF border polyline (quasi-one-dimensional)
  grid2d    - posts and targets on a lattice over the Legal Amazon (two-dimensional, regular)
  poisson2d - posts and targets drawn uniformly over the same region (two-dimensional, irregular)

Writes results/dimension_sweep.json.
"""
from __future__ import annotations

import json
import time

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.approximation import treewidth_min_fill_in
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from _paths import DATA_DIR, RESULTS_DIR
from calc_block_decomposition import decompose
from calc_ilp_geographic import load_municipalities

EARTH_R_KM = 6371.0088
TW_LIMIT = 900
SEED = 3


def dmat(la1, lo1, la2, lo2):
    p1, p2 = np.radians(la1)[:, None], np.radians(la2)[None, :]
    dl = np.radians(lo2)[None, :] - np.radians(lo1)[:, None]
    h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def hav(a1, o1, a2, o2):
    p1, p2 = np.radians(a1), np.radians(a2)
    dl = np.radians(o2 - o1)
    h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def frontier_polyline():
    df = pd.read_excel(DATA_DIR / "pef_planilha.xlsx")
    lat = df["latitude"].to_numpy(float); lon = df["longitude"].to_numpy(float)
    o = np.argsort(-lat)
    return lat[o], lon[o]


def walk(lat, lon, spacing):
    la, lo = [lat[0]], [lon[0]]
    for i in range(1, len(lat)):
        n = max(int(hav(lat[i - 1], lon[i - 1], lat[i], lon[i]) / spacing), 1)
        for j in range(1, n + 1):
            f = j / n
            la.append(lat[i - 1] + f * (lat[i] - lat[i - 1]))
            lo.append(lon[i - 1] + f * (lon[i] - lon[i - 1]))
    return np.array(la), np.array(lo)


def lattice(la0, la1, lo0, lo1, spacing, offset=0.0):
    dla = spacing / 111.0
    out_la, out_lo = [], []
    for a in np.arange(la0, la1 + dla, dla):
        dlo = spacing / (111.0 * max(np.cos(np.radians(a)), 0.2))
        for o in np.arange(lo0, lo1 + dlo, dlo):
            out_la.append(a + offset); out_lo.append(o + offset)
    return np.array(out_la), np.array(out_lo)


def build(kind, R, bbox, poly, rng):
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
    A = (dmat(t[0], t[1], b[0], b[1]) <= R).astype(np.int8)
    A = A[:, A.sum(0) > 0]
    cov = float((A.sum(1) >= 1).mean())
    return A[A.sum(1) >= 1], cov


def analyse(A):
    d = decompose(A)
    C = (A.T.astype(np.int32) @ A.astype(np.int32) > 0).astype(np.int8)
    np.fill_diagonal(C, 0)
    tw = int(treewidth_min_fill_in(nx.from_numpy_array(C))[0]) if A.shape[1] <= TW_LIMIT else None
    Af = A.astype(float)
    lp = linprog(np.ones(Af.shape[1]), A_ub=-Af, b_ub=-np.ones(Af.shape[0]),
                 bounds=(0, 1), method="highs")
    t0 = time.perf_counter()
    r = milp(c=np.ones(Af.shape[1]),
             constraints=LinearConstraint(Af, lb=np.ones(Af.shape[0]), ub=np.inf),
             integrality=np.ones(Af.shape[1]), bounds=Bounds(0, 1),
             options={"time_limit": 300})
    el = time.perf_counter() - t0
    k = None if r.x is None else int(round(r.x.sum()))
    return dict(n_b=int(A.shape[1]), n_t=int(A.shape[0]), n_ZZ=d["n_ZZ"],
                n_components=d["n_components"], largest_block=d["largest_block"],
                direct_sum=d["direct_sum"], treewidth_ub=tw,
                lp=round(float(lp.fun), 4), k_opt=k, ilp_status=int(r.status),
                lp_gap=None if k is None else round(k - float(lp.fun), 4),
                seconds=round(el, 4))


def main() -> None:
    tla, tlo, _ = load_municipalities("centroid")
    bbox = (tla.min(), tla.max(), tlo.min(), tlo.max())
    poly = frontier_polyline()
    out = {"seed": SEED, "tw_limit": TW_LIMIT, "regimes": {}}
    for kind, radii in (("frontier", (300, 200, 150, 100, 75, 50)),
                        ("grid2d", (400, 300, 200, 150, 100)),
                        ("poisson2d", (400, 300, 200, 150, 120))):
        rng = np.random.default_rng(SEED)
        rows = []
        print(f"\n=== {kind} ===")
        print(f"{'R':>5} {'n_B':>6} {'n_ZZ':>7} {'blk':>5} {'tw<=':>5} {'LP':>9} "
              f"{'k_opt':>6} {'gap':>6} {'t (s)':>8} {'cov':>6}")
        for R in radii:
            A, cov = build(kind, float(R), bbox, poly, rng)
            r = analyse(A); r.update(R_km=R, covered=round(cov, 4))
            rows.append(r)
            print(f"{R:>5} {r['n_b']:>6} {r['n_ZZ']:>7} {r['n_components']:>5} "
                  f"{str(r['treewidth_ub']):>5} {r['lp']:>9.2f} {str(r['k_opt']):>6} "
                  f"{str(r['lp_gap']):>6} {r['seconds']:>8.3f} {cov:>6.3f}", flush=True)
        out["regimes"][kind] = rows
        tw = [(r["n_b"], r["treewidth_ub"]) for r in rows if r["treewidth_ub"]]
        if len(tw) >= 3:
            p = np.polyfit(np.log([a for a, _ in tw]), np.log([b for _, b in tw]), 1)
            out["regimes"][kind + "_tw_exponent"] = round(float(p[0]), 3)
            print(f"  treewidth scaling: tw ~ n_B^{p[0]:.2f}")
    json.dump(out, open(RESULTS_DIR / "dimension_sweep.json", "w"), indent=1)
    print("\n-> results/dimension_sweep.json")


if __name__ == "__main__":
    main()
