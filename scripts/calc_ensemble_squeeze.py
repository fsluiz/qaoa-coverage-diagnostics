#!/usr/bin/env python3
"""Ensemble test of the cost/triviality coupling across instances and register sizes.

The squeeze was established on one frozen 24-base instance whose waypoint seed was selected
to give a greedy cover of ten.  This script removes both restrictions: bases are drawn from
the 773 official IBGE Legal Amazon municipality centroids, register sizes span
n_B in {20, 40, 80, 160}, and waypoints are drawn with NO acceptance filter beyond
feasibility.  For every instance and radius we record the circuit cost n_ZZ, the
propagation-fixed fraction phi, and the residual search space.

Writes results/ensemble_squeeze.json.
"""
from __future__ import annotations

import csv
import json
import sys
from math import comb

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from _paths import DATA_DIR, RESULTS_DIR
from calc_triviality import n_zz_of, propagate

EARTH_R_KM = 6371.0088
N_B_VALUES = (20, 40, 80, 160)
N_INSTANCES = 25
R_FRACTIONS = (0.30, 0.45, 0.60, 0.80, 1.00, 1.30)   # of the mean nearest-neighbour spacing
ILP_TIME_LIMIT = 30.0


def load_sites() -> np.ndarray:
    rows = list(csv.DictReader(open(DATA_DIR / "ibge_amazonia_legal_2024.csv", encoding="utf-8")))
    return np.array([[float(r["centroid_lat"]), float(r["centroid_lon"])] for r in rows])


def haversine(a_lat, a_lon, b_lat, b_lon):
    p1, p2 = np.radians(a_lat)[:, None], np.radians(b_lat)[None, :]
    dl = np.radians(b_lon)[None, :] - np.radians(a_lon)[:, None]
    h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def make_instance(sites, n_b, n_t, rng, R):
    idx = rng.choice(len(sites), n_b, replace=False)
    b = sites[idx]
    anchor = rng.integers(0, n_b, size=n_t)
    off = rng.uniform(0.0, 0.7 * R, size=n_t)
    ang = rng.uniform(0.0, 2 * np.pi, size=n_t)
    tlat = b[anchor, 0] + off * np.cos(ang) / 111.0
    tlon = b[anchor, 1] + off * np.sin(ang) / (111.0 * np.cos(np.radians(b[anchor, 0])))
    A = (haversine(tlat, tlon, b[:, 0], b[:, 1]) <= R).astype(int)
    return A[A.sum(1) >= 1]                                   # keep reachable targets only


def solve(A):
    """Return (k, status, gap).  status == 0 is HiGHS "optimal"; status == 1 is a time or
    iteration limit, in which case `r.x` still holds the incumbent and the value is NOT a
    proven optimum.  The caller must not conflate the two."""
    r = milp(c=np.ones(A.shape[1]),
             constraints=LinearConstraint(A, lb=np.ones(A.shape[0]), ub=np.inf),
             integrality=np.ones(A.shape[1]), bounds=Bounds(0, 1),
             options={"time_limit": ILP_TIME_LIMIT})
    if r.x is None:
        return None, int(r.status), None
    gap = getattr(r, "mip_gap", None)
    return int(round(r.x.sum())), int(r.status), (None if gap is None else float(gap))


def main() -> None:
    sites = load_sites()
    D_all = haversine(sites[:, 0], sites[:, 1], sites[:, 0], sites[:, 1])
    np.fill_diagonal(D_all, np.inf)
    spacing = float(np.median(D_all.min(1)))                  # mean nearest-neighbour spacing
    print(f"{len(sites)} IBGE sites; median nearest-neighbour spacing {spacing:.1f} km")

    rows = []
    for n_b in N_B_VALUES:
        # radius scale set by the spacing of a random n_b-subset, so R/delta is comparable
        rng0 = np.random.default_rng(0)
        sub = sites[rng0.choice(len(sites), n_b, replace=False)]
        Ds = haversine(sub[:, 0], sub[:, 1], sub[:, 0], sub[:, 1])
        np.fill_diagonal(Ds, np.inf)
        delta = float(np.median(Ds.min(1)))
        for f in R_FRACTIONS:
            R = f * delta * 3.0
            for s in range(N_INSTANCES):
                rng = np.random.default_rng(10_000 * n_b + 100 * int(f * 100) + s)
                A = make_instance(sites, n_b, n_b, rng, R)
                if A.shape[0] < 5:
                    continue
                k, status, gap = solve(A)
                if k is None:
                    continue
                p = propagate(A, np.ones(A.shape[0], int))
                free_b, need_k = A.shape[1] - p["forced"], k - p["forced"]
                rows.append(dict(n_b=n_b, R_km=round(R, 1), R_over_delta=round(R / delta, 3),
                                 seed=s, n_t=int(A.shape[0]), n_ZZ=n_zz_of(A), k_opt=k,
                                 forced=p["forced"], phi=round(p["forced"] / k, 4),
                                 deg1=int((A.sum(1) == 1).sum()),
                                 residual=comb(free_b, need_k) if 0 <= need_k <= free_b else None,
                                 ilp_status=status, mip_gap=gap,
                                 mean_deg=round(float(A.sum(1).mean()), 3)))
            done = [r for r in rows if r["n_b"] == n_b and abs(r["R_over_delta"] - R / delta) < 1e-9]
            if done:
                print(f"  n_B={n_b:4}  R/delta={R/delta:5.2f}  n={len(done):3}  "
                      f"n_ZZ={np.mean([r['n_ZZ'] for r in done]):8.1f}  "
                      f"phi={np.mean([r['phi'] for r in done]):.3f}", flush=True)

    n_proven = sum(1 for r in rows if r["ilp_status"] == 0)
    print(f"\nILP status: {n_proven}/{len(rows)} proven optimal "
          f"(status 0); {len(rows) - n_proven} hit the {ILP_TIME_LIMIT:.0f} s cap")
    out = dict(meta=dict(n_sites=len(sites), spacing_km=spacing, n_b_values=list(N_B_VALUES),
                         n_instances=N_INSTANCES, r_fractions=list(R_FRACTIONS),
                         note="bases drawn from IBGE Legal Amazon centroids; NO greedy filter"),
               rows=rows, n_proven_optimal=n_proven)
    # correlation between circuit cost and triviality, pooled and per register size
    import scipy.stats as st
    lz = np.log10([max(r["n_ZZ"], 1) for r in rows]); ph = [r["phi"] for r in rows]
    out["corr_pooled"] = dict(pearson=float(st.pearsonr(lz, ph)[0]),
                              spearman=float(st.spearmanr(lz, ph)[0]), n=len(rows))
    out["corr_by_n_b"] = {}
    for n_b in N_B_VALUES:
        sel = [r for r in rows if r["n_b"] == n_b]
        if len(sel) > 5:
            out["corr_by_n_b"][str(n_b)] = dict(
                spearman=float(st.spearmanr(np.log10([max(r["n_ZZ"], 1) for r in sel]),
                                            [r["phi"] for r in sel])[0]), n=len(sel))
    json.dump(out, open(RESULTS_DIR / "ensemble_squeeze.json", "w"), indent=1)
    print(f"\npooled: {len(rows)} instances, Spearman(log n_ZZ, phi) = "
          f"{out['corr_pooled']['spearman']:+.3f}")
    for k, v in out["corr_by_n_b"].items():
        print(f"  n_B={k:>4}: Spearman = {v['spearman']:+.3f}  (n={v['n']})")
    print("-> results/ensemble_squeeze.json")


if __name__ == "__main__":
    main()
