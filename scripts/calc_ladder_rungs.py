#!/usr/bin/env python3
"""The geographic ladder, read through the factorisation.

Three rungs, three answers to the same pair of questions -- does the cost operator come
apart, and is the circuit that encodes it classically simulable?

  Rung 1  PEF frontier, 24 posts   -- 13 blocks of <= 4; the state is an exact product, so
                                      the ideal reference is computable (calc_block_factorisation)
  Rung 2  Arc of Fire, 79 posts    -- 70 blocks of <= 3; 2^79 is out of reach but the ideal
                                      distribution is not, because it factorises
  Rung 3  Legal Amazon, 773 municipality centroids -- one block of 770; neither mechanism
                                      applies, and the strip bound uses the density it requires

The strip bound tw <= 2 rho R H needs the MAXIMUM post density over strip positions.  Using
the mean understates it on real geography, where centroids cluster along rivers and roads, so the
maximum is measured here by sweeping a 2R strip across both axes.

Writes results/ladder_rungs.json.
"""
from __future__ import annotations

import json
import math
import time

import networkx as nx
import numpy as np
from scipy.optimize import minimize
from networkx.algorithms.approximation import treewidth_min_fill_in

from _paths import DATA_DIR, RESULTS_DIR
from calc_block_factorisation import blocks_of, load, qaoa_state
from calc_dimension_sweep import hav
from calc_ilp_geographic import haversine_matrix, load_municipalities


def optimise_angles(tag, grid=41):
    """Rung 2's angles were transferred from Rung 1.  Because the instance factorises, the
    ideal optimal-cover weight is a convolution and costs almost nothing to evaluate, so the
    angles can simply be optimised -- which separates "these angles cannot reach it" from
    "this ansatz cannot reach it at any angles"."""
    f = lambda gb: ideal_optimum_probability(tag, gamma=gb[0], beta=gb[1])["p_ideal"]
    d = json.load(open(DATA_DIR / tag))
    g0 = d["qaoa_angles"]["1"]["gamma"][0]
    b0 = d["qaoa_angles"]["1"]["beta"][0]
    best = (f((g0, b0)), g0, b0)
    transferred = best[0]
    for g in np.linspace(-np.pi / 2, np.pi / 2, grid):
        for b in np.linspace(-np.pi / 2, np.pi / 2, grid):
            v = f((g, b))
            if v > best[0]:
                best = (v, g, b)
    r = minimize(lambda x: -f(x), x0=[best[1], best[2]], method="Nelder-Mead",
                 options={"xatol": 1e-4, "fatol": 1e-22, "maxiter": 600})
    p_max = float(-r.fun)
    return dict(gamma_transferred=g0, beta_transferred=b0,
                p_ideal_transferred=transferred,
                gamma_opt=float(r.x[0]), beta_opt=float(r.x[1]),
                p_ideal_optimised=p_max,
                gain_from_optimising=p_max / transferred,
                shots_for_one_hit_optimised=1.0 / p_max,
                ansatz_limited=bool(1.0 / p_max > 8192))


def ideal_optimum_probability(tag, gamma=None, beta=None):
    """Ideal p=1 QAOA weight of the optimal covers, obtained blockwise.

    Every target's cover-set lies inside one block, so both the coverage constraint and the
    cardinality constraint factorise: per block, a polynomial in weight whose coefficients are
    the ideal probability (and the count) of patterns covering that block's own targets.
    """
    d, A, h, J = load(tag)
    n, k_opt, n_t = d["n_b"], d["k_opt"], A.shape[0]
    g = d["qaoa_angles"]["1"]["gamma"][0] if gamma is None else gamma
    b = d["qaoa_angles"]["1"]["beta"][0] if beta is None else beta
    blk = blocks_of(n, J)

    owner = np.full(n_t, -1)
    for bi, sb in enumerate(blk):
        for t in range(n_t):
            if A[t, sb].sum() > 0:
                assert owner[t] in (-1, bi), "a target straddles two blocks: not a direct sum"
                owner[t] = bi
    assert (owner >= 0).all(), "some target is unreachable"

    probs, counts = [], []
    for bi, sb in enumerate(blk):
        m = len(sb)
        pr = np.abs(qaoa_state(list(sb), h, J, g, b)) ** 2
        mine = np.where(owner == bi)[0]
        pp, cc = {}, {}
        for code in range(2 ** m):
            bits = np.array([(code >> (m - 1 - t)) & 1 for t in range(m)])
            if len(mine) and not (A[np.ix_(mine, sb)] @ bits >= 1).all():
                continue
            w = int(bits.sum())
            pp[w] = pp.get(w, 0.0) + pr[code]
            cc[w] = cc.get(w, 0) + 1
        probs.append(pp)
        counts.append(cc)

    def convolve(polys, unit):
        acc = {0: unit}
        for poly in polys:
            new: dict = {}
            for w0, p0 in acc.items():
                for w1, p1 in poly.items():
                    w = w0 + w1
                    if w <= k_opt:
                        new[w] = new.get(w, type(unit)()) + p0 * p1
            acc = new
        return acc

    p_ideal = convolve(probs, 1.0).get(k_opt, 0.0)
    n_opt = convolve(counts, 1).get(k_opt, 0)
    in_sector = math.comb(n, k_opt)
    return dict(n_b=n, k_opt=k_opt, n_blocks=len(blk),
                largest_block=max(len(s) for s in blk),
                n_optimal_covers=int(n_opt), in_sector_strings=float(in_sector),
                p_ideal=float(p_ideal),
                baseline=float(n_opt / in_sector),
                eta_ideal=float(p_ideal / (n_opt / in_sector)) if n_opt else None,
                shots_for_one_ideal_hit=float(1 / p_ideal) if p_ideal > 0 else None)


def strip_bound(lat, lon, R):
    """Worst occupancy of a 2R-wide strip: the density the bound actually requires."""
    out = {}
    for axis, name in ((0, "north_south"), (1, "east_west")):
        if axis == 0:
            proj = np.array([hav(la, lon.mean(), lat.min(), lon.mean()) for la in lat])
        else:
            # each point's own latitude: projecting along a shared reference parallel would
            # mis-scale km-per-degree away from it and could exceed the true separation,
            # which would void the non-expansiveness the pathwidth argument relies on
            proj = np.array([hav(la, lo, la, lon.min()) for la, lo in zip(lat, lon)])
        order = np.sort(proj)
        worst = max(int(((order >= s) & (order <= s + 2 * R)).sum()) for s in order)
        out[name] = dict(extent_km=round(float(order.max()), 1), worst_strip=worst,
                         pathwidth_bound=worst - 1)
    return out


def uniform_reference(n_b: int, R: float, draws: int = 200, seed: int = 20260930) -> dict:
    """The strip bound for n_b points uniform (on the sphere) inside the official outline.

    Replaces a mean-density comparator built on the latitude-longitude bounding box, whose
    empty area outside the Legal Amazon inflated the apparent clustering.  The polygon area
    is the acceptance fraction times the spherical area of the bounding box.
    """
    from shapely.geometry import shape
    try:                                             # shapely >= 2
        from shapely import contains_xy
    except ImportError:                              # shapely 1.8 (locked environment)
        from shapely.vectorized import contains as contains_xy
    geo = json.load(open(DATA_DIR / "legal_amazon_outline_2024.geojson"))
    poly = shape(geo["features"][0]["geometry"])
    lo0, la0, lo1, la1 = poly.bounds
    rng = np.random.default_rng(seed)
    s0, s1 = np.sin(np.radians(la0)), np.sin(np.radians(la1))
    box_area = 6371.0 ** 2 * np.radians(lo1 - lo0) * (s1 - s0)

    def draw(k):
        lat_, lon_ = np.empty(0), np.empty(0)
        tried = hit = 0
        while len(lat_) < k:
            lo = rng.uniform(lo0, lo1, 4 * k)
            la = np.degrees(np.arcsin(rng.uniform(s0, s1, 4 * k)))
            inside = contains_xy(poly, lo, la)
            tried, hit = tried + len(lo), hit + int(inside.sum())
            lat_, lon_ = np.r_[lat_, la[inside]], np.r_[lon_, lo[inside]]
        return lat_[:k], lon_[:k], hit / tried

    widths, frac = [], []
    for _ in range(draws):
        la, lo, f = draw(n_b)
        frac.append(f)
        widths.append(min(v["pathwidth_bound"] for v in strip_bound(la, lo, R).values()))
    widths = np.array(widths)
    return dict(draws=draws, seed=seed, polygon_area_km2=round(float(box_area * np.mean(frac)), -3),
                bounding_box_area_km2=round(float(box_area), -3),
                strip_bound_mean=round(float(widths.mean()), 1),
                strip_bound_ci95=[float(np.percentile(widths, 2.5)), float(np.percentile(widths, 97.5))],
                strip_bound_max=int(widths.max()))


def main() -> None:
    out = {"R_km": 200.0}

    print("=== Rung 2: the ideal reference the 79-qubit null never had ===")
    r2 = ideal_optimum_probability("frozen_arco.json")
    out["rung2"] = r2
    for k, v in r2.items():
        print(f"  {k:>26}  {v}")

    print("\n--- and at optimised angles, to separate the angles from the ansatz ---")
    ang = optimise_angles("frozen_arco.json")
    out["rung2_angles"] = ang
    for k, v in ang.items():
        print(f"  {k:>26}  {v}")

    print("\n=== Rung 3: neither mechanism applies ===")
    lat, lon, _ = load_municipalities("centroid")
    R = out["R_km"]
    D = haversine_matrix(lat, lon)
    A = (D <= R).astype(np.int64)
    np.fill_diagonal(A, 1)
    C = (A.T @ A > 0).astype(np.int8)
    np.fill_diagonal(C, 0)
    n_b, n_zz = len(lat), int(C.sum() // 2)
    G = nx.from_numpy_array(C)
    t0 = time.perf_counter()
    tw = int(treewidth_min_fill_in(G)[0])
    secs = time.perf_counter() - t0

    # min-fill is an UPPER bound and can never show a contraction is expensive.  A clique is
    # a valid LOWER bound (tw >= omega - 1); a greedy maximal clique suffices for that.
    deg = dict(G.degree())
    omega = 0
    for seed in sorted(deg, key=deg.get, reverse=True)[:400]:
        cand = set(G[seed])
        clique = {seed}
        while cand:
            v = max(cand, key=lambda x: len(cand & set(G[x])))
            clique.add(v)
            cand &= set(G[v])
        omega = max(omega, len(clique))

    sb = strip_bound(lat, lon, R)
    honest = min(v["pathwidth_bound"] for v in sb.values())
    uni = uniform_reference(n_b, R)
    area = uni["polygon_area_km2"]
    mean_based = 2 * (n_b / area) * R * max(v["extent_km"] for v in sb.values())

    out["rung3"] = dict(n_b=n_b, n_ZZ=n_zz, mean_degree=round(2 * n_zz / n_b, 1),
                        treewidth_ub=tw, treewidth_seconds=round(secs, 1),
                        max_clique_found=int(omega), treewidth_lb=int(omega - 1),
                        strip=sb, strip_bound_max_density=honest,
                        strip_bound_mean_density=round(float(mean_based), 1),
                        uniform_in_outline=uni,
                        clustering_factor=round(honest / uni["strip_bound_mean"], 2))
    for k, v in out["rung3"].items():
        print(f"  {k:>26}  {v}")
    print(f"\n  real centroids vs uniform points in the official outline: "
          f"{out['rung3']['clustering_factor']}x")
    print(f"  treewidth bracketed [{omega - 1}, {tw}]; contraction costs at least "
          f"2^{omega - 1}, which is what makes this rung's circuit unsimulable")

    json.dump(out, open(RESULTS_DIR / "ladder_rungs.json", "w"), indent=1)
    print("\n-> results/ladder_rungs.json")


if __name__ == "__main__":
    main()
