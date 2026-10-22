#!/usr/bin/env python3
"""Does persistent surveillance break the decomposition, or only enlarge the instance?

The manuscript solves a static cover: a waypoint is "covered" when some base lies within
range of it.  Operationally that is not surveillance.  A drone on station leaves, and unless
another arrives the waypoint goes dark, so the honest formulation covers pairs (waypoint,
time) rather than waypoints -- persistent surveillance, or spatiotemporal set cover.

The question this script answers is not whether that formulation is bigger.  It is whether it
is structurally different, and there is a specific reason to think it might be: a drone that
occupies a post in one block at time t and a post in another block at t+1 creates an
interaction between two blocks that share no target.  Mobility is exactly the thing the
direct-sum argument assumes away.

Model.  y[b,t] = 1 iff a drone is on station at post b during slice t.
  coverage    sum_b A[j,b] y[b,t] >= 1            for every waypoint j and slice t
  fleet       sum_b y[b,t] <= N                   at most N drones airborne
  mobility    a drone leaving b at t can only reach posts within v*dt, enforced as a flow:
              y[b,t+1] <= sum_{b' in reach(b)} y[b',t]   (a post can only be occupied if some
              reachable post was occupied in the previous slice)
  endurance   no drone stays airborne more than L consecutive slices, enforced per post as
              sum_{s=t}^{t+L} y[b,s] <= L

Writes results/persistent_surveillance.json.
"""
from __future__ import annotations

import json
import time

import networkx as nx
import numpy as np
from networkx.algorithms.approximation import treewidth_min_fill_in
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from _paths import DATA_DIR, RESULTS_DIR

T_LIM = 99      # endurance longer than the horizon: the constraint does not bind
from calc_block_decomposition import interaction_graph
from calc_dimension_sweep import dmat


def load_rung1():
    d = json.load(open(DATA_DIR / "frozen_rung1.json"))
    A = np.asarray(d["A"], int)
    bla = np.asarray(d["base_lats"], float)
    blo = np.asarray(d["base_lons"], float)
    return d, A, bla, blo


def temporal_graph(A, reach, T):
    """Interaction graph of the time-expanded QUBO on the variables y[b,t]."""
    n_b = A.shape[1]
    n = n_b * T
    idx = lambda b, t: t * n_b + b
    C = np.zeros((n, n), np.int8)
    S = interaction_graph(A)                       # co-covering pairs, within a slice
    for t in range(T):
        for b in range(n_b):
            for c in range(n_b):
                if S[b, c]:
                    C[idx(b, t), idx(c, t)] = 1
    for t in range(T - 1):                         # mobility couples consecutive slices
        for b in range(n_b):
            for c in np.where(reach[b])[0]:
                C[idx(b, t), idx(c, t + 1)] = 1
                C[idx(c, t + 1), idx(b, t)] = 1
    np.fill_diagonal(C, 0)
    return C


def components(C):
    n, lab = connected_components(csr_matrix(C), directed=False)
    return int(n), int(np.bincount(lab).max())


def solve(A, reach, T, fleet, endurance):
    n_b = A.shape[1]
    n_t = A.shape[0]
    n = n_b * T
    idx = lambda b, t: t * n_b + b
    rows, lb, ub = [], [], []

    for t in range(T):                                     # coverage of every waypoint, always
        for j in range(n_t):
            r = np.zeros(n)
            r[[idx(b, t) for b in np.where(A[j] == 1)[0]]] = 1.0
            rows.append(r); lb.append(1.0); ub.append(np.inf)
    for t in range(T):                                     # fleet size
        r = np.zeros(n)
        r[[idx(b, t) for b in range(n_b)]] = 1.0
        rows.append(r); lb.append(-np.inf); ub.append(float(fleet))
    for t in range(T - 1):                                 # mobility
        for b in range(n_b):
            r = np.zeros(n)
            r[idx(b, t + 1)] = 1.0
            r[[idx(c, t) for c in np.where(reach[b])[0]]] = -1.0
            rows.append(r); lb.append(-np.inf); ub.append(0.0)
    for b in range(n_b):                                   # endurance
        for t in range(T - endurance):
            r = np.zeros(n)
            r[[idx(b, s) for s in range(t, t + endurance + 1)]] = 1.0
            rows.append(r); lb.append(-np.inf); ub.append(float(endurance))

    M = np.vstack(rows)
    c = np.ones(n)
    lp = linprog(c, A_ub=np.vstack([-M[np.isfinite(lb)], M[np.isfinite(ub)]]),
                 b_ub=np.concatenate([-np.array(lb)[np.isfinite(lb)],
                                      np.array(ub)[np.isfinite(ub)]]),
                 bounds=(0, 1), method="highs")
    t0 = time.perf_counter()
    r = milp(c=c, constraints=LinearConstraint(M, lb=np.array(lb), ub=np.array(ub)),
             integrality=np.ones(n), bounds=Bounds(0, 1),
             options={"time_limit": 600, "mip_rel_gap": 0.0})
    el = time.perf_counter() - t0
    k = None if r.x is None else int(round(r.x.sum()))
    return dict(status=int(r.status), drone_slices=k, seconds=round(el, 4),
                lp=None if lp.status else round(float(lp.fun), 4),
                lp_gap=None if (k is None or lp.status) else round(k - float(lp.fun), 4))


def main() -> None:
    d, A, bla, blo = load_rung1()
    n_b = A.shape[1]
    D = dmat(bla, blo, bla, blo)
    static_C = interaction_graph(A)
    s_comp, s_big = components(static_C)
    print(f"static instance: n_B={n_b}, {s_comp} blocks, largest {s_big}")

    out = {"static": {"n_b": n_b, "n_components": s_comp, "largest_block": s_big},
           "rows": []}
    # a drone covers v*dt km between slices; sweep how far it may travel
    for endurance, reach_km in ((T_LIM, 0.0), (T_LIM, 150.0), (T_LIM, 300.0),
                                (T_LIM, 600.0), (1, 600.0), (2, 600.0)):
        reach = (D <= reach_km)
        np.fill_diagonal(reach, True)                 # staying put is always allowed
        for T in (2, 3, 4):
            C = temporal_graph(A, reach, T)
            nc, big = components(C)
            G = nx.from_numpy_array(C)
            tw = int(treewidth_min_fill_in(G)[0]) if C.shape[0] <= 400 else None
            r = solve(A, reach, T, fleet=n_b, endurance=endurance)
            row = dict(reach_km=reach_km, T=T, endurance=endurance,
                       n_vars=int(C.shape[0]),
                       n_components=nc, largest_block=big, treewidth_ub=tw, **r)
            out["rows"].append(row)
            feas = "feasible" if r["status"] == 0 else "INFEASIBLE"
            print(f"  reach {reach_km:>5.0f} km  T={T}  L={endurance}  vars={C.shape[0]:>4}  "
                  f"blocks={nc:>3}  largest={big:>4}  tw<={str(tw):>4}  "
                  f"drone-slices={str(r['drone_slices']):>4}  gap={str(r['lp_gap']):>5}  "
                  f"{r['seconds']:>7.3f} s  {feas}", flush=True)
    # Why is it infeasible under a binding endurance?  A waypoint reachable by one base needs
    # that base occupied at every slice, which no finite-endurance drone sustains without
    # relief -- and for those waypoints no relief exists.  Test the attribution by removing
    # exactly those waypoints and re-solving.
    print("\n=== attribution of the infeasibility (endurance L=1, reach 600 km) ===")
    reach600 = (D <= 600.0)
    np.fill_diagonal(reach600, True)
    deg = A.sum(1)
    single = int((deg == 1).sum())
    attrib = {"singly_reachable_waypoints": single, "n_t": int(A.shape[0]), "rows": []}
    for tag, Ax in (("full instance", A), ("singly-reachable waypoints removed", A[deg > 1])):
        stat = []
        for T in (2, 3, 4):
            r = solve(Ax, reach600, T, fleet=Ax.shape[1], endurance=1)
            stat.append(r["status"] == 0)
        attrib["rows"].append(dict(variant=tag, feasible_by_horizon=stat))
        print(f"  {tag:>36}: " + "  ".join(f"T={T}:{'OK' if f else 'infeasible'}"
                                           for T, f in zip((2, 3, 4), stat)))
    attrib["caused_by_singly_reachable"] = (
        not any(attrib["rows"][0]["feasible_by_horizon"])
        and all(attrib["rows"][1]["feasible_by_horizon"]))
    print(f"  -> infeasibility attributable to the {single} singly-reachable waypoints: "
          f"{attrib['caused_by_singly_reachable']}")
    out["attribution"] = attrib

    json.dump(out, open(RESULTS_DIR / "persistent_surveillance.json", "w"), indent=1)
    print("\n-> results/persistent_surveillance.json")


if __name__ == "__main__":
    main()
