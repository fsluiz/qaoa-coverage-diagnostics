#!/usr/bin/env python3
"""
Search for a PEF-based m-cover instance where the GREEDY heuristic OVERSHOOTS
the optimum (greedy_k > k_opt). Such an instance makes the QWOA k-descent
non-trivial: QWOA must descend through k_greedy -> ... -> k_opt sectors.

Reuses the exact greedy / ILP / brute logic from benchmark_qwoa_qaoa.py so the
winning instance plugs straight into that benchmark.

Sweep axes:
  * cluster_seed : which real PEF base anchors the n_qubits-base cluster
  * task_seed    : RNG seed for uniform task placement in the cluster bbox
  * n_qubits     : size of the base subset (kept small for tractable QWOA sectors)
  * R (km)       : coverage radius (density knob)
  * m            : coverage multiplicity (1=set cover, 2/3 = k-domination)

Winners are brute-verified, scored by (descent depth = greedy - k_opt, sector
tractability), and the best is frozen to overshoot_instance.json.

Pure NumPy/SciPy.  Author: assistant for F. S. Luiz (IFGW/UNICAMP).
"""
from __future__ import annotations
import json
import time
import math
from itertools import combinations

import numpy as np
import pandas as pd

# reuse the canonical implementations (import is side-effect-free: main() is guarded)
from benchmark_qwoa_qaoa import (
    haversine_km, greedy_mcover, ilp_mcover, precompute, classical_optimum,
)

from _paths import DATA_DIR

# ---- sweep grid (trim/expand as needed) ------------------------------------
CLUSTER_SEEDS = list(range(0, 24, 2))      # 12 anchor bases
TASK_SEEDS = list(range(25))               # 25 task layouts
N_QUBITS_LIST = [12, 14]                    # C(14,7)=3432 sector still tractable
R_LIST = list(range(120, 561, 40))         # 11 radii (km)
M_LIST = [1, 2, 3]
N_TASKS = 23
MAX_SECTOR_DIM = 6000                       # cap C(n, k_greedy) for downstream QWOA


def _pairwise_km(lats, lons):
    n = len(lats)
    D = np.zeros((n, n))
    for i in range(n):
        D[i] = haversine_km(lats[i], lons[i], lats, lons)
    return D


def load_bases():
    df = pd.read_excel(DATA_DIR / "pef_planilha.xlsx")
    cols = {c.lower(): c for c in df.columns}
    def pick(*cands):
        for c in cands:
            for lc, orig in cols.items():
                if c in lc:
                    return orig
        raise KeyError(cands)
    lat_c, lon_c = pick("lat"), pick("lon", "long")
    df = df.dropna(subset=[lat_c, lon_c]).reset_index(drop=True)
    return df[lat_c].to_numpy(float), df[lon_c].to_numpy(float)


def build_A(base_lats, base_lons, D, cluster_seed, task_seed, n_qubits, R, n_tasks):
    """Coverage matrix A_q[base, task] for a cluster of n_qubits real bases."""
    q_idx = sorted(np.argsort(D[cluster_seed])[:n_qubits].tolist())
    lats_q = base_lats[np.array(q_idx)]
    lons_q = base_lons[np.array(q_idx)]
    rng = np.random.RandomState(task_seed)
    mlat = 0.1 * (lats_q.max() - lats_q.min() + 1e-6)
    mlon = 0.1 * (lons_q.max() - lons_q.min() + 1e-6)
    tlat = rng.uniform(lats_q.min() - mlat, lats_q.max() + mlat, n_tasks)
    tlon = rng.uniform(lons_q.min() - mlon, lons_q.max() + mlon, n_tasks)
    lat1 = np.radians(lats_q[:, None]); lat2 = np.radians(tlat[None, :])
    lon1 = np.radians(lons_q[:, None]); lon2 = np.radians(tlon[None, :])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    dist = 2 * 6371.0 * np.arcsin(np.sqrt(a))
    A_q = (dist <= R).astype(int)
    return A_q, q_idx, lats_q, lons_q


def feasible_for_m(A_q, m):
    """Every task must be coverable to its target = min(m, c_t) with c_t >= 1."""
    c_t = A_q.sum(axis=0)
    return bool((c_t >= 1).all()) and bool((c_t >= np.minimum(m, c_t)).all())


def main():
    t0 = time.time()
    base_lats, base_lons = load_bases()
    D = _pairwise_km(base_lats, base_lons)
    print(f"Loaded {len(base_lats)} PEF bases. Sweeping...")

    hits = []
    n_eval = 0
    for cs in CLUSTER_SEEDS:
        for ts in TASK_SEEDS:
            for nq in N_QUBITS_LIST:
                for R in R_LIST:
                    A_q, q_idx, lats_q, lons_q = build_A(
                        base_lats, base_lons, D, cs, ts, nq, R, N_TASKS)
                    if A_q.sum(axis=0).min() < 1:
                        continue                      # an unreachable task
                    for m in M_LIST:
                        if not feasible_for_m(A_q, m):
                            continue
                        n_eval += 1
                        k_greedy, _ = greedy_mcover(A_q, m)
                        k_ilp, ilp_sel, status = ilp_mcover(A_q, m)
                        if k_ilp is None or k_greedy <= k_ilp:
                            continue
                        # OVERSHOOT found -> brute-verify the optimum
                        pc = precompute(A_q, m)
                        k_opt, opt_states = classical_optimum(pc)
                        if k_opt is None or k_greedy <= k_opt:
                            continue
                        sector_dim = math.comb(nq, k_greedy)
                        hits.append(dict(
                            cluster_seed=cs, task_seed=ts, n_qubits=nq, R=R, m=m,
                            n_tasks=N_TASKS,
                            k_greedy=int(k_greedy), k_ilp=int(k_ilp), k_opt=int(k_opt),
                            overshoot=int(k_greedy - k_opt),
                            n_feasible=int(pc["feasible"].sum()),
                            n_opt=int(len(opt_states)),
                            sector_dim_greedy=int(sector_dim),
                            mean_k_t=float(A_q.sum(axis=0).mean()),
                            q_idx=[int(i) for i in q_idx],
                            A_q=A_q.tolist(),
                            lats_q=lats_q.tolist(), lons_q=lons_q.tolist(),
                        ))
    dt = time.time() - t0
    print(f"Evaluated {n_eval} instances in {dt:.1f}s -> {len(hits)} overshoot hits.")
    if not hits:
        print("No overshoot found. Widen the grid (more task_seeds / R / n_qubits).")
        return

    # rank: bigger descent depth first, then tractable sector, then more feasible mass
    def score(h):
        tractable = h["sector_dim_greedy"] <= MAX_SECTOR_DIM
        return (h["overshoot"], int(tractable), -h["sector_dim_greedy"], h["n_feasible"])
    hits.sort(key=score, reverse=True)

    print("\nTop overshoot instances:")
    print(f"{'cs':>3} {'ts':>3} {'nq':>3} {'R':>4} {'m':>2} "
          f"{'grdy':>4} {'opt':>4} {'over':>4} {'dimC':>7} {'#feas':>6}")
    for h in hits[:15]:
        print(f"{h['cluster_seed']:>3} {h['task_seed']:>3} {h['n_qubits']:>3} "
              f"{h['R']:>4} {h['m']:>2} {h['k_greedy']:>4} {h['k_opt']:>4} "
              f"{h['overshoot']:>4} {h['sector_dim_greedy']:>7} {h['n_feasible']:>6}")

    # freeze the best tractable instance for the QWOA descent benchmark
    best = next((h for h in hits if h["sector_dim_greedy"] <= MAX_SECTOR_DIM), hits[0])
    json.dump(best, open(DATA_DIR / "overshoot_instance.json", "w"), indent=2)
    print(f"\nFrozen best instance -> overshoot_instance.json")
    print(f"  cluster_seed={best['cluster_seed']} task_seed={best['task_seed']} "
          f"n_qubits={best['n_qubits']} R={best['R']}km m={best['m']}")
    print(f"  greedy={best['k_greedy']}  k_opt={best['k_opt']}  "
          f"overshoot={best['overshoot']}  descent sectors: "
          f"{best['k_greedy']} -> {best['k_opt']} "
          f"(dim C({best['n_qubits']},{best['k_greedy']})={best['sector_dim_greedy']})")


if __name__ == "__main__":
    main()
