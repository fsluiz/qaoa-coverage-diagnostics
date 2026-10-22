#!/usr/bin/env python3
"""Corrected analysis of the regular-grid ILP hardness windows.

The earlier analysis (`calc_symmetry_breaking.py`, mode B) reported that the
R/delta timeout windows "dissolve under symmetry breaking": 300 s+ -> 2.1 s with
k_opt = 532.  That conclusion was wrong, and this script documents why.

Mode B added, for every pair (b, b') of EQUAL DEGREE with b < b', the constraint
x_b >= x_b'.  Equal degree is not a graph automorphism, so these are not valid
symmetry-breaking constraints.  What they actually enforce is that the selected
set be an index PREFIX within each degree class, which collapses the feasible
region from 2^625 points to (|C_2|+1)(|C_3|+1)(|C_4|+1) = 246,450 points and
excludes the optimum.  The 2.1 s "solve" is that collapse, and 532 is 3.8x the
true domination number of the 25x25 grid.

Valid automorphism-based symmetry breaking cannot produce such a speedup:
Aut(P_n box P_n) is the dihedral group D4 of order 8, so orbit pruning can remove
at most a factor 8 of the search tree.

Writes results/ilp_symmetry_corrected.json.
"""
from __future__ import annotations

import json
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from _paths import DATA_DIR, RESULTS_DIR

N_GRID = 25
DELTA_KM = 10.0
N = N_GRID * N_GRID
AUT_ORDER = 8  # |D4| for the square grid P_n box P_n, n >= 2


def positions() -> np.ndarray:
    return np.array([(i * DELTA_KM, j * DELTA_KM)
                     for i in range(N_GRID) for j in range(N_GRID)], float)


def distances() -> np.ndarray:
    p = positions()
    return np.linalg.norm(p[:, None] - p[None, :], axis=-1)


def closed_adj(D: np.ndarray, R: float) -> np.ndarray:
    return (D <= R).astype(float)


def lp_bound(Nc: np.ndarray) -> float:
    r = linprog(np.ones(N), A_ub=-Nc, b_ub=-np.ones(N), bounds=(0, 1), method="highs")
    return float(r.fun)


def heuristic_cover(Nc: np.ndarray) -> np.ndarray:
    """Greedy cover followed by redundancy pruning; returns a certified dominating set."""
    x = np.zeros(N, int)
    cov = np.zeros(N)
    while (cov < 1).any():
        u = np.where(cov < 1)[0]
        gains = Nc[u].sum(0) * (1 - x)
        x[int(gains.argmax())] = 1
        cov = Nc @ x
    for v in np.where(x == 1)[0]:
        x[v] = 0
        if not (Nc @ x >= 1).all():
            x[v] = 1
    return x


def diagonal_cover(Nc: np.ndarray) -> np.ndarray:
    """Period-5 diagonal pattern + repair; the classical grid-domination construction."""
    best = None
    for c in range(5):
        x = np.zeros(N, int)
        for i in range(N_GRID):
            for j in range(N_GRID):
                if (i + 2 * j) % 5 == c:
                    x[i * N_GRID + j] = 1
        cov = Nc @ x
        while (cov < 1).any():
            u = np.where(cov < 1)[0]
            x[int((Nc[u].sum(0) * (1 - x)).argmax())] = 1
            cov = Nc @ x
        for v in np.where(x == 1)[0]:
            x[v] = 0
            if not (Nc @ x >= 1).all():
                x[v] = 1
        if best is None or x.sum() < best.sum():
            best = x.copy()
    return best


def mode_b_constraints(D: np.ndarray, R: float):
    """Rebuild the invalid mode-B constraint set: x_b >= x_b' for equal-degree b < b'."""
    deg = ((D > 0) & (D <= R)).sum(1).astype(int)
    rows, npairs = [], 0
    classes: dict[int, list[int]] = {}
    for v in range(N):
        classes.setdefault(int(deg[v]), []).append(v)
    for members in classes.values():
        for a, b in zip(members[:-1], members[1:]):     # chain form, same feasible set
            r = np.zeros(N)
            r[a], r[b] = 1.0, -1.0
            rows.append(r)
        npairs += len(members) * (len(members) - 1) // 2
    return np.array(rows) if rows else np.zeros((0, N)), npairs, classes


def main() -> None:
    D = distances()
    out: dict = {"grid": {"n": N_GRID, "delta_km": DELTA_KM, "n_nodes": N},
                 "aut_group": {"group": "D4", "order": AUT_ORDER,
                               "max_pruning_factor": AUT_ORDER,
                               "note": "Aut(P_n box P_n) = D4; orbit pruning removes at "
                                       "most |Aut| of the tree, so it cannot explain 300 s -> 2 s."}}

    # ---- the R/delta = 1.0 window: what is the true optimum? ----------------
    R = 1.0 * DELTA_KM
    Nc = closed_adj(D, R)
    lb = lp_bound(Nc)
    ub_greedy = int(heuristic_cover(Nc).sum())
    ub_diag = int(diagonal_cover(Nc).sum())
    formula = ((N_GRID + 2) * (N_GRID + 2)) // 5 - 4
    out["window_R_over_delta_1"] = {
        "R_km": R, "lp_lower_bound": round(lb, 3), "lp_rounded_up": int(np.ceil(lb)),
        "ub_greedy": ub_greedy, "ub_diagonal_pattern": ub_diag,
        "closed_form_gamma": formula,
        "closed_form_ref": "gamma(P_n box P_m) = floor((n+2)(m+2)/5) - 4 for n,m >= 16",
        "certified_interval": [int(np.ceil(lb)), min(ub_greedy, ub_diag)],
    }
    print(f"R/delta = 1.0 : LP >= {lb:.2f} -> gamma >= {int(np.ceil(lb))}; "
          f"constructed cover {min(ub_greedy, ub_diag)}; closed form {formula}")

    # ---- reproduce mode B and show it excludes the optimum ------------------
    A_sym, npairs, classes = mode_b_constraints(D, R)
    cons = [LinearConstraint(Nc, lb=np.ones(N), ub=np.inf)]
    if len(A_sym):
        cons.append(LinearConstraint(A_sym, lb=np.zeros(len(A_sym)), ub=np.inf))
    t0 = time.perf_counter()
    res = milp(c=np.ones(N), constraints=cons, integrality=np.ones(N),
               bounds=Bounds(0, 1), options={"time_limit": 120})
    dt = time.perf_counter() - t0
    k_b = int(round(res.x.sum())) if res.x is not None else None
    prefix_space = 1
    for m in classes.values():
        prefix_space *= len(m) + 1
    is_prefix = None
    if res.x is not None:
        x = np.round(res.x).astype(int)
        is_prefix = all(
            all(x[m[i]] >= x[m[i + 1]] for i in range(len(m) - 1)) for m in classes.values()
        )
    out["mode_B_invalid"] = {
        "constraint": "x_b >= x_b' for every equal-degree pair b < b'",
        "n_equal_degree_pairs": npairs,
        "degree_class_sizes": {str(d): len(m) for d, m in sorted(classes.items())},
        "feasible_points_after_constraint": prefix_space,
        "k_returned": k_b, "t_s": round(dt, 2),
        "solution_is_per_class_prefix": is_prefix,
        "true_gamma_closed_form": formula,
        "excess_over_optimum": None if k_b is None else round(k_b / formula, 2),
        "verdict": "INVALID: equal degree is not an automorphism; the constraint forces an "
                   "index prefix per degree class and excludes the optimum.",
    }
    print(f"mode B        : k = {k_b} in {dt:.1f}s, per-class prefix = {is_prefix}, "
          f"{k_b / formula:.2f}x the true optimum ({formula})")
    print(f"                feasible space collapsed to {prefix_space:,} points")

    # ---- LP bound and absolute gap across the sweep -------------------------
    bench = json.load(open(DATA_DIR / "ilp_ratio_benchmark.json"))
    sweep = []
    print(f"\n{'R/d':>5} {'status':>8} {'k_opt':>6} {'LP':>8} {'abs gap':>8}")
    for row in bench:
        Nc_r = closed_adj(D, row["R_km"])
        lb_r = lp_bound(Nc_r)
        k = row["k_opt"]
        if k is None and abs(row["ratio"] - 1.0) < 1e-9 or (k is None and row["ratio"] in (1.2, 1.4)):
            k = formula                                  # closed-form optimum for this window
        gap = None if k is None else round(k - lb_r, 2)
        sweep.append(dict(ratio=row["ratio"], status=row["status"], k_opt=k,
                          lp_bound=round(lb_r, 3), abs_gap=gap, avg_deg=row["avg_deg"]))
        print(f"{row['ratio']:>5.1f} {row['status']:>8} {str(k):>6} {lb_r:>8.2f} "
              f"{'n/a' if gap is None else f'{gap:>8.2f}'}")
    out["lp_gap_sweep"] = sweep

    # ---- second window: certify bounds with a long MILP run -----------------
    Nc2 = closed_adj(D, 2.0 * DELTA_KM)
    lb2 = lp_bound(Nc2)
    r2 = milp(c=np.ones(N), constraints=LinearConstraint(Nc2, lb=np.ones(N), ub=np.inf),
              integrality=np.ones(N), bounds=Bounds(0, 1), options={"time_limit": 420})
    inc = int(round(r2.x.sum())) if r2.x is not None else None
    status = int(r2.status)          # 0 = proven optimal, 1 = time limit
    gap = float(getattr(r2, 'mip_gap', float('nan')))
    dual = getattr(r2, "mip_dual_bound", None)
    out["window_R_over_delta_2"] = {
        "R_km": 2.0 * DELTA_KM, "lp_lower_bound": round(lb2, 3),
        "incumbent": inc, "dual_bound": dual,
        "milp_status": status, "mip_rel_gap": gap,
        "proven_optimal": status == 0,
        "certified_interval": [None if dual is None else int(np.ceil(dual)), inc],
        "note": "absolute LP gap here is small and overlaps the solved ratios; gap size "
                "alone does not explain this window.  When milp_status is 1 the time "
                "limit was reached and `incumbent` is an upper bound, not the optimum.",
    }
    print(f"\nR/delta = 2.0 : LP {lb2:.2f}, dual bound {dual}, incumbent {inc}")
    solved = [s["abs_gap"] for s in sweep if s["status"] == "optimal" and s["abs_gap"] is not None]
    timeout = [s["abs_gap"] for s in sweep if s["status"] == "TIMEOUT" and s["abs_gap"] is not None]
    out["gap_summary"] = {"solved_max_abs_gap": max(solved), "solved_mean": round(float(np.mean(solved)), 2),
                          "timeout_abs_gaps": timeout}
    print(f"\nsolved rows : absolute LP gap <= {max(solved):.2f} (mean {np.mean(solved):.2f})")
    print(f"timeout rows: absolute LP gap = {timeout}")

    json.dump(out, open(RESULTS_DIR / "ilp_symmetry_corrected.json", "w"), indent=1)
    print("\n-> results/ilp_symmetry_corrected.json")


if __name__ == "__main__":
    main()
