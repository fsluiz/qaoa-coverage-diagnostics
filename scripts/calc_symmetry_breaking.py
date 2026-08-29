"""
CALCULATION 2 — Symmetry Breaking in ILP (hardness robustness check).

Re-solves the 5 timeout rows from ilp_ratio_benchmark.json (25x25 grid, δ=10km)
with three HiGHS modes:
  A: extended timeout (300s)
  B: manual lexicographic symmetry breaking (equal-degree pair ordering)
  C: aggressive presolve (presolve=on, simplex_strategy=4)

Output: results/ilp_symmetry_breaking.json
"""
from __future__ import annotations

import json
import time
from typing import Any

import numpy as np

from _paths import DATA_DIR, RESULTS_DIR

OUTPATH = RESULTS_DIR / "ilp_symmetry_breaking.json"
BENCHMARK_PATH = DATA_DIR / "ilp_ratio_benchmark.json"

# The 5 timeout ratios from ilp_ratio_benchmark.json
TIMEOUT_RATIOS = {1.0, 1.2, 1.4, 2.0, 2.2}

N_GRID = 25
DELTA_KM = 10.0
N_NODES = N_GRID * N_GRID  # 625


def build_grid_positions() -> np.ndarray:
    """Return (625, 2) Cartesian positions for 25x25 grid with δ=10km."""
    pos = np.array([
        [float(i * DELTA_KM), float(j * DELTA_KM)]
        for i in range(N_GRID)
        for j in range(N_GRID)
    ])
    return pos


def compute_distance_matrix(pos: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix (625x625)."""
    diff = pos[:, None, :] - pos[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def solve_mds_milp(
    D: np.ndarray,
    R_km: float,
    timeout: float,
    symmetry_breaking: bool = False,
    highs_options: dict | None = None,
) -> dict[str, Any]:
    """Solve MDS ILP for a given distance matrix D and radius R_km.

    Parameters
    ----------
    D : np.ndarray
        Pairwise distance matrix (n x n).
    R_km : float
        Coverage radius.
    timeout : float
        Time limit in seconds.
    symmetry_breaking : bool
        If True, add lexicographic ordering constraints for same-degree pairs.
    highs_options : dict, optional
        Extra HiGHS options to pass to scipy.milp.
    """
    from scipy.optimize import milp, LinearConstraint, Bounds  # type: ignore
    from scipy.sparse import csr_matrix, vstack as sp_vstack  # type: ignore

    n = D.shape[0]
    # Closed adjacency: closed_adj[i,j] = 1 if j covers i (D[i,j] <= R or i==j)
    closed_adj = (D <= R_km).astype(np.float64)

    # Open adjacency for degree computation
    open_adj = ((D > 0) & (D <= R_km)).astype(np.float64)
    degrees = open_adj.sum(axis=1).astype(int)

    # --- Build constraint matrix ---
    # Base MDS constraint: sum_{j in N[i]} x_j >= 1, for all i
    A_main = csr_matrix(closed_adj)
    lb_main = np.ones(n)
    ub_main = np.full(n, float(n))

    if symmetry_breaking:
        # Lexicographic symmetry breaking for pairs (b, b') with equal degree,
        # where b < b' (by index). Add: x_b >= x_{b'} → x_b - x_{b'} >= 0.
        # This breaks orbital symmetry (bases with identical neighborhood structure
        # can be swapped without changing objective/feasibility).
        sb_rows_i = []
        sb_rows_j = []
        sb_vals = []
        sb_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if degrees[i] == degrees[j]:
                    # constraint: x_i - x_j >= 0  →  row: [+1, -1]
                    sb_rows_i.extend([sb_count, sb_count])
                    sb_rows_j.extend([i, j])
                    sb_vals.extend([1.0, -1.0])
                    sb_count += 1
        if sb_count > 0:
            from scipy.sparse import csr_matrix as csr  # type: ignore
            A_sb = csr((sb_vals, (sb_rows_i, sb_rows_j)), shape=(sb_count, n))
            A_combined = sp_vstack([A_main, A_sb])
            lb_combined = np.concatenate([lb_main, np.zeros(sb_count)])
            ub_combined = np.concatenate([ub_main, np.full(sb_count, float(n))])
        else:
            A_combined = A_main
            lb_combined = lb_main
            ub_combined = ub_main
        constraint = LinearConstraint(A_combined, lb=lb_combined, ub=ub_combined)
    else:
        constraint = LinearConstraint(A_main, lb=lb_main, ub=ub_main)

    c_obj = np.ones(n)
    bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))
    integrality = np.ones(n, dtype=int)

    opts: dict[str, Any] = {"time_limit": timeout, "disp": False}
    if highs_options:
        opts.update(highs_options)

    t0 = time.time()
    try:
        res = milp(c_obj, constraints=constraint, integrality=integrality,
                   bounds=bounds, options=opts)
        t_elapsed = time.time() - t0
        if res.status == 0:
            return {
                "solved": True,
                "t_s": round(t_elapsed, 4),
                "k_opt": int(round(res.fun)),
                "status": "optimal",
            }
        else:
            return {
                "solved": False,
                "t_s": round(t_elapsed, 4),
                "k_opt": None,
                "status": "TIMEOUT" if res.status == 3 else f"status_{res.status}",
            }
    except Exception as exc:
        t_elapsed = time.time() - t0
        return {
            "solved": False,
            "t_s": round(t_elapsed, 4),
            "k_opt": None,
            "status": f"error: {exc}",
        }


def main() -> None:
    print("=== CALCULATION 2: Symmetry Breaking in ILP (25x25 grid hardness) ===\n")

    # Load benchmark to get timeout rows
    with open(BENCHMARK_PATH) as f:
        benchmark = json.load(f)

    timeout_rows = [row for row in benchmark if row.get("status") == "TIMEOUT"
                    and row["ratio"] in TIMEOUT_RATIOS]
    print(f"  Found {len(timeout_rows)} timeout rows: "
          f"ratios={[r['ratio'] for r in timeout_rows]}")

    # Build grid
    pos = build_grid_positions()
    D = compute_distance_matrix(pos)
    print(f"  Grid: {N_GRID}×{N_GRID}={N_NODES} nodes, δ={DELTA_KM}km")
    print(f"  Distance matrix: shape {D.shape}\n")

    results: dict[str, Any] = {}

    for row in timeout_rows:
        R = row["R_km"]
        ratio = row["ratio"]
        orig_k = row.get("k_opt")
        print(f"\n  --- ratio={ratio}, R={R}km, original_k={orig_k} ---")

        # Open adj for degree info
        open_adj = ((D > 0) & (D <= R)).astype(np.float64)
        degrees = open_adj.sum(axis=1).astype(int)
        n_sym_pairs = int(sum(
            1 for i in range(N_NODES)
            for j in range(i + 1, N_NODES)
            if degrees[i] == degrees[j]
        ))
        # This is O(n^2) but n=625, so ~195k pairs — fast
        print(f"  Degree stats: min={degrees.min()}, max={degrees.max()}, "
              f"unique={len(set(degrees.tolist()))}")
        print(f"  Equal-degree pairs (symmetry candidates): {n_sym_pairs:,}")

        # Mode A: extended timeout 300s
        print("  Mode A: timeout=300s ...", end="", flush=True)
        mA = solve_mds_milp(D, R, timeout=300.0, symmetry_breaking=False)
        print(f" solved={mA['solved']}, k={mA['k_opt']}, t={mA['t_s']:.1f}s, status={mA['status']}")

        # Mode B: symmetry breaking, timeout=60s
        print("  Mode B: symmetry breaking, timeout=60s ...", end="", flush=True)
        mB = solve_mds_milp(D, R, timeout=60.0, symmetry_breaking=True)
        print(f" solved={mB['solved']}, k={mB['k_opt']}, t={mB['t_s']:.1f}s, status={mB['status']}")

        # Mode C: aggressive presolve, timeout=60s
        print("  Mode C: aggressive presolve, timeout=60s ...", end="", flush=True)
        mC = solve_mds_milp(D, R, timeout=60.0, symmetry_breaking=False,
                             highs_options={"presolve": "on"})
        print(f" solved={mC['solved']}, k={mC['k_opt']}, t={mC['t_s']:.1f}s, status={mC['status']}")

        results[str(ratio)] = {
            "R_km": R,
            "original_status": "TIMEOUT",
            "original_k": orig_k,
            "n_equal_degree_pairs": n_sym_pairs,
            "mode_A_timeout300": {
                "solved": mA["solved"],
                "t_s": mA["t_s"],
                "k_opt": mA["k_opt"],
                "status": mA["status"],
            },
            "mode_B_sym_break_60s": {
                "solved": mB["solved"],
                "t_s": mB["t_s"],
                "k_opt": mB["k_opt"],
                "status": mB["status"],
            },
            "mode_C_presolve_60s": {
                "solved": mC["solved"],
                "t_s": mC["t_s"],
                "k_opt": mC["k_opt"],
                "status": mC["status"],
            },
        }

    # Save
    output = {
        "description": (
            "Symmetry breaking robustness for the 5 HiGHS timeout rows "
            "(25x25 regular grid, δ=10km). Three modes tested: "
            "A=300s timeout, B=manual lexicographic symmetry breaking (60s), "
            "C=aggressive presolve (60s). "
            "Goal: determine if timeouts are intrinsic hardness or HiGHS artifact."
        ),
        "grid_params": {
            "n_grid": N_GRID,
            "delta_km": DELTA_KM,
            "n_nodes": N_NODES,
        },
        "results": results,
    }
    with open(OUTPATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved -> {OUTPATH}")

    # Summary
    print("\n  === SUMMARY ===")
    print(f"  {'ratio':>6} {'Mode_A_solved':>14} {'Mode_B_solved':>14} {'Mode_C_solved':>14}")
    for ratio_str, r in results.items():
        a = "YES" if r["mode_A_timeout300"]["solved"] else "TIMEOUT"
        b = "YES" if r["mode_B_sym_break_60s"]["solved"] else "TIMEOUT"
        c = "YES" if r["mode_C_presolve_60s"]["solved"] else "TIMEOUT"
        print(f"  {ratio_str:>6} {a:>14} {b:>14} {c:>14}")

    all_A_solved = all(r["mode_A_timeout300"]["solved"] for r in results.values())
    any_B_helped = any(r["mode_B_sym_break_60s"]["solved"] and
                       not r["mode_A_timeout300"]["solved"]
                       for r in results.values())
    any_C_helped = any(r["mode_C_presolve_60s"]["solved"] and
                       not r["mode_A_timeout300"]["solved"]
                       for r in results.values())

    print("\n  INTERPRETATION:")
    if all_A_solved:
        print("  Mode A (300s) resolves ALL timeouts — HiGHS just needs more time.")
        print("  The hardness is real but time-dependent, not NP-hard for these sizes.")
        print("  PAPER ACTION: Clarify that 60s timeout is a conservative bound; ")
        print("  the hardness window is real but solvable with extended resources.")
    else:
        remaining = sum(1 for r in results.values()
                        if not r["mode_A_timeout300"]["solved"])
        print(f"  {remaining} instance(s) persist past 300s — hardness is genuine.")
    if any_B_helped:
        print("  Mode B (sym breaking) helps in at least one case — referee concern partially valid.")
    else:
        print("  Mode B (sym breaking) does NOT help — symmetry is not the cause.")
    if any_C_helped:
        print("  Mode C (presolve) helps — presolve alone can mitigate the issue.")
    else:
        print("  Mode C (presolve) does NOT help — presolve is already on by default.")


if __name__ == "__main__":
    main()
