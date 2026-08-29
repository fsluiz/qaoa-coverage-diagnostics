"""
m-ladder ILP for the Drones paper (Rung 1 + Rung 1' base-protection).

Reproduces the canonical Scale-1 PEF-24 instance (R=200 km, 23 tasks, greedy k=10,
identical seed-search to compute_final.py), then computes the m-fold difficulty ladder
for three formulations:

  (T)  task set cover            : every TASK covered by >= m active bases
  (T+P) base-protected           : (T) AND every BASE dominated by >= m active bases
                                   (closed neighbourhood: an active base protects itself)
  (T+TP) total base-protection   : (T) AND every BASE has >= m active OTHER bases in range
                                   (total domination: an active base still needs external cover)

For each: exact ILP k_opt(m) (scipy.milp) and greedy(m) -> overshoot = greedy - k_opt.
Plus a single-base-failure RESILIENCE demo showing m=1 covers open holes, m=2 do not.

Output: m_ladder_results.json
"""
import json
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds

from _paths import DATA_DIR, RESULTS_DIR

EARTH_R_KM = 6371.0
RANGE_KM = 200.0
N_T = 23
K_OPT_TARGET = 10
M_VALUES = [1, 2, 3]


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2.0 * EARTH_R_KM * np.arcsin(np.sqrt(a))


def build_instance():
    """Identical task generation + seed search as compute_final.py.
    Returns base + task coordinates (positions fixed; coverage recomputed per R)."""
    df = pd.read_excel(DATA_DIR / "pef_planilha.xlsx")
    base_lats = df["latitude"].values.astype(np.float64)
    base_lons = df["longitude"].values.astype(np.float64)
    n_b = len(base_lats)
    MAX_OFFSET_KM = RANGE_KM * 0.7

    for seed in range(200):
        rng = np.random.default_rng(seed)
        anchor = rng.integers(0, n_b, size=N_T)
        off = rng.uniform(0.0, MAX_OFFSET_KM, size=N_T)
        ang = rng.uniform(0.0, 2 * np.pi, size=N_T)
        tlat, tlon = np.zeros(N_T), np.zeros(N_T)
        for t in range(N_T):
            lb, lo = base_lats[anchor[t]], base_lons[anchor[t]]
            tlat[t] = lb + off[t] * np.cos(ang[t]) / 111.0
            tlon[t] = lo + off[t] * np.sin(ang[t]) / (111.0 * np.cos(np.radians(lb)))
        A = coverage(base_lats, base_lons, tlat, tlon, RANGE_KM)
        if A.sum(axis=1).min() < 1:
            continue
        Aw, cnt = A.copy(), 0
        while Aw.max() > 0:
            Aw[Aw[:, int(Aw.sum(axis=0).argmax())] > 0, :] = 0.0
            cnt += 1
        if cnt == K_OPT_TARGET:
            return base_lats, base_lons, tlat, tlon, seed, n_b
    raise RuntimeError("no seed gives k=10")


def coverage(blat, blon, tlat, tlon, R):
    A = np.zeros((len(tlat), len(blat)))
    for t in range(len(tlat)):
        A[t] = (haversine(tlat[t], tlon[t], blat, blon) <= R).astype(float)
    return A


def base_graphs(blat, blon, R):
    n = len(blat)
    Dbb = np.zeros((n, n))
    for i in range(n):
        Dbb[i] = haversine(blat[i], blon[i], blat, blon)
    N_closed = (Dbb <= R).astype(float)
    N_open = N_closed.copy()
    np.fill_diagonal(N_open, 0.0)
    return N_closed, N_open


def ilp_min_cover(rows, m):
    """min sum(x) s.t. rows @ x >= m, x in {0,1}. Returns (k_opt, sel) or (None,None)."""
    n = rows.shape[1]
    # feasibility: every row must have >= m ones
    if rows.shape[0] and rows.sum(axis=1).min() < m:
        return None, None
    c = np.ones(n)
    con = LinearConstraint(rows, lb=m, ub=np.inf)
    res = milp(c=c, constraints=con, integrality=np.ones(n),
               bounds=Bounds(0, 1))
    if not res.success:
        return None, None
    sel = [i for i, v in enumerate(np.round(res.x)) if v > 0.5]
    return len(sel), sel


def greedy_cover(rows, m):
    """Greedy m-fold cover over generic demand rows. Returns count or None if infeasible."""
    if rows.shape[0] and rows.sum(axis=1).min() < m:
        return None
    n = rows.shape[1]
    cover = np.zeros(rows.shape[0])           # current coverage per row
    chosen = np.zeros(n, dtype=bool)
    picks = 0
    while np.any(cover < m):
        need = (cover < m).astype(float)      # rows still needing coverage
        # gain of column j = number of still-needy rows it covers
        gains = (rows * need[:, None]).sum(axis=0)
        gains[chosen] = -1
        j = int(gains.argmax())
        if gains[j] <= 0:
            return None
        chosen[j] = True
        cover += rows[:, j]
        picks += 1
    return picks


def resilience(A, sel, m):
    """For an m-cover `sel`, worst-case #tasks uncovered after a single base failure."""
    sel = list(sel)
    worst = 0
    breakers = []
    for b in sel:
        remaining = [s for s in sel if s != b]
        cov = A[:, remaining].sum(axis=1) if remaining else np.zeros(A.shape[0])
        uncovered = int((cov < 1).sum())
        if uncovered > worst:
            worst = uncovered
        if uncovered > 0:
            breakers.append(int(b))
    return worst, breakers


def ladder_at_R(blat, blon, tlat, tlon, R):
    """k_opt(m)/greedy(m) for the three formulations at radius R."""
    A = coverage(blat, blon, tlat, tlon, R)
    N_closed, N_open = base_graphs(blat, blon, R)
    forms = {
        "T_taskcover": A,
        "TP_base_protected": np.vstack([A, N_closed]),
        "TTP_total_protection": np.vstack([A, N_open]),
    }
    res = {"R_km": R,
           "min_task_deg": int(A.sum(axis=1).min()),
           "min_base_closed_deg": int(N_closed.sum(axis=1).min()),
           "min_base_open_deg": int(N_open.sum(axis=1).min()),
           "n_zz_taskcover": int((A.T @ A > 0).sum() - (np.diag(A.T @ A) > 0).sum()) // 2,
           "ladder": {}}
    for name, rows in forms.items():
        res["ladder"][name] = {}
        for m in M_VALUES:
            k_opt, sel = ilp_min_cover(rows, m)
            g = greedy_cover(rows, m)
            res["ladder"][name][f"m={m}"] = {
                "k_opt": k_opt, "greedy": g,
                "overshoot": (g - k_opt) if (g is not None and k_opt is not None) else None,
                "feasible": k_opt is not None,
            }
    return res, A


def main():
    blat, blon, tlat, tlon, seed, n_b = build_instance()
    R_SWEEP = [200, 250, 300, 350, 400, 450, 500, 550, 600]
    out = {
        "meta": {"instance": "PEF-24 canonical Scale-1", "n_b": n_b, "n_t": N_T,
                 "task_seed": int(seed), "base_range_canonical_km": RANGE_KM,
                 "note": "task positions fixed (seed 0); coverage recomputed per R"},
        "sweep": [], "thresholds": {}, "resilience": {},
    }

    feas = {f"T_taskcover_m={m}": None for m in [2, 3]}
    for R in R_SWEEP:
        res, _ = ladder_at_R(blat, blon, tlat, tlon, float(R))
        out["sweep"].append(res)
        for m in [2, 3]:
            key = f"T_taskcover_m={m}"
            if feas[key] is None and res["ladder"]["T_taskcover"][f"m={m}"]["feasible"]:
                feas[key] = R
    out["thresholds"] = feas

    # resilience demo at canonical R=200 (m=1, the fragile baseline)
    A200 = coverage(blat, blon, tlat, tlon, RANGE_KM)
    k1, sel1 = ilp_min_cover(A200, 1)
    worst, breakers = resilience(A200, sel1, 1)
    out["resilience"]["m=1_at_R200"] = {
        "k_opt": k1, "worst_uncovered_after_1_failure": worst,
        "n_single_failures_that_open_a_hole": len(breakers),
    }
    # resilience at the smallest R where m=2 task-cover is feasible
    if feas["T_taskcover_m=2"] is not None:
        R2 = float(feas["T_taskcover_m=2"])
        A2 = coverage(blat, blon, tlat, tlon, R2)
        k2, sel2 = ilp_min_cover(A2, 2)
        w2, br2 = resilience(A2, sel2, 2)
        out["resilience"]["m=2_at_threshold"] = {
            "R_km": R2, "k_opt": k2,
            "worst_uncovered_after_1_failure": w2,
            "n_single_failures_that_open_a_hole": len(br2),
        }

    with open(RESULTS_DIR / "m_ladder_results.json", "w") as f:
        json.dump(out, f, indent=1)

    # console summary
    print(f"Instance: PEF-{n_b}, {N_T} tasks, canonical R={RANGE_KM:.0f} km, seed={seed}\n")
    hdr = f"{'R':>5} {'minTdeg':>7} {'minBopen':>8} {'nZZ':>5} | " + \
          "  ".join(f"T m{m}" for m in M_VALUES) + " | " + \
          "  ".join(f"TP m{m}" for m in M_VALUES) + " | " + \
          "  ".join(f"TTP m{m}" for m in M_VALUES)
    print(hdr)
    for r in out["sweep"]:
        def cell(form, m):
            e = r["ladder"][form][f"m={m}"]
            return f"{e['k_opt']}/{e['greedy']}" if e["feasible"] else "—"
        row = f"{r['R_km']:>5.0f} {r['min_task_deg']:>7} {r['min_base_open_deg']:>8} {r['n_zz_taskcover']:>5} | "
        row += "  ".join(f"{cell('T_taskcover',m):>4}" for m in M_VALUES) + " | "
        row += "  ".join(f"{cell('TP_base_protected',m):>5}" for m in M_VALUES) + " | "
        row += "  ".join(f"{cell('TTP_total_protection',m):>5}" for m in M_VALUES)
        print(row)
    print("\nFeasibility thresholds (min R, km):", out["thresholds"])
    print("Resilience:", json.dumps(out["resilience"], indent=1))
    print("\n(cells are ILP/greedy; '—' = infeasible) -> m_ladder_results.json written")


if __name__ == "__main__":
    main()
