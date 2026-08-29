#!/usr/bin/env python3
"""
Freeze the two canonical instances for the IBM Kingston re-run, so the m-ladder,
all tables, and the hardware circuits use the IDENTICAL coverage matrix.

  frozen_rung1.json     -- PEF-24 frontier, R=200 km, m=1, k_opt=10 (anchored seed 0;
                           reproducible; n_ZZ=17).  Rung-1 hardware demo + difficulty axis.
  frozen_overshoot.json -- n=14, m=3 triple-cover (cluster_seed=22, task_seed=3):
                           greedy=12, k_opt=9.  Tests greedy+QWOA-descent vs QAOA.

CIRCUIT COST = quadratic QUBO  H = sum_b x_b + lam * sum_t (m - sum_b A_tb x_b)^2,
mapped to Ising (h_b Z_b + J_bb' Z_b Z_b') so the cost layer is RZZ+RZ gates.
(The Path-B *simulation* used a one-sided exact oracle that is NOT a 2-local circuit;
on hardware we must use this quadratic form.)
Pure NumPy/SciPy.  Author: assistant for F. S. Luiz (IFGW/UNICAMP).
"""
import json
import numpy as np
import itertools

from _paths import DATA_DIR


def quadratic_qubo_ising(A, m, lam):
    """H = sum_b x_b + lam sum_t (t_t - sum_b A_tb x_b)^2 with per-task target
    t_t = min(m, deg_t)  ->  Ising h, J (x=(1-Z)/2).  Using min(m, deg_t) is required:
    a task reachable by < m bases can never be m-covered, so its target is its degree.
    Returns (h, J, lin, quad, n_ZZ)."""
    n_t, n_b = A.shape
    target = np.minimum(m, A.sum(axis=1))                  # (n_t,)  t_t = min(m, deg_t)
    # QUBO: linear q_b, quadratic Q_bb' (b<b')
    lin = np.ones(n_b, float)
    for b in range(n_b):
        lin[b] += lam * np.sum(A[:, b] * (1 - 2 * target))  # -2 t_t A + A (since x^2=x)
    quad = {}
    for b1, b2 in itertools.combinations(range(n_b), 2):
        c = 2.0 * lam * float(np.sum(A[:, b1] * A[:, b2]))   # 2 lam sum_t A_tb A_tb'
        if c != 0.0:
            quad[(b1, b2)] = c
    n_ZZ = len(quad)
    # Ising: x_b = (1 - Z_b)/2
    h = np.zeros(n_b)
    J = {}
    for b in range(n_b):
        h[b] += -lin[b] / 2.0
    for (b1, b2), c in quad.items():
        J[(b1, b2)] = c / 4.0
        h[b1] += -c / 4.0
        h[b2] += -c / 4.0
    return h, J, lin, quad, n_ZZ


def cooperation_edges(A):
    """XY-mixer edges: base pairs that co-cover at least one task."""
    n_b = A.shape[1]
    return [[b1, b2] for b1, b2 in itertools.combinations(range(n_b), 2)
            if int(np.sum((A[:, b1] > 0) & (A[:, b2] > 0))) > 0]


def greedy_k(A):
    Aw = A.copy().astype(float); c = 0
    while Aw.max() > 0:
        Aw[Aw[:, int(Aw.sum(0).argmax())] > 0, :] = 0.0; c += 1
    return c


def freeze_rung1():
    from run_m_ladder import build_instance, coverage
    blat, blon, tlat, tlon, seed, n_b = build_instance()      # anchored seed-0
    R, m, lam = 200.0, 1, 10.0
    A = coverage(blat, blon, tlat, tlon, R).astype(int)        # (23, 24)
    h, J, lin, quad, n_zz = quadratic_qubo_ising(A, m, lam)
    out = dict(
        name="PEF-24 frontier (Rung 1)", n_b=int(n_b), n_t=int(A.shape[0]),
        R_km=R, m=m, lam=lam, task_seed=int(seed), generator="anchored (run_m_ladder)",
        k_opt=10, k_greedy=int(greedy_k(A)),
        links=int(A.sum()), n_ZZ=int(n_zz),
        base_lats=blat.tolist(), base_lons=blon.tolist(),
        task_lats=tlat.tolist(), task_lons=tlon.tolist(),
        A=A.tolist(),
        h=h.tolist(), J={f"{a},{b}": v for (a, b), v in J.items()},
        xy_edges=cooperation_edges(A),
        # transferred p=1 QAOA angles (parameter concentration); refine in submit if desired
        qaoa_angles={"1": {"gamma": [-0.04], "beta": [0.35]}},
        note="m=1: greedy=k_opt=10 (no overshoot); descent is trivial here.",
    )
    json.dump(out, open(DATA_DIR / "frozen_rung1.json", "w"), indent=1)
    print(f"frozen_rung1: n_b={n_b} n_t={A.shape[0]} links={out['links']} "
          f"n_ZZ={n_zz} k_opt=10 k_greedy={out['k_greedy']} xy_edges={len(out['xy_edges'])}")


def freeze_overshoot():
    src = json.load(open(DATA_DIR / "overshoot_instance.json"))
    A = np.asarray(src["A_q"], int)                            # (14, 23)? -> check orientation
    # overshoot_instance A_q is (n_b, n_t) per build; coverage matrix wants (n_t, n_b)
    if A.shape[0] == src["n_qubits"]:
        A = A.T                                                 # -> (n_t, n_b)
    m, lam = int(src["m"]), 25.0
    h, J, lin, quad, n_zz = quadratic_qubo_ising(A, m, lam)
    out = dict(
        name="overshoot n=14 m=3 (descent demo)", n_b=int(A.shape[1]), n_t=int(A.shape[0]),
        R_km=src["R"], m=m, lam=lam,
        cluster_seed=src["cluster_seed"], task_seed=src["task_seed"],
        k_opt=int(src["k_opt"]), k_greedy=int(src["k_greedy"]), n_opt=int(src["n_opt"]),
        links=int(A.sum()), n_ZZ=int(n_zz),
        lats=src["lats_q"], lons=src["lons_q"], A=A.tolist(),
        h=h.tolist(), J={f"{a},{b}": v for (a, b), v in J.items()},
        xy_edges=cooperation_edges(A),
        descent_sectors=[12, 11, 10, 9],
        note="greedy=12 -> k_opt=9 (overshoot 3). QWOA-descent runs k=12,11,10,9.",
    )
    json.dump(out, open(DATA_DIR / "frozen_overshoot.json", "w"), indent=1)
    print(f"frozen_overshoot: n_b={A.shape[1]} n_t={A.shape[0]} links={out['links']} "
          f"n_ZZ={n_zz} k_opt={out['k_opt']} greedy={out['k_greedy']} xy_edges={len(out['xy_edges'])}")


def freeze_arco():
    """Rung-2 Arco do Fogo (79 municipalities, R=100, set cover m=1), from the stored
    reproducible coords in arco_fogo_problem.json.  n_ZZ=9 (sparsification)."""
    from run_m_ladder import coverage, ilp_min_cover
    src = json.load(open(DATA_DIR / "arco_fogo_problem.json"))
    bl, bo = np.array(src["base_lats"]), np.array(src["base_lons"])
    tl, to = np.array(src["task_lats"]), np.array(src["task_lons"])
    R, m, lam = 100.0, 1, 10.0
    A = coverage(bl, bo, tl, to, R).astype(int)            # (n_t, n_b)
    reach = A.sum(1) >= 1                                    # drop unreachable tasks
    A = A[reach]
    h, J, lin, quad, n_zz = quadratic_qubo_ising(A, m, lam)
    k_opt, _ = ilp_min_cover(A, 1)
    out = dict(
        name="Arco do Fogo (Rung 2)", n_b=int(A.shape[1]), n_t=int(A.shape[0]),
        R_km=R, m=m, lam=lam, generator="arco_fogo_problem.json coords",
        k_opt=int(k_opt), k_greedy=int(greedy_k(A)),
        links=int(A.sum()), n_ZZ=int(n_zz),
        base_lats=bl.tolist(), base_lons=bo.tolist(),
        task_lats=tl[reach].tolist(), task_lons=to[reach].tolist(),
        A=A.tolist(), h=h.tolist(), J={f"{a},{b}": v for (a, b), v in J.items()},
        xy_edges=cooperation_edges(A),
        qaoa_angles={"1": {"gamma": [src.get("gamma_opt_transfer", -0.04)],
                           "beta": [src.get("beta_opt_transfer", 0.35)]}},
    )
    json.dump(out, open(DATA_DIR / "frozen_arco.json", "w"), indent=1)
    print(f"frozen_arco: n_b={A.shape[1]} n_t={A.shape[0]} links={out['links']} "
          f"n_ZZ={n_zz} k_opt={k_opt} k_greedy={out['k_greedy']}")


if __name__ == "__main__":
    freeze_rung1()
    freeze_overshoot()
    freeze_arco()
    print("-> frozen_rung1.json, frozen_overshoot.json, frozen_arco.json written")
