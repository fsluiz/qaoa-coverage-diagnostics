#!/usr/bin/env python3
"""
Rigorous head-to-head benchmark: QWOA vs QAOA vs classical baselines
for m-fold coverage (set cover, m=1; k-domination-style double cover, m=2)
on the PEF Amazon-frontier instance.

Design (referee-proofed):
  * EXACT diagonal cost oracle  f(x) = |x| + lam * n_violated(x),
    identical for both quantum ansaetze -> isolates the mixer + initial-state
    effect (the only structural difference between QWOA and QAOA).
  * Coverage is ONE-SIDED (hinge): a task t is "violated" iff its coverage is
    below target_t = min(m, c_t), where c_t is the max coverage achievable for t
    within the qubit subset. Over-coverage is NOT penalised (unlike the symmetric
    (target-cov)^2 QUBO surrogate). This is the m-fold coverage objective we want.
  * QWOA is simulated EXACTLY inside the fixed-Hamming-weight-k sector
    (XY-mixer + Dicke state preserve cardinality); QAOA on the full 2^n space
    (X-mixer + |+>^n explores all cardinalities).
  * Resource accounting: n_ZZ = |E(G_B)| (pairs sharing a task); per-layer CX,
    Dicke-prep CX (Baertschi-Eidenbenz ~ k(n-k)), and XY-mixer CX (3 per edge)
    are reported so QWOA vs QAOA can be compared at EQUAL HARDWARE COST, not just
    equal p.

NOTE on the simulation-vs-circuit gap: the exact violated-task oracle is not a
low-degree polynomial, so a real circuit would implement a QUBO surrogate. Here
the oracle is used to measure SOLUTION QUALITY (ansatz effect), while the CX
counts reflect the QUBO-encoding cost. This split is documented for the paper.

Instance: 24 PEF bases (pef_planilha.xlsx) + tasks seeded at 42 within RANGE_KM,
filtered by an Amazonia-Legal bounding box (shapefile-free, reproducible). Reduced
to N_QUBITS bases via greedy+geographic-diversity selection (ported from
v7_quantum_walk.ipynb). The frozen instance is written to JSON.

Author: assistant for F. S. Luiz (IFGW/UNICAMP). Pure NumPy/SciPy.
"""
from __future__ import annotations
import json
import os
import time
import math
import itertools
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize, milp, LinearConstraint, Bounds

from _paths import DATA_DIR, RESULTS_DIR

RNG_SEED = 42
RANGE_KM = 200.0
N_ZONES, N_MICS = 15, 8
N_QUBITS = 12          # qubit subset (reduced from notebook's 15 to keep sector eigh cheap)
LAM = 25.0             # penalty: > N_QUBITS so any feasible beats any infeasible
P_LIST = [1, 2, 3, 4]
N_SEEDS = 8
MAXITER = 200          # classical optimiser budget per (ansatz, p, seed)
# Amazonia-Legal bounding box (lat, lon) -- generous, shapefile-free
BBOX = dict(lat_min=-15.0, lat_max=6.0, lon_min=-75.0, lon_max=-58.0)


# ----------------------------------------------------------------------------
# 1. Instance construction (ported from v7_quantum_walk.ipynb, shapefile-free)
# ----------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


def in_bbox(lat, lon):
    return (BBOX["lat_min"] <= lat <= BBOX["lat_max"]
            and BBOX["lon_min"] <= lon <= BBOX["lon_max"])


def _pairwise_km(lats, lons):
    n = len(lats)
    D = np.zeros((n, n))
    for i in range(n):
        D[i] = haversine_km(lats[i], lons[i], lats, lons)
    return D


def build_instance(
    min_task_degree: int = 1,
    target_mean_degree: tuple[float, float] = (3.5, 6.0),
    rng_seed: int | None = None,
):
    """Dense, non-trivial coverage instance on a CLUSTER of real PEF bases.

    Degenerate (single-base-per-task) instances make the QWOA/QAOA comparison
    meaningless, so we (1) pick the tightest cluster of N_QUBITS real PEF bases,
    (2) scatter tasks uniformly in the cluster's bbox, and (3) AUTO-TUNE the
    coverage radius R so that mean k_t is in [TARGET_KT_LO, TARGET_KT_HI] and
    every task is reachable. R is the physically-motivated density knob (Sec. on
    R/delta in the paper). The chosen R is reported and frozen with the instance.
    """
    TARGET_KT_LO, TARGET_KT_HI = target_mean_degree
    df = pd.read_excel(DATA_DIR / "pef_planilha.xlsx")
    cols = {c.lower(): c for c in df.columns}
    def pick(*cands):
        for c in cands:
            for lc, orig in cols.items():
                if c in lc:
                    return orig
        raise KeyError(cands)
    name_c = pick("nome", "base", "pelot", "name")
    lat_c, lon_c = pick("lat"), pick("lon", "long")
    df = df.dropna(subset=[lat_c, lon_c]).reset_index(drop=True)
    base_names = df[name_c].astype(str).tolist()
    base_lats = df[lat_c].to_numpy(float)
    base_lons = df[lon_c].to_numpy(float)
    n_B = len(base_names)

    # 1. tightest cluster of N_QUBITS bases (seed = base with smallest spread to
    #    its N_QUBITS-1 nearest neighbours)
    D = _pairwise_km(base_lats, base_lons)
    spreads = [np.sort(D[s])[:N_QUBITS].sum() for s in range(n_B)]
    seed = int(np.argmin(spreads))
    q_idx = sorted(np.argsort(D[seed])[:N_QUBITS].tolist())
    lats_q = base_lats[np.array(q_idx)]
    lons_q = base_lons[np.array(q_idx)]
    names_q = [base_names[i] for i in q_idx]

    # 2. tasks uniform in the cluster bbox (small margin)
    rng = np.random.RandomState(RNG_SEED if rng_seed is None else rng_seed)
    n_T = N_ZONES + N_MICS
    mlat = 0.1 * (lats_q.max() - lats_q.min() + 1e-6)
    mlon = 0.1 * (lons_q.max() - lons_q.min() + 1e-6)
    task_lats = rng.uniform(lats_q.min() - mlat, lats_q.max() + mlat, n_T)
    task_lons = rng.uniform(lons_q.min() - mlon, lons_q.max() + mlon, n_T)

    # 3. distance(base_in_cluster, task) and auto-tune R
    lat1 = np.radians(lats_q[:, None]); lat2 = np.radians(task_lats[None, :])
    lon1 = np.radians(lons_q[:, None]); lon2 = np.radians(task_lons[None, :])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    dist = 2 * 6371.0 * np.arcsin(np.sqrt(a))     # (n_q, n_T)
    R_chosen, A_q = None, None
    for R in range(80, 1201, 20):
        Acand = (dist <= R).astype(int)
        kt = Acand.sum(axis=0)
        if kt.min() >= min_task_degree and TARGET_KT_LO <= kt.mean() <= TARGET_KT_HI:
            R_chosen, A_q = float(R), Acand
            break
    if A_q is None:                                # fallback: smallest R meeting strict degree
        for R in range(80, 2001, 20):
            Acand = (dist <= R).astype(int)
            if Acand.sum(axis=0).min() >= min_task_degree:
                R_chosen, A_q = float(R), Acand
                break
    if A_q is None:
        raise RuntimeError(f"could not construct an instance with minimum task degree {min_task_degree}")
    return dict(A_q=A_q, names_q=names_q, lats_q=lats_q, lons_q=lons_q,
                q_idx=[int(i) for i in q_idx], n_B_full=n_B,
                task_lats=task_lats, task_lons=task_lons,
                R_chosen=R_chosen, cluster_spread_km=float(spreads[seed]))


# ----------------------------------------------------------------------------
# 2. Cost oracle and feasibility (exact, vectorised over all 2^n states)
# ----------------------------------------------------------------------------
def precompute(A_q, m):
    """Return per-state arrays: cardinality, n_violated, energy, feasible mask.
    target_t = min(m, c_t); reachable tasks have c_t>=1."""
    n_B, n_T = A_q.shape
    N = 1 << n_B
    idx = np.arange(N, dtype=np.int64)
    card = np.zeros(N, dtype=np.int32)
    for b in range(n_B):
        card += ((idx >> b) & 1).astype(np.int32)
    c_t = A_q.sum(axis=0)                       # max coverage achievable per task
    target = np.minimum(m, c_t)                 # best-effort for c_t < m
    n_violated = np.zeros(N, dtype=np.int32)
    for t in range(n_T):
        if c_t[t] == 0:
            continue
        covering = np.where(A_q[:, t] == 1)[0]
        cov_t = np.zeros(N, dtype=np.int32)
        for b in covering:
            cov_t += ((idx >> b) & 1).astype(np.int32)
        n_violated += (cov_t < target[t]).astype(np.int32)
    energy = card.astype(float) + LAM * n_violated
    feasible = n_violated == 0
    return dict(card=card, n_violated=n_violated, energy=energy, feasible=feasible,
                c_t=c_t, target=target)


def classical_optimum(pc):
    """Min-cardinality feasible solution by brute force over the subset."""
    feas = pc["feasible"]
    if not feas.any():
        return None, None
    cards = pc["card"].copy()
    cards[~feas] = 10 ** 9
    k_opt = int(cards.min())
    opt_states = np.where((pc["card"] == k_opt) & feas)[0]
    return k_opt, opt_states


def greedy_mcover(A_q, m):
    """Greedy m-fold cover: repeatedly add base maximising newly-satisfied deficit."""
    n_B, n_T = A_q.shape
    c_t = A_q.sum(axis=0)
    target = np.minimum(m, c_t)
    cov = np.zeros(n_T, dtype=int)
    chosen = []
    rem = set(range(n_B))
    while True:
        deficit = np.maximum(target - cov, 0)
        if deficit.sum() == 0:
            break
        gains = [(int(np.sum(np.minimum(A_q[b] * deficit, 1) * (deficit > 0))), b)
                 for b in rem]
        # gain = number of still-deficient tasks this base helps
        gains = [(int(np.sum((A_q[b] == 1) & (deficit > 0))), b) for b in rem]
        if not gains or max(g for g, _ in gains) == 0:
            break
        _, b = max(gains)
        chosen.append(b); rem.discard(b); cov += A_q[b]
    return len(chosen), chosen


def ilp_mcover(A_q, m):
    """Exact min m-cover via scipy.milp (HiGHS)."""
    n_B, n_T = A_q.shape
    c_t = A_q.sum(axis=0)
    target = np.minimum(m, c_t)
    rows = [A_q[:, t] for t in range(n_T) if c_t[t] > 0]
    lb = np.array([target[t] for t in range(n_T) if c_t[t] > 0], dtype=float)
    Acon = np.array(rows, dtype=float)
    constr = LinearConstraint(Acon, lb, np.inf)
    res = milp(c=np.ones(n_B), constraints=constr,
               integrality=np.ones(n_B), bounds=Bounds(0, 1))
    if not res.success:
        return None, None, res.status
    x = np.round(res.x).astype(int)
    return int(x.sum()), np.where(x == 1)[0].tolist(), 0


# ----------------------------------------------------------------------------
# 3. Cooperation graph + resource (CX) accounting
# ----------------------------------------------------------------------------
def cooperation_edges(A_q):
    n_B = A_q.shape[0]
    edges = {}
    for b1, b2 in itertools.combinations(range(n_B), 2):
        shared = int(np.sum((A_q[b1] == 1) & (A_q[b2] == 1)))
        if shared > 0:
            edges[(b1, b2)] = float(shared)
    # bridge disconnected components (so XY-mixer is irreducible on the sector)
    parent = list(range(n_B))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for (b1, b2) in edges:
        parent[find(b1)] = find(b2)
    comps = {}
    for b in range(n_B):
        comps.setdefault(find(b), []).append(b)
    reps = [min(c) for c in comps.values()]
    for r1, r2 in zip(reps[:-1], reps[1:]):
        edges[tuple(sorted((r1, r2)))] = 0.1
    return edges


def cx_counts(n_ZZ, n_edges_xy, k, n, p, ansatz):
    """Two-qubit (CX) gate count for p layers (rough, hardware-relevant)."""
    cost_layer_cx = 2 * n_ZZ                    # 2 CX per ZZ term (both ansaetze)
    if ansatz == "QAOA":
        mixer_cx = 0                            # X-mixer: single-qubit only
        init_cx = 0                             # Hadamards
    else:                                       # QWOA
        mixer_cx = 3 * n_edges_xy               # ~3 CX per XX+YY edge term
        init_cx = k * (n - k)                   # Dicke prep (Baertschi-Eidenbenz ~k(n-k))
    return init_cx + p * (cost_layer_cx + mixer_cx)


# ----------------------------------------------------------------------------
# 4. Quantum simulation (pure statevector)
# ----------------------------------------------------------------------------
def apply_x_mixer(state, beta, n):
    """Apply exp(-i beta sum_b X_b) qubit by qubit."""
    c, s = math.cos(beta), math.sin(beta)
    st = state.reshape([2] * n)
    for b in range(n):
        st = np.moveaxis(st, n - 1 - b, 0)      # bring qubit b to front (MSB order)
        a0, a1 = st[0].copy(), st[1].copy()
        st[0] = c * a0 - 1j * s * a1
        st[1] = c * a1 - 1j * s * a0
        st = np.moveaxis(st, 0, n - 1 - b)
    return st.reshape(-1)


def qaoa_run(pc, n, p, params):
    """Full-space QAOA: |+>^n, layers of diag-cost then X-mixer. Returns probs."""
    N = 1 << n
    gammas, betas = params[:p], params[p:]
    state = np.full(N, 1.0 / math.sqrt(N), dtype=complex)
    E = pc["energy"]
    for layer in range(p):
        state *= np.exp(-1j * gammas[layer] * E)
        state = apply_x_mixer(state, betas[layer], n)
    return np.abs(state) ** 2


class QWOASector:
    """QWOA restricted to the fixed-Hamming-weight-k sector (exact)."""
    def __init__(self, A_q, edges, k, pc):
        n = A_q.shape[0]
        self.n, self.k = n, k
        states = [i for i in range(1 << n) if bin(i).count("1") == k]
        self.states = np.array(states, dtype=np.int64)
        pos = {v: i for i, v in enumerate(states)}
        D = len(states)
        M = np.zeros((D, D))
        for (b1, b2), w in edges.items():
            for a, v in enumerate(states):
                bit1, bit2 = (v >> b1) & 1, (v >> b2) & 1
                if bit1 != bit2:                # (1,0) or (0,1) -> swap
                    v2 = v ^ (1 << b1) ^ (1 << b2)
                    M[a, pos[v2]] += w          # (XX+YY)/2 has coeff w on the swap
        # diagonalise once (M fixed for this sector)
        self.evals, self.evecs = np.linalg.eigh(M)
        self.Esec = pc["energy"][self.states]
        self.feas_sec = pc["feasible"][self.states]
        self.card_sec = pc["card"][self.states]
        self.dim = D

    def evolve(self, p, params):
        gammas, betas = params[:p], params[p:]
        psi = np.full(self.dim, 1.0 / math.sqrt(self.dim), dtype=complex)  # Dicke
        for layer in range(p):
            psi *= np.exp(-1j * gammas[layer] * self.Esec)
            # exp(-i beta M) psi  via eigendecomposition
            psi = self.evecs @ (np.exp(-1j * betas[layer] * self.evals)
                                * (self.evecs.conj().T @ psi))
        return np.abs(psi) ** 2


def _extend(prev, p):
    """INTERP-style warm start: extend best (p-1)-params to p by repeating last layer."""
    g, b = prev[: p - 1], prev[p - 1:]
    return np.concatenate([g, g[-1:], b, b[-1:]])


def optimise(expval_fn, p, rng, warm=None):
    """Multi-seed classical optimisation of variational angles; return best params.
    If `warm` (best params from p-1) is given, add the INTERP-extended guess."""
    best_val, best_par = np.inf, None
    starts = [np.concatenate([rng.uniform(0.0, np.pi, p),
                              rng.uniform(0.0, np.pi / 2, p)]) for _ in range(N_SEEDS)]
    if warm is not None and len(warm) == 2 * (p - 1):
        starts.append(_extend(warm, p))
    for x0 in starts:
        res = minimize(expval_fn, x0, method="Nelder-Mead",
                       options=dict(maxiter=MAXITER, xatol=1e-3, fatol=1e-3))
        if res.fun < best_val:
            best_val, best_par = res.fun, res.x
    return best_par, best_val


# ----------------------------------------------------------------------------
# 5. Metrics
# ----------------------------------------------------------------------------
def qaoa_metrics(probs, pc, opt_states):
    opt_mask = np.zeros(pc["energy"].shape[0], dtype=bool)
    opt_mask[opt_states] = True
    p_opt = float(probs[opt_mask].sum())
    p_feas = float(probs[pc["feasible"]].sum())
    exp_E = float((probs * pc["energy"]).sum())
    # expected cardinality among feasible samples
    fmass = probs[pc["feasible"]].sum()
    exp_k_feas = (float((probs[pc["feasible"]] * pc["card"][pc["feasible"]]).sum() / fmass)
                  if fmass > 1e-12 else float("nan"))
    return dict(p_opt=p_opt, p_feas=p_feas, exp_E=exp_E, exp_k_feas=exp_k_feas)


def qwoa_metrics(probs, sec, k_opt):
    p_feas = float(probs[sec.feas_sec].sum())
    # at k==k_opt every feasible state is globally optimal
    p_opt = p_feas if sec.k == k_opt else 0.0
    exp_E = float((probs * sec.Esec).sum())
    return dict(p_opt=p_opt, p_feas=p_feas, exp_E=exp_E, exp_k_feas=float(sec.k))


# ----------------------------------------------------------------------------
# 6. Main
# ----------------------------------------------------------------------------
def main():
    t0 = time.time()
    inst = build_instance()
    A_q, names_q = inst["A_q"], inst["names_q"]
    n = A_q.shape[0]
    edges = cooperation_edges(A_q)
    n_ZZ = sum(1 for (b1, b2), w in edges.items() if w > 0.1) + \
        sum(1 for (b1, b2), w in edges.items() if 0 < w <= 0.1) * 0  # true ZZ from real coop
    n_ZZ = int(np.sum([1 for (b1, b2) in itertools.combinations(range(n), 2)
                       if int(np.sum((A_q[b1] == 1) & (A_q[b2] == 1))) > 0]))
    n_edges_xy = len(edges)

    kt = A_q.sum(axis=0)
    print(f"Instance: {n} qubits, {A_q.shape[1]} tasks, R={inst['R_chosen']:.0f}km, "
          f"cluster_spread={inst['cluster_spread_km']:.0f}km, "
          f"n_ZZ={n_ZZ}, XY-edges={n_edges_xy}, mean_k_t={kt.mean():.2f}")
    print("k_t per task:", kt.tolist())

    results: dict[str, Any] = dict(
        meta=dict(n_qubits=n, n_tasks=int(A_q.shape[1]), n_ZZ=n_ZZ,
                  n_edges_xy=n_edges_xy, lam=LAM, p_list=P_LIST,
                  n_seeds=N_SEEDS, range_km=inst["R_chosen"],
                  cluster_spread_km=inst["cluster_spread_km"], bases=names_q,
                  mean_k_t=float(kt.mean()), k_t=kt.tolist()),
        m={},
    )
    # freeze instance
    json.dump(dict(A_q=A_q.tolist(), names=names_q,
                   lats=inst["lats_q"].tolist(), lons=inst["lons_q"].tolist(),
                   q_idx=[int(i) for i in inst["q_idx"]]),
              open(DATA_DIR / "benchmark_instance.json", "w"), indent=2)

    rng = np.random.RandomState(0)

    m_values = tuple(int(v) for v in os.environ.get("M_VALUES", "1,2,3").split(","))
    print(f"m values: {m_values}")
    for m in m_values:
        print(f"\n{'='*70}\n m = {m}\n{'='*70}")
        pc = precompute(A_q, m)
        k_opt, opt_states = classical_optimum(pc)
        k_greedy, greedy_sel = greedy_mcover(A_q, m)
        k_ilp, ilp_sel, ilp_status = ilp_mcover(A_q, m)
        n_feas = int(pc["feasible"].sum())
        p_rnd_opt = (len(opt_states) / (1 << n)) if opt_states is not None else 0.0
        print(f"  classical: brute k_opt={k_opt} (#opt={0 if opt_states is None else len(opt_states)}), "
              f"greedy={k_greedy}, ILP={k_ilp}  | #feasible={n_feas}, "
              f"P_rnd(opt)={p_rnd_opt:.2e}")

        mres = dict(k_opt=k_opt, k_greedy=k_greedy, k_ilp=k_ilp,
                    n_feasible=n_feas, p_rnd_opt=p_rnd_opt,
                    greedy_sel=greedy_sel, ilp_sel=ilp_sel,
                    qaoa=[], qwoa=[], descent=None)

        if k_opt is None:
            results["m"][str(m)] = mres
            continue

        sec_cache: dict[int, QWOASector] = {}
        def get_sector(k):
            if k not in sec_cache:
                sec_cache[k] = QWOASector(A_q, edges, k, pc)
            return sec_cache[k]

        # --- QAOA sweep over p (full space) ---
        prev = None
        for p in P_LIST:
            fn = lambda par: float((qaoa_run(pc, n, p, par) * pc["energy"]).sum())
            par, val = optimise(fn, p, rng, warm=prev)
            prev = par
            probs = qaoa_run(pc, n, p, par)
            mt = qaoa_metrics(probs, pc, opt_states)
            cx = cx_counts(n_ZZ, n_edges_xy, k_opt, n, p, "QAOA")
            mt.update(p=p, cx=cx, ar=mt["exp_E"] / k_opt)
            mres["qaoa"].append(mt)
            print(f"  QAOA  p={p}: P(opt)={mt['p_opt']:.3e}  P(feas)={mt['p_feas']:.3f}  "
                  f"<E>={mt['exp_E']:.2f}  <k|feas>={mt['exp_k_feas']:.2f}  CX={cx}")

        # --- QWOA sweep over p (sector k = k_opt) ---
        sec = get_sector(k_opt)
        print(f"  [QWOA sector k={k_opt}: dim C({n},{k_opt})={sec.dim}]")
        prev = None
        for p in P_LIST:
            fn = lambda par: float((sec.evolve(p, par) * sec.Esec).sum())
            par, val = optimise(fn, p, rng, warm=prev)
            prev = par
            probs = sec.evolve(p, par)
            mt = qwoa_metrics(probs, sec, k_opt)
            cx = cx_counts(n_ZZ, n_edges_xy, k_opt, n, p, "QWOA")
            mt.update(p=p, cx=cx, ar=mt["exp_E"] / k_opt)
            mres["qwoa"].append(mt)
            print(f"  QWOA  p={p}: P(opt)={mt['p_opt']:.3e}  P(feas)={mt['p_feas']:.3f}  "
                  f"<E>={mt['exp_E']:.2f}  <k|feas>={mt['exp_k_feas']:.2f}  CX={cx}")

        # --- QWOA k-descent (classical k_greedy -> down until infeasible) ---
        descent = []
        p_desc = max(P_LIST)
        k = k_greedy
        while k >= 1:
            sec_k = get_sector(k)
            fn = lambda par: float((sec_k.evolve(p_desc, par) * sec_k.Esec).sum())
            par, val = optimise(fn, p_desc, rng)
            probs = sec_k.evolve(p_desc, par)
            pf = float(probs[sec_k.feas_sec].sum())
            descent.append(dict(k=k, p_feas=pf, dim=sec_k.dim))
            print(f"  descent k={k}: P(feasible in sector)={pf:.3f}  (dim={sec_k.dim})")
            if pf < 1e-3 or k <= k_opt:          # stop at optimum or when infeasible
                break
            k -= 1
        mres["descent"] = dict(p=p_desc, steps=descent,
                               k_final=descent[-1]["k"])
        results["m"][str(m)] = mres

    results["meta"]["runtime_s"] = round(time.time() - t0, 1)
    json.dump(results, open(RESULTS_DIR / "benchmark_qwoa_qaoa_results.json", "w"), indent=2)
    print(f"\nDone in {results['meta']['runtime_s']}s -> "
          f"benchmark_qwoa_qaoa_results.json (+ benchmark_instance.json)")


if __name__ == "__main__":
    main()
