#!/usr/bin/env python3
"""
IBM Kingston re-run for the Drones paper.  Builds the frozen-instance circuits,
optimizes angles noiselessly, transpiles to the backend, reports the two-qubit-gate
cost of every circuit, and (with --submit) runs them with DD + Pauli twirling.

  RUN SET
   Rung-1 (frozen_rung1.json, 24q):  QAOA p=1,2,3      -- core result + noise-depth.
   Overshoot (frozen_overshoot.json, 14q):
       QAOA p=1,2,3 ; QWOA-direct k=9 p=1,2 ;
       QWOA-descent k=12,11,10,9 p=1  -- tests greedy+descent P(opt)-per-CX.

  USAGE
    python submit_kingston.py                 # DRY RUN: build, optimize, transpile, report CX
    python submit_kingston.py --submit        # submit to ibm_kingston (needs your account)
    python submit_kingston.py --shots 8192 --backend ibm_kingston

  Cost layer uses the QUADRATIC QUBO Ising (h,J) frozen with the instance (RZZ+RZ).
  QWOA initial state = exact Baertschi-Eidenbenz Dicke (dicke.py, complement-optimized).
  If the QWOA circuits are noise-dominated on IBM (expected at 14q), the same instances
  port to QuEra Aquila, where the Dicke/fixed-weight manifold is prepared analogically.
  Author: assistant for F. S. Luiz (IFGW/UNICAMP).
"""
import json, argparse, itertools
import numpy as np
from qiskit import QuantumCircuit, transpile
from scipy.optimize import minimize
from dicke import dicke_circuit

from _paths import DATA_DIR, RESULTS_DIR


# ----------------------------------------------------------------------------
def load_instance(path):
    d = json.load(open(path))
    d["A"] = np.asarray(d["A"], int)                       # (n_t, n_b)
    d["h"] = np.asarray(d["h"], float)
    d["J"] = {tuple(int(i) for i in k.split(",")): v for k, v in d["J"].items()}
    d["xy_edges"] = [tuple(e) for e in d["xy_edges"]]
    return d


def diag_energies(inst):
    """Per-bitstring cardinality, feasibility (cov>=target, target_t=min(m,deg_t)),
    and the optimum mask (min-cardinality feasible m-cover)."""
    A, m = inst["A"], inst["m"]
    n = A.shape[1]
    target = np.minimum(m, A.sum(axis=1))                               # (n_t,)
    bits = ((np.arange(1 << n)[:, None] >> np.arange(n)[None, :]) & 1)   # (2^n, n) LSB=qubit0
    card = bits.sum(1)
    cov = bits @ A.T                                                     # (2^n, n_t)
    feasible = (cov >= target[None, :]).all(1)
    energy = card + inst["lam"] * ((target[None, :] - cov) ** 2).sum(1)
    is_opt = feasible & (card == inst["k_opt"])
    return energy, feasible, is_opt, card


# ----------------------------------------------------------------------------
def qaoa_circuit(inst, p, params, measure=True):
    n = inst["A"].shape[1]
    g, b = params[:p], params[p:]
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for L in range(p):
        for q in range(n):
            if inst["h"][q] != 0.0:
                qc.rz(2 * g[L] * inst["h"][q], q)
        for (i, j), Jij in inst["J"].items():
            qc.rzz(2 * g[L] * Jij, i, j)
        for q in range(n):
            qc.rx(2 * b[L], q)
    if measure:
        qc.measure_all()
    return qc


def qwoa_circuit(inst, k, p, params, measure=True):
    n = inst["A"].shape[1]
    g, b = params[:p], params[p:]
    qc = dicke_circuit(n, k)                          # exact Dicke initial state |D^n_k>
    for L in range(p):
        for q in range(n):
            if inst["h"][q] != 0.0:
                qc.rz(2 * g[L] * inst["h"][q], q)
        for (i, j), Jij in inst["J"].items():
            qc.rzz(2 * g[L] * Jij, i, j)
        for (i, j) in inst["xy_edges"]:              # XY mixer: exp(-i b (XX+YY)) per edge
            qc.rxx(2 * b[L], i, j)
            qc.ryy(2 * b[L], i, j)
    if measure:
        qc.measure_all()
    return qc


# ----------------------------------------------------------------------------
def arco_subproblem_angles(inst, n_sub=20, p=1, n_starts=8, seed=0):
    """(turing fix) Optimize QAOA angles on the n_sub HIGHEST-DEGREE bases (2^20
    statevector, tractable), then transfer to the full instance with a gamma rescale
    by the J energy-scale ratio.  Returns (params_transfer, p_opt_subproblem)."""
    from qiskit.quantum_info import Statevector
    from freeze_instances import quadratic_qubo_ising
    from run_m_ladder import ilp_min_cover
    A_full, m, lam = inst["A"], inst["m"], inst["lam"]
    top = np.argsort(A_full.sum(0))[-n_sub:]                  # densest bases
    A_sub = A_full[:, top]
    A_sub = A_sub[A_sub.any(1)]                                # reachable tasks only
    h_s, J_s, _, _, _ = quadratic_qubo_ising(A_sub, m, lam)
    k_sub, _ = ilp_min_cover(A_sub, m)
    n = n_sub
    target = np.minimum(m, A_sub.sum(1))
    bits = ((np.arange(1 << n)[:, None] >> np.arange(n)[None, :]) & 1)
    is_opt = (bits @ A_sub.T >= target[None, :]).all(1) & (bits.sum(1) == int(k_sub))

    def neg(params):
        g, b = params[:p], params[p:]
        qc = QuantumCircuit(n); qc.h(range(n))
        for L in range(p):
            for q in range(n):
                if h_s[q] != 0.0:
                    qc.rz(2 * g[L] * h_s[q], q)
            for (i, j), Jij in J_s.items():
                qc.rzz(2 * g[L] * Jij, i, j)
            for q in range(n):
                qc.rx(2 * b[L], q)
        return -float(Statevector(qc).probabilities()[is_opt].sum())

    rng = np.random.default_rng(seed); best = (None, np.inf)
    bnds = [(-np.pi, np.pi)] * p + [(0.0, np.pi)] * p
    for _ in range(n_starts):
        x0 = np.concatenate([rng.uniform(-np.pi, np.pi, p), rng.uniform(0, np.pi / 2, p)])
        r = minimize(neg, x0, method="L-BFGS-B", bounds=bnds,
                     options=dict(maxiter=200, ftol=1e-10))
        if r.fun < best[1]:
            best = (r.x, r.fun)
    params = best[0].copy()
    Jf = [abs(v) for v in inst["J"].values()]; Js = [abs(v) for v in J_s.values()]
    params[:p] *= (max(Jf) if Jf else 1.0) / (max(Js) if Js else 1.0)   # rescale gamma only
    return params, -best[1]


def _noiseless_popt_full(inst, p, params, chunk=1 << 20):
    """(turing fix) Exact noiseless P(opt) for n<=26 via chunked statevector
    classification (avoids materializing 2^n x n at once)."""
    from qiskit.quantum_info import Statevector
    A, n = inst["A"], inst["A"].shape[1]
    kopt, target = inst["k_opt"], np.minimum(inst["m"], inst["A"].sum(1))
    probs = Statevector(qaoa_circuit(inst, p, params, False)).probabilities()
    tot = 0.0
    for s in range(0, 1 << n, chunk):
        idx = np.arange(s, min(s + chunk, 1 << n))
        bits = ((idx[:, None] >> np.arange(n)[None, :]) & 1)
        msk = bits.sum(1) == kopt
        if msk.any():
            feas = (bits[msk] @ A.T >= target[None, :]).all(1)
            tot += probs[idx[msk][feas]].sum()
    return float(tot)


def _mps_noiseless_popt(inst, p, params, shots=40000):
    """(turing fix) MPS-sampled noiseless P(opt) for large n (Arco 79q): the sparse
    QAOA circuit (n_ZZ=9) keeps the bond dimension small."""
    from qiskit_aer import AerSimulator
    counts = AerSimulator(method="matrix_product_state").run(
        qaoa_circuit(inst, p, params, True), shots=shots).result().get_counts()
    A, kopt = inst["A"], inst["k_opt"]
    target = np.minimum(inst["m"], inst["A"].sum(1))
    opt = sum(c for bs, c in counts.items()
              if (lambda x: x.sum() == kopt and (A @ x >= target).all())
              (np.array([int(ch) for ch in bs.replace(" ", "")[::-1]], int)))
    return opt / shots


def optimise_angles(inst, kind, p, k=None, n_starts=8, seed=0):
    """Noiseless angle optimization via exact statevector + diagonal cost.
    Returns (best_params, p_opt_noiseless). For n>20: Rung-1 (n<=26) transfers angles
    + exact noiseless P(opt); Arco (n>26) optimizes a 20-base subproblem + MPS P(opt)."""
    from qiskit.quantum_info import Statevector
    n = inst["A"].shape[1]
    if n > 20:
        if kind != "qaoa":
            raise ValueError("QWOA not run at large n")
        if n <= 26:                                   # Rung-1 (24q): transfer + exact P(opt)
            base = inst.get("qaoa_angles", {}).get("1", {"gamma": [-0.04], "beta": [0.35]})
            g0, b0 = base["gamma"][0], base["beta"][0]
            params = np.array([g0] * p + [b0] * p)
            return params, _noiseless_popt_full(inst, p, params)
        params, _ = arco_subproblem_angles(inst, p=p, seed=seed)   # Arco (79q)
        return params, _mps_noiseless_popt(inst, p, params)
    energy, feasible, is_opt, card = diag_energies(inst)
    # target metric: P(opt) for QAOA / QWOA-direct at k_opt; P(feasible@k) for descent k!=k_opt.
    if kind == "qwoa" and k != inst["k_opt"]:
        target_mask = feasible & (card == k)              # feasible cover at this cardinality
    else:
        target_mask = is_opt                              # the global optimum

    def neg_metric(params):
        qc = (qaoa_circuit(inst, p, params, False) if kind == "qaoa"
              else qwoa_circuit(inst, k, p, params, False))
        pr = Statevector(qc).probabilities()
        return -float(pr[target_mask].sum())              # maximize P(target)

    rng = np.random.default_rng(seed)
    best = (None, np.inf)
    bnds = [(-np.pi, np.pi)] * p + [(0.0, np.pi)] * p
    for _ in range(n_starts):
        x0 = np.concatenate([rng.uniform(-np.pi, np.pi, p), rng.uniform(0, np.pi / 2, p)])
        r = minimize(neg_metric, x0, method="L-BFGS-B", bounds=bnds,
                     options=dict(maxiter=150, ftol=1e-10))
        if r.fun < best[1]:
            best = (r.x, r.fun)
    return best[0], -best[1]                               # (angles, noiseless P(target))


# ----------------------------------------------------------------------------
def build_run_set(insts):
    """Yield dicts describing each circuit to run."""
    jobs = []
    for p in (1, 2, 3):
        jobs.append(dict(tag=f"rung1_qaoa_p{p}", inst="rung1", kind="qaoa", p=p, k=None))
    for p in (1, 2, 3):                                   # Rung-2 Arco (79q) for cross-rung
        jobs.append(dict(tag=f"arco_qaoa_p{p}", inst="arco", kind="qaoa", p=p, k=None))
    for p in (1, 2, 3):
        jobs.append(dict(tag=f"ov_qaoa_p{p}", inst="overshoot", kind="qaoa", p=p, k=None))
    for p in (1, 2):
        jobs.append(dict(tag=f"ov_qwoa_direct_k9_p{p}", inst="overshoot", kind="qwoa", p=p, k=9))
    for k in (12, 11, 10, 9):                         # descent at p=1
        jobs.append(dict(tag=f"ov_qwoa_descent_k{k}_p1", inst="overshoot", kind="qwoa", p=1, k=k))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--backend", default="ibm_kingston")
    ap.add_argument("--shots", type=int, default=8192)
    ap.add_argument("--only", default=None,
                    help="run only one instance group (e.g. --only arco): fresh angles, separate plan")
    args = ap.parse_args()

    insts = dict(rung1=load_instance(DATA_DIR / "frozen_rung1.json"),
                 arco=load_instance(DATA_DIR / "frozen_arco.json"),
                 overshoot=load_instance(DATA_DIR / "frozen_overshoot.json"))
    jobs = build_run_set(insts)
    if args.only:
        jobs = [j for j in jobs if j["inst"] == args.only]
        print(f"  --only {args.only}: {len(jobs)} circuits (FRESH angles, reuse disabled)")
    suffix = f"_{args.only}" if args.only else ""
    plan_path = RESULTS_DIR / f"kingston_rerun_plan{suffix}.json"

    # reuse angles from a previous dry run when submitting (skip slow re-optimization);
    # disabled under --only so the re-optimized angles actually take effect.
    stored = {}
    if args.submit and not args.only:
        try:
            for mj in json.load(open(RESULTS_DIR / "kingston_rerun_plan.json")).get("jobs", []):
                stored[mj["tag"]] = mj
            print(f"  reusing stored angles for {len(stored)} circuits from kingston_rerun_plan.json")
        except FileNotFoundError:
            pass

    # 1. optimize angles + build logical circuits + report logical CX
    print(f"{'tag':28} {'n':>3} {'p':>2} {'k':>3} {'logCX':>6} {'depth':>6} {'P_opt_nl':>10}")
    circuits, meta = [], []
    for j in jobs:
        inst = insts[j["inst"]]
        if j["tag"] in stored and "params" in stored[j["tag"]]:
            params = np.array(stored[j["tag"]]["params"])
            p_opt_nl = stored[j["tag"]].get("p_opt_noiseless", float("nan"))
        else:
            params, p_opt_nl = optimise_angles(inst, j["kind"], j["p"], k=j["k"])
        qc = (qaoa_circuit(inst, j["p"], params) if j["kind"] == "qaoa"
              else qwoa_circuit(inst, j["k"], j["p"], params))
        dec = transpile(qc, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1)
        logcx = dec.count_ops().get("cx", 0)
        n = inst["A"].shape[1]
        print(f"{j['tag']:28} {n:>3} {j['p']:>2} {str(j['k']):>3} {logcx:>6} "
              f"{dec.depth():>6} {p_opt_nl:>10.3e}")
        circuits.append(qc)
        meta.append(dict(**j, n=n, params=params.tolist(), logical_cx=logcx,
                         p_opt_noiseless=p_opt_nl))

    summary = dict(backend=args.backend, shots=args.shots, jobs=meta)
    json.dump(summary, open(plan_path, "w"), indent=1)
    print("\n-> kingston_rerun_plan.json (angles + logical CX) written")

    if not args.submit:
        print("DRY RUN complete. Re-run with --submit to execute on hardware.")
        return

    # 2. submit to hardware with DD + Pauli twirling
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2, Batch
    service = QiskitRuntimeService()
    backend = service.backend(args.backend)
    tqc = transpile(circuits, backend=backend, optimization_level=3)
    for m, c in zip(meta, tqc):
        m["physical_cx"] = c.count_ops().get("cz", 0) + c.count_ops().get("cx", 0)
        m["physical_depth"] = c.depth()
        print(f"  {m['tag']}: physical 2q = {m['physical_cx']}, depth = {m['physical_depth']}")
    with Batch(backend=backend):
        sampler = SamplerV2()
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XY4"
        sampler.options.twirling.enable_gates = True
        sampler.options.twirling.enable_measure = True
        job = sampler.run(tqc, shots=args.shots)
        print(f"  submitted job {job.job_id()} ({len(tqc)} circuits x {args.shots} shots)")
    summary["job_id"] = job.job_id()
    json.dump(summary, open(plan_path, "w"), indent=1)
    print("-> job submitted; run analyze_kingston.py <job_id> when complete.")


if __name__ == "__main__":
    main()
