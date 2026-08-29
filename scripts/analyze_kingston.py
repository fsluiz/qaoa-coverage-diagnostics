#!/usr/bin/env python3
"""
Analyze the IBM Kingston re-run.  Retrieves the SamplerV2 result, classifies every
measured bitstring against the frozen instances, and reports per circuit:
  P(opt)      -- min-cardinality feasible m-cover  (Rung-1: k=10 full cover)
  P(feasible) -- cov_t >= min(m, deg_t) for all tasks
  <k>, mean QUBO energy, energy reduction vs random, and the random baselines.

This is where the infeasibility shows up as DATA: for the deep QWOA circuits we expect
P(feasible) ~ in-sector random (noise), while the shallow Rung-1 QAOA should concentrate.

  python analyze_kingston.py <job_id>     # or reads job_id from kingston_rerun_plan.json
Writes kingston_rerun_results.json.  Author: assistant for F. S. Luiz (IFGW/UNICAMP).
"""
import sys, json
import numpy as np
from math import comb

from _paths import DATA_DIR, RESULTS_DIR


def load_instance(path):
    d = json.load(open(path))
    d["A"] = np.asarray(d["A"], int)
    d["target"] = np.minimum(d["m"], d["A"].sum(1))           # per-task target
    return d


def classify(bitstr, inst):
    """bitstr is qiskit little-endian (qubit 0 = rightmost). Return (card, feasible, opt)."""
    x = np.array([int(c) for c in bitstr[::-1]], int)         # x[qubit]
    cov = inst["A"] @ x
    card = int(x.sum())
    feasible = bool((cov >= inst["target"]).all())
    opt = feasible and card == inst["k_opt"]
    return card, feasible, opt, x


def energy(x, inst):
    cov = inst["A"] @ x
    return float(x.sum() + inst["lam"] * ((inst["target"] - cov) ** 2).sum())


def analyze_counts(counts, inst):
    shots = sum(counts.values())
    p_opt = p_feas = 0.0
    e_mean = 0.0
    kbar = 0.0
    for bs, c in counts.items():
        card, feas, opt, x = classify(bs, inst)
        w = c / shots
        p_opt += w * opt
        p_feas += w * feas
        e_mean += w * energy(x, inst)
        kbar += w * card
    return dict(shots=shots, p_opt=p_opt, p_feas=p_feas, mean_energy=e_mean, mean_k=kbar)


def random_baselines(inst, k=None, n_samples=200000):
    """In-sector (weight-k) random P(opt)/P(feasible).  Enumerate if C(n,k) is small,
    else Monte-Carlo sample (n=79 cannot be enumerated)."""
    from math import comb
    from itertools import combinations
    A, n = inst["A"], inst["A"].shape[1]
    if k is None:
        k = inst["k_opt"]
    rng = np.random.default_rng(0)
    nf = no = 0
    if comb(n, k) <= 2_000_000:                                # exact enumeration
        states = list(combinations(range(n), k)); dim = len(states)
        for ones in states:
            x = np.zeros(n, int); x[list(ones)] = 1
            f = (A @ x >= inst["target"]).all()
            nf += f; no += (f and k == inst["k_opt"])
        return dict(p_opt_rnd=no / dim, p_feas_rnd=nf / dim, dim=dim)
    for _ in range(n_samples):                                 # Monte-Carlo weight-k
        x = np.zeros(n, int); x[rng.choice(n, k, replace=False)] = 1
        f = (A @ x >= inst["target"]).all()
        nf += f; no += (f and k == inst["k_opt"])
    return dict(p_opt_rnd=no / n_samples, p_feas_rnd=nf / n_samples, dim=f"~C({n},{k})")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("job_id", nargs="?", default=None)
    ap.add_argument("--plan", default=RESULTS_DIR / "kingston_rerun_plan.json",
                    help="plan JSON (use kingston_rerun_plan_arco.json for the Arco-only run)")
    ap.add_argument("--out", default=RESULTS_DIR / "kingston_rerun_results.json")
    a = ap.parse_args()
    plan = json.load(open(a.plan))
    job_id = a.job_id or plan.get("job_id")
    if not job_id:
        sys.exit("no job_id (pass as arg or run submit with --submit first)")
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService()
    result = service.job(job_id).result()

    insts = dict(rung1=load_instance(DATA_DIR / "frozen_rung1.json"),
                 arco=load_instance(DATA_DIR / "frozen_arco.json"),
                 overshoot=load_instance(DATA_DIR / "frozen_overshoot.json"))
    out = []
    print(f"{'tag':28} {'P_opt':>9} {'P_feas':>9} {'<k>':>5} {'Ered%':>7} {'P_feas_rnd':>10}")
    for meta, pub in zip(plan["jobs"], result):
        inst = insts[meta["inst"]]
        counts = pub.data.meas.get_counts()
        r = analyze_counts(counts, inst)
        base = random_baselines(inst, k=meta["k"]) if meta["kind"] == "qwoa" \
            else random_baselines(inst, k=inst["k_opt"])
        # energy reduction vs random (normalized to the optimum)
        e_rnd = base.get("p_feas_rnd")                        # placeholder; full E_rnd below
        r.update(tag=meta["tag"], logical_cx=meta["logical_cx"], p_opt_nl=meta["p_opt_noiseless"],
                 **{k: v for k, v in base.items() if k != "note"})
        out.append(r)
        print(f"{meta['tag']:28} {r['p_opt']:>9.3e} {r['p_feas']:>9.3e} {r['mean_k']:>5.1f} "
              f"{'':>7} {base.get('p_feas_rnd', float('nan')):>10.3e}")
    json.dump(dict(job_id=job_id, circuits=out), open(a.out, "w"), indent=1)
    print("\n-> kingston_rerun_results.json written")
    print("Interpretation: QWOA rows with P_feas ~ P_feas_rnd are NOISE-DOMINATED (the "
          "infeasibility result); the Rung-1 QAOA p=1 row is the clean hardware demo.")


if __name__ == "__main__":
    main()
