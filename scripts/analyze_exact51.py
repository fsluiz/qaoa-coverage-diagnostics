#!/usr/bin/env python3
"""Read the 51-qubit faithful-encoding counts and report what they measure.

P(opt) is not among those things: its calculated ideal p=1 maximum is 8.6e-9, so a zero
here carries no information about the device.  The counts do measure the mean energy of
the implemented operator and the fraction of shots whose 24 post bits form a full cover.

    python analyze_exact51.py --counts results/exact51_counts.json
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

from _paths import DATA_DIR, RESULTS_DIR
from build_exact_circuit import blocks_from_edges, exact_qubo

LAM = 10.0


def energy_of(bits, lin, quad, const):
    e = bits @ lin + const
    for (i, j), v in quad.items():
        e = e + v * bits[:, i] * bits[:, j]
    return e


def exact_feasible_baseline(A):
    """Uniform feasibility probability, enumerated independently per base block."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    graph = (A.T @ A) > 0
    np.fill_diagonal(graph, False)
    n_components, labels = connected_components(csr_matrix(graph), directed=False)
    feasible_assignments = 1
    for component in range(n_components):
        bases = np.where(labels == component)[0]
        targets = np.where(A[:, bases].any(axis=1))[0]
        states = ((np.arange(2 ** len(bases))[:, None] >> np.arange(len(bases))) & 1)
        if len(targets):
            valid = (A[np.ix_(targets, bases)] @ states.T >= 1).all(axis=0)
            feasible_assignments *= int(valid.sum())
        else:
            feasible_assignments *= len(states)
    return feasible_assignments / 2 ** A.shape[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("job_id", nargs="?", default=None)
    ap.add_argument("--counts", default=None,
                    help="archive produced by fetch_kingston_counts.py; avoids IBM access")
    ap.add_argument("--out", default=RESULTS_DIR / "exact51_results.json")
    ap.add_argument("--plan", default=RESULTS_DIR / "exact51_plan.json")
    ap.add_argument("--instance", default=None, help="IBM instance name, e.g. open-instance")
    args = ap.parse_args()

    plan = json.load(open(args.plan))
    job_id = args.job_id or plan["job_id"]
    d = json.load(open(DATA_DIR / "frozen_rung1.json"))
    A = np.asarray(d["A"], int)
    lin, quad, n, n_b, w, slack_off, const = exact_qubo(A, LAM)

    if args.counts:
        archive = json.load(open(args.counts))
        if archive["job_id"] != job_id:
            raise ValueError(f"counts job {archive['job_id']} does not match plan job {job_id}")
        counts = archive["circuits"]["faithful_exact51_p1"]["counts"]
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService
        res = QiskitRuntimeService(instance=args.instance).job(job_id).result()
        counts = res[0].data.meas.get_counts()
    shots = sum(counts.values())

    # qiskit prints the register with qubit 0 rightmost; reverse to index by qubit number
    keys = list(counts)
    bits = np.array([[int(k[::-1][q]) for q in range(n)] for k in keys], dtype=np.int8)
    mult = np.array([counts[k] for k in keys], dtype=float)

    e = energy_of(bits.astype(float), lin, quad, const)
    mean_e = float(e @ mult / shots)
    posts = bits[:, :n_b]
    covered = (A @ posts.T >= 1).all(0)                # every waypoint reached
    p_feas = float(mult[covered].sum() / shots)
    k_mean = float((posts.sum(1) * mult).sum() / shots)
    is_opt = covered & (posts.sum(1) == d["k_opt"])
    p_opt = float(mult[is_opt].sum() / shots)

    # Exact uniform baselines.  For binary y, E[y_i]=1/2 and E[y_i y_j]=1/4.
    e_rand = float(const + lin.sum() / 2 + sum(quad.values()) / 4)
    p_feas_rand = exact_feasible_baseline(A)

    out = dict(job_id=job_id, shots=shots, n_qubits=n,
               gamma=plan["gamma"], beta=plan["beta"],
               mean_energy=mean_e, energy_random=e_rand,
               energy_ideal=plan["energy_ideal"], energy_ground=plan["energy_ground"],
               p_opt_ideal_at_energy_angles=plan["p_opt_ideal_at_energy_angles"],
               p_opt_ideal_max=plan["p_opt_ideal_max"],
               reduction_pct=100 * (1 - mean_e / e_rand),
               ideal_reduction_pct=plan["ideal_reduction_pct"],
               p_feasible=p_feas, p_feasible_random=p_feas_rand,
               feasible_enhancement=(p_feas / p_feas_rand) if p_feas_rand else None,
               mean_k=k_mean, k_opt=d["k_opt"], p_opt=p_opt,
               distinct_strings=len(keys),
               mitigation=plan.get("mitigation", "sampler defaults"),
               physical_2q=plan.get("physical_2q"), physical_depth=plan.get("physical_depth"))
    for k, v in out.items():
        print(f"{k:>24}  {v}")
    err = math.sqrt(p_feas * (1 - p_feas) / shots)
    print(f"\nP(feasible) = {p_feas:.4f} +/- {err:.4f} against {p_feas_rand:.2e} at random")
    print(f"energy reduction {out['reduction_pct']:.1f}% against an ideal ceiling of "
          f"{out['ideal_reduction_pct']:.1f}%")
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
