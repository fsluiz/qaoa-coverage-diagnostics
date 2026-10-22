#!/usr/bin/env python3
"""Run the faithful 51-qubit encoding on hardware, measuring what it can actually show.

`build_exact_circuit.py` proves offline that this operator's ground state is exactly the 288
minimum covers, unlike the two-sided surrogate whose argmin is 144 infeasible weight-nine
states.  This script calculates the ideal optimum probability instead of assuming it; the
maximum is of order 1e-8, so P(opt) is not measurable with 10^5 shots.  The hardware run
therefore targets the two quantities that are measurable at that budget:

  energy reduction   <H> against the value a uniform random bitstring gives
  P(feasible)        the fraction of shots whose 24 post bits form a full cover

Angles are chosen to minimise <H>, which is what QAOA optimises, rather than to maximise a
probability that cannot be reached.  Because the operator is block-local the expectation
factorises, so the objective is evaluated exactly and cheaply.

    python submit_exact51.py                  # build, optimise, transpile, report -- no job
    python submit_exact51.py --submit         # the above, then run on hardware
    python submit_exact51.py --submit --matched-mitigation --out ../results/exact51_matched_plan.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.optimize import minimize

from _paths import DATA_DIR, RESULTS_DIR
from build_exact_circuit import (blocks_from_edges, exact_qubo, qaoa_circuit_exact,
                                 qubo_to_ising)

LAM = 10.0


def block_energy_table(block, lin, quad):
    """All bitstrings of one block with their energy contribution."""
    m = len(block)
    pos = {q: t for t, q in enumerate(block)}
    bits = ((np.arange(2 ** m)[:, None] >> np.arange(m - 1, -1, -1)) & 1).astype(np.int8)
    e = bits @ lin[block]
    for (i, j), v in quad.items():
        if i in pos and j in pos:
            e = e + v * bits[:, pos[i]] * bits[:, pos[j]]
    return bits, e


def expected_energy(params, blocks, lin, quad, tables):
    """<H> of the p=1 QAOA state, evaluated block by block (the operator is block-local)."""
    g, b = params
    total = 0.0
    for blk, (bits, e) in zip(blocks, tables):
        m = len(blk)
        psi = np.exp(-1j * g * e) / np.sqrt(2 ** m)
        c, s = np.cos(b), -1j * np.sin(b)
        t = psi.reshape([2] * m)
        for q in range(m):
            t = np.moveaxis(t, q, 0)
            t = np.stack([c * t[0] + s * t[1], s * t[0] + c * t[1]])
            t = np.moveaxis(t, 0, q)
        total += float(np.abs(t.reshape(-1)) ** 2 @ e)
    return total


def ground_probability(params, blocks, tables):
    """Probability of the global ground manifold, exactly factorised over blocks."""
    g, b = params
    total = 1.0
    for block, (_, energy) in zip(blocks, tables):
        m = len(block)
        state = (np.exp(-1j * g * energy) / np.sqrt(2 ** m)).reshape([2] * m)
        c, s = np.cos(b), -1j * np.sin(b)
        for q in range(m):
            state = np.moveaxis(state, q, 0)
            state = np.stack([c * state[0] + s * state[1],
                              s * state[0] + c * state[1]])
            state = np.moveaxis(state, 0, q)
        probabilities = np.abs(state.reshape(-1)) ** 2
        total *= float(probabilities[np.isclose(energy, energy.min())].sum())
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--shots", type=int, default=100000)
    ap.add_argument("--backend", default="ibm_kingston")
    ap.add_argument("--instance", default=None, help="IBM instance name, e.g. open-instance")
    ap.add_argument("--out", default=RESULTS_DIR / "exact51_plan.json")
    ap.add_argument("--matched-mitigation", action="store_true",
                    help="XY4 dynamical decoupling + gate and measurement twirling, "
                         "the settings of the QAOA jobs (submit_kingston.py)")
    a = ap.parse_args()

    d = json.load(open(DATA_DIR / "frozen_rung1.json"))
    A = np.asarray(d["A"], int)
    lin, quad, n, n_b, w, slack_off, const = exact_qubo(A, LAM)
    h, J = qubo_to_ising(lin, quad, n)
    blocks = blocks_from_edges(n, quad.keys())
    tables = [block_energy_table(b, lin, quad) for b in blocks]
    print(f"faithful encoding: {n} qubits ({n_b} posts + {n - n_b} slack), "
          f"n_ZZ={len(quad)}, {len(blocks)} blocks")

    # angles that minimise <H>; grid then polish, on the exact factorised objective
    best = (np.inf, 0.0, 0.0)
    for g in np.linspace(-np.pi / 2, np.pi / 2, 41):
        for b in np.linspace(-np.pi / 2, np.pi / 2, 41):
            v = expected_energy((g, b), blocks, lin, quad, tables)
            if v < best[0]:
                best = (v, g, b)
    r = minimize(expected_energy, x0=[best[1], best[2]],
                 args=(blocks, lin, quad, tables), method="Nelder-Mead",
                 options={"xatol": 1e-5, "fatol": 1e-9})
    gamma, beta = float(r.x[0]), float(r.x[1])

    # exact_qubo() keeps the additive constant out of lin/quad, so it must be put back
    # before any energy is quoted: without it the ground state reads -220 instead of k_opt=10
    # and the "reduction" exceeds 100%.
    e_rand = sum(float(e.mean()) for _, e in tables) + const
    e_ideal = float(r.fun) + const
    e_min = sum(float(e.min()) for _, e in tables) + const
    assert abs(e_min - d["k_opt"]) < 1e-6, (
        f"ground energy {e_min} should equal k_opt={d['k_opt']}; the constant is wrong")
    print(f"angles minimising <H>: gamma={gamma:+.4f} beta={beta:+.4f}")
    print(f"  <H> random  = {e_rand:.2f}")
    print(f"  <H> ideal   = {e_ideal:.2f}   -> ideal reduction {100 * (1 - e_ideal / e_rand):.1f}%")
    print(f"  ground      = {e_min:.2f}")

    # Independently optimise the probability of the ground manifold.  This is not
    # the objective used for the hardware angles; it quantifies why a measured
    # optimum count is not informative at the available shot budget.
    best_p = (-1.0, 0.0, 0.0)
    for g in np.linspace(-np.pi / 2, np.pi / 2, 41):
        for b in np.linspace(-np.pi / 2, np.pi / 2, 41):
            value = ground_probability((g, b), blocks, tables)
            if value > best_p[0]:
                best_p = (value, g, b)
    rp = minimize(lambda x: -ground_probability(x, blocks, tables),
                  x0=[best_p[1], best_p[2]], method="Nelder-Mead",
                  options={"xatol": 1e-9, "fatol": 1e-18})
    p_at_energy_angles = ground_probability((gamma, beta), blocks, tables)
    p_max = -float(rp.fun)
    print(f"  P(opt) at energy angles = {p_at_energy_angles:.6e}")
    print(f"  max P(opt) = {p_max:.6e} at gamma={rp.x[0]:+.6f}, beta={rp.x[1]:+.6f}")

    qc = qaoa_circuit_exact(n, h, J, gamma, beta, measure=True)
    out = dict(n_qubits=n, n_posts=n_b, n_slack=n - n_b, n_ZZ=len(quad),
               blocks=len(blocks), gamma=gamma, beta=beta, lam=LAM,
               energy_random=e_rand, energy_ideal=e_ideal, energy_ground=e_min,
               ideal_reduction_pct=100 * (1 - e_ideal / e_rand),
               p_opt_ideal_at_energy_angles=p_at_energy_angles,
               p_opt_ideal_max=p_max,
               p_opt_angles=dict(gamma=float(rp.x[0]), beta=float(rp.x[1])),
               logical_depth=qc.depth(),
               logical_2q=sum(1 for i in qc.data if i.operation.num_qubits == 2),
               shots=a.shots, submitted=False, job_id=None)

    if a.submit:
        from qiskit import transpile
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        be = QiskitRuntimeService(instance=a.instance).backend(a.backend)
        tqc = transpile(qc, backend=be, optimization_level=3, seed_transpiler=11)
        out.update(backend=a.backend, physical_depth=tqc.depth(),
                   physical_2q=sum(1 for i in tqc.data if i.operation.num_qubits == 2))
        print(f"transpiled: depth {out['physical_depth']}, 2q {out['physical_2q']}")
        sampler = SamplerV2(mode=be)
        if a.matched_mitigation:
            sampler.options.dynamical_decoupling.enable = True
            sampler.options.dynamical_decoupling.sequence_type = "XY4"
            sampler.options.twirling.enable_gates = True
            sampler.options.twirling.enable_measure = True
        out.update(mitigation=("XY4 DD + gate/measure twirling" if a.matched_mitigation
                               else "sampler defaults"))
        job = sampler.run([tqc], shots=a.shots)
        out.update(submitted=True, job_id=job.job_id())
        print(f"submitted job {job.job_id()} (1 circuit x {a.shots} shots)")
    else:
        print(f"logical: depth {out['logical_depth']}, 2q {out['logical_2q']}")
        print("DRY RUN -- re-run with --submit to execute on hardware.")

    json.dump(out, open(a.out, "w"), indent=1)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
