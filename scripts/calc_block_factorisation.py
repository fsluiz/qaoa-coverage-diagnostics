#!/usr/bin/env python3
"""The ideal logical circuit, read through its own block structure.

The cost operator of the frozen Rung-1 instance is an exact direct sum over the 13
components of G_ZZ.  The QAOA mixer is a product of single-qubit rotations and the initial
state is a product, so the whole p-layer evolution factorises: the ideal logical unitary
submitted on 24 qubits is equivalent to 13 block circuits of at most four qubits and carries
no ideal entanglement between them.  This script verifies that on the state vector and draws the consequence the
manuscript needs -- the noiseless reference, which is computable in closed form precisely
because of the factorisation.

Writes results/block_factorisation.json.
"""
from __future__ import annotations

import json
import math

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from _paths import DATA_DIR, RESULTS_DIR


def load(tag="frozen_rung1.json"):
    d = json.load(open(DATA_DIR / tag))
    J = {tuple(int(x) for x in k.split(",")): float(v) for k, v in d["J"].items()}
    return d, np.asarray(d["A"], int), np.asarray(d["h"], float), J


def blocks_of(n, J):
    C = np.zeros((n, n), int)
    for (i, j) in J:
        C[i, j] = C[j, i] = 1
    nc, lab = connected_components(csr_matrix(C), directed=False)
    return [np.where(lab == c)[0] for c in range(nc)]


def qaoa_state(qubits, h, J, gamma, beta):
    """One p=1 QAOA layer on a sub-register, using only the couplings internal to it."""
    m = len(qubits)
    idx = {q: t for t, q in enumerate(qubits)}
    z = 1 - 2 * ((np.arange(2 ** m)[:, None] >> np.arange(m)[None, ::-1]) & 1)
    E = z @ h[qubits]
    for (i, j), v in J.items():
        if i in idx and j in idx:
            E = E + v * z[:, idx[i]] * z[:, idx[j]]
    psi = (np.exp(-1j * gamma * E) / np.sqrt(2 ** m)).reshape([2] * m)
    c_, s_ = np.cos(beta), -1j * np.sin(beta)
    for t in range(m):
        psi = np.moveaxis(psi, t, 0)
        psi = np.stack([c_ * psi[0] + s_ * psi[1], s_ * psi[0] + c_ * psi[1]])
        psi = np.moveaxis(psi, 0, t)
    return psi.reshape(-1)


def main() -> None:
    d, A, h, J = load()
    n, k_opt = d["n_b"], d["k_opt"]
    g = d["qaoa_angles"]["1"]["gamma"][0]
    b = d["qaoa_angles"]["1"]["beta"][0]
    blk = blocks_of(n, J)

    full = qaoa_state(list(range(n)), h, J, g, b)
    prod = np.ones(1, complex)
    order: list[int] = []
    for s in blk:
        prod = np.kron(prod, qaoa_state(list(s), h, J, g, b))
        order += list(s)
    prod = np.moveaxis(prod.reshape([2] * n), range(n),
                       np.argsort(np.argsort(order))).reshape(-1)
    fidelity = float(abs(np.vdot(prod, full)) ** 2)

    big = max(blk, key=len)
    rest = [q for q in range(n) if q not in set(big.tolist())]
    m = np.moveaxis(full.reshape([2] * n), list(big) + rest, range(n)).reshape(2 ** len(big), -1)
    sv = np.linalg.svd(m, compute_uv=False)
    sv = sv[sv > 1e-12] ** 2
    entropy = float(-(sv * np.log2(sv)).sum())

    # the optimal covers, and whether they too are a free product across blocks
    bits = ((np.arange(2 ** n)[:, None] >> np.arange(n - 1, -1, -1)[None, :]) & 1).astype(np.int8)
    weight_ok = bits.sum(1) == k_opt
    cand = np.where(weight_ok)[0]
    covers = cand[(A @ bits[cand].T >= 1).all(0)]
    per_block = [len({tuple(bits[s, q] for q in s_) for s in covers}) for s_ in blk]

    p_ideal = float(np.abs(full[covers]) ** 2 @ np.ones(len(covers)))
    p_insector = len(covers) / float(math.comb(n, k_opt))
    p_uniform = len(covers) / 2.0 ** n

    out = dict(n_b=n, k_opt=k_opt, n_blocks=len(blk),
               block_sizes=sorted((len(s) for s in blk), reverse=True),
               product_state_fidelity=round(fidelity, 15),
               entanglement_entropy_bits=entropy,
               n_optimal_covers=int(len(covers)),
               optima_per_block=per_block,
               optima_product=int(np.prod(per_block)),
               optima_factorise=bool(np.prod(per_block) == len(covers)),
               p_ideal=p_ideal, p_insector_baseline=p_insector,
               p_uniform_baseline=p_uniform,
               eta_ideal_insector=p_ideal / p_insector,
               eta_ideal_uniform=p_ideal / p_uniform,
               largest_block_qubits=int(len(big)),
               classical_simulation_cost=int(sum(2 ** len(s) for s in blk)))
    for k, v in out.items():
        print(f"{k:>28}  {v}")
    json.dump(out, open(RESULTS_DIR / "block_factorisation.json", "w"), indent=1)
    print("\n-> results/block_factorisation.json")


if __name__ == "__main__":
    main()
