#!/usr/bin/env python3
"""Does the cardinality-preserving ansatz escape the factorisation argument?

Equation (factor) needs a product initial state, and the QWOA initial state is a Dicke state,
which is not one.  The conclusion survives anyway, in a weaker form, for a reason specific to
this construction: the XY mixer acts on the cooperation edges, and those are exactly the ZZ
couplings, so the mixer is internal to the blocks of G_ZZ.  Both the cost operator and the
mixer therefore conserve the Hamming weight of each block separately, and the Dicke state is
a superposition over ways of splitting k across the blocks:

    |D^n_k> = sum_{j_1+...+j_m=k} c_j  |D^{|V_1|}_{j_1}> x ... x |D^{|V_m|}_{j_m}>

Every term evolves independently.  Across any cut of the block partition the Schmidt rank is
therefore at most the number of achievable partial weights, k+1, so the state is a matrix
product state of bond dimension <= k+1 in the block ordering and its output distribution is
obtained by convolution at cost O(k sum_i 2^{|V_i|}) rather than 2^n.

This script verifies the bound on the state vector instead of asserting it.

Writes results/qwoa_simulability.json.
"""
from __future__ import annotations

import json

import numpy as np

from _paths import DATA_DIR, RESULTS_DIR
from calc_block_factorisation import blocks_of, load


def dicke_state(n, k):
    """|D^n_k>: the uniform superposition of all weight-k basis states."""
    idx = np.arange(2 ** n)
    w = np.zeros(2 ** n, dtype=np.int8)
    for q in range(n):
        w += ((idx >> q) & 1).astype(np.int8)
    sel = np.where(w == k)[0]
    psi = np.zeros(2 ** n, complex)
    psi[sel] = 1.0 / np.sqrt(len(sel))
    return psi


def apply_cost(psi, n, h, J, gamma):
    m = int(np.log2(len(psi)))
    z = 1 - 2 * ((np.arange(2 ** m)[:, None] >> np.arange(m)[None, ::-1]) & 1)
    E = z @ h
    for (i, j), v in J.items():
        E = E + v * z[:, i] * z[:, j]
    return psi * np.exp(-1j * gamma * E)


def apply_xy(psi, n, edges, beta):
    """exp(-i beta (XX+YY)) on each edge: identity on |00>,|11>, a rotation on {|01>,|10>}."""
    c, s = np.cos(2 * beta), -1j * np.sin(2 * beta)
    t = psi.reshape([2] * n)
    for (i, j) in edges:
        t = np.moveaxis(t, (i, j), (0, 1))
        a01, a10 = t[0, 1].copy(), t[1, 0].copy()
        t[0, 1] = c * a01 + s * a10
        t[1, 0] = s * a01 + c * a10
        t = np.moveaxis(t, (0, 1), (i, j))
    return t.reshape(-1)


def schmidt(psi, n, part):
    rest = [q for q in range(n) if q not in set(part)]
    m = np.moveaxis(psi.reshape([2] * n), list(part) + rest, range(n)).reshape(2 ** len(part), -1)
    sv = np.linalg.svd(m, compute_uv=False)
    sv = sv[sv > 1e-10]
    p = sv ** 2
    p = p / p.sum()
    return len(sv), float(-(p * np.log2(p)).sum())


def main() -> None:
    d, A, h, J = load()
    n, k = d["n_b"], d["k_opt"]
    g = d["qaoa_angles"]["1"]["gamma"][0]
    b = d["qaoa_angles"]["1"]["beta"][0]
    blk = blocks_of(n, J)
    xy = [tuple(int(x) for x in e) for e in d["xy_edges"]]
    assert {tuple(sorted(e)) for e in xy} == {tuple(sorted(e)) for e in J}, \
        "the XY mixer is not block-internal; the argument below does not apply"

    psi = dicke_state(n, k)
    psi = apply_cost(psi, n, h, J, g)
    psi = apply_xy(psi, n, xy, b)
    psi /= np.linalg.norm(psi)

    big = max(blk, key=len)
    rank_big, ent_big = schmidt(psi, n, list(big))

    # the worst cut of the block partition: split the blocks into a prefix and the rest
    order = [q for s in blk for q in s]
    worst_rank, worst_ent, worst_at = 0, 0.0, None
    cum = 0
    for i in range(1, len(blk)):
        cum += len(blk[i - 1])
        r, e = schmidt(psi, n, order[:cum])
        if r > worst_rank:
            worst_rank, worst_ent, worst_at = r, e, i
    # a product state would have rank 1; the bound is the number of achievable weight splits
    out = dict(n_b=n, k_opt=k, n_blocks=len(blk),
               largest_block=int(len(big)),
               schmidt_rank_largest_block=rank_big,
               entropy_largest_block_bits=round(ent_big, 4),
               worst_block_cut_rank=worst_rank,
               worst_block_cut_entropy_bits=round(worst_ent, 4),
               worst_cut_after_block=worst_at,
               bond_dimension_bound=k + 1,
               bound_holds=bool(worst_rank <= k + 1),
               product_state_rank=1,
               mps_simulation_cost=int((k + 1) * sum(2 ** len(s) for s in blk)),
               naive_cost=int(2 ** n))
    for key, v in out.items():
        print(f"{key:>32}  {v}")
    print(f"\n  QWOA is entangled across blocks (rank {worst_rank} > 1) but the rank is capped "
          f"at k+1 = {k + 1},")
    print(f"  so the distribution is obtained at {out['mps_simulation_cost']:,} operations "
          f"against 2^{n} = {2 ** n:,}.")
    json.dump(out, open(RESULTS_DIR / "qwoa_simulability.json", "w"), indent=1)
    print("\n-> results/qwoa_simulability.json")


if __name__ == "__main__":
    main()
