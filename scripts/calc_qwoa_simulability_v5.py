#!/usr/bin/env python3
"""Analytical block-cut Schmidt audit for the exact global QWOA mixer.

The published QWOA ansatz uses

    H_XY = 1/2 sum_(i,j) w_ij (X_i X_j + Y_i Y_j),
    U_B(beta) = exp(-i beta H_XY),

evaluated spectrally in a fixed-Hamming-weight sector.  Both the cost and mixer
Hamiltonians are block-local, so they are local unitaries across every cut that
is a union of interaction blocks.  They cannot change the Schmidt coefficients
of the Dicke initial state across such a cut.  For a cut containing ``a`` of
``n`` qubits in weight sector ``k``, the squared Schmidt coefficients are the
hypergeometric probabilities

    C(a,j) C(n-a,k-j) / C(n,k).

This calculation avoids the sequential overlapping-edge product used by the
retired diagnostic script, which would be only a first-order Trotter circuit.

Writes results/qwoa_simulability_v5.json.
"""
from __future__ import annotations

import json
import math

from _paths import RESULTS_DIR
from calc_block_factorisation import blocks_of, load


def hypergeometric_schmidt_spectrum(n: int, k: int, a: int) -> list[float]:
    """Return the nonzero squared Schmidt coefficients for an ``a | n-a`` cut."""
    if not (0 <= k <= n and 0 <= a <= n):
        raise ValueError((n, k, a))
    lo = max(0, k - (n - a))
    hi = min(k, a)
    denom = math.comb(n, k)
    probs = [math.comb(a, j) * math.comb(n - a, k - j) / denom
             for j in range(lo, hi + 1)]
    if not math.isclose(sum(probs), 1.0, rel_tol=0.0, abs_tol=1e-14):
        raise AssertionError("Schmidt spectrum is not normalized")
    return probs


def entropy_bits(probabilities: list[float]) -> float:
    return -sum(p * math.log2(p) for p in probabilities if p > 0.0)


def main() -> None:
    data, _A, _h, couplings = load()
    n, k = data["n_b"], data["k_opt"]
    blocks = blocks_of(n, couplings)
    xy_edges = {tuple(sorted(int(x) for x in edge)) for edge in data["xy_edges"]}
    zz_edges = {tuple(sorted(edge)) for edge in couplings}
    assert xy_edges == zz_edges, (
        "the XY mixer is not block-internal; the analytical argument does not apply"
    )

    largest = max(blocks, key=len)
    largest_spectrum = hypergeometric_schmidt_spectrum(n, k, len(largest))

    cuts = []
    prefix_size = 0
    for after_block, block in enumerate(blocks[:-1], start=1):
        prefix_size += len(block)
        spectrum = hypergeometric_schmidt_spectrum(n, k, prefix_size)
        cuts.append({
            "after_block": after_block,
            "prefix_qubits": prefix_size,
            "schmidt_rank": len(spectrum),
            "entropy_bits": entropy_bits(spectrum),
        })

    max_rank = max(row["schmidt_rank"] for row in cuts)
    first_max_rank = next(row for row in cuts if row["schmidt_rank"] == max_rank)
    max_entropy = max(cuts, key=lambda row: row["entropy_bits"])
    proxy = (k + 1) * sum(2 ** len(block) for block in blocks)

    out = {
        "schema_version": 2,
        "method": "analytical_hypergeometric_schmidt_spectrum",
        "mixer_definition": "H_XY=0.5*sum_ij w_ij*(X_i X_j + Y_i Y_j)",
        "mixer_evolution": "global_continuous_time_exact_sector_spectral",
        "hamiltonian_prefactor": 0.5,
        "n_b": n,
        "k_opt": k,
        "n_blocks": len(blocks),
        "block_sizes": [len(block) for block in blocks],
        "largest_block": len(largest),
        "schmidt_rank_largest_block": len(largest_spectrum),
        "entropy_largest_block_bits": entropy_bits(largest_spectrum),
        "maximum_block_cut_rank": max_rank,
        "first_maximum_rank_cut_after_block": first_max_rank["after_block"],
        "maximum_block_cut_entropy_bits": max_entropy["entropy_bits"],
        "maximum_entropy_cut_after_block": max_entropy["after_block"],
        "maximum_entropy_cut_prefix_qubits": max_entropy["prefix_qubits"],
        "bond_dimension_bound": k + 1,
        "bound_holds": max_rank <= k + 1,
        "product_state_rank": 1,
        "mps_simulation_cost": proxy,
        "naive_cost": 2 ** n,
        "block_prefix_cuts": cuts,
    }
    for key, value in out.items():
        if key != "block_prefix_cuts":
            print(f"{key:>40}  {value}")
    output = RESULTS_DIR / "qwoa_simulability_v5.json"
    output.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"\n-> {output.relative_to(RESULTS_DIR.parent)}")


if __name__ == "__main__":
    main()
