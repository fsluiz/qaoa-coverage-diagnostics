#!/usr/bin/env python3
"""Block decomposition of the QAOA cost Hamiltonian.

A QUBO whose interaction graph G_ZZ has n_B vertices and n_ZZ edges has at least
n_B - n_ZZ connected components.  For a bipartite coverage instance the cost operator is an
*exact* direct sum over those components, because every target's cover-set is a clique in
G_ZZ and therefore lies inside a single component.  The exact optimum is then obtained by
enumerating each block independently, at cost sum_i 2^{|block_i|} rather than 2^{n_B}.

This is the sharpest available statement of "the executable instance is already decided":
it is a theorem about the circuit the paper runs, not a correlation.

Writes results/block_decomposition.json.
"""
from __future__ import annotations

import json
import math
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from _paths import DATA_DIR, RESULTS_DIR


def interaction_graph(A: np.ndarray) -> np.ndarray:
    # Promote before the product: the entries count shared targets, and a narrow input
    # dtype (int8 from a distance threshold) would wrap and silently drop edges.
    B = np.asarray(A, dtype=np.int64)
    C = (B.T @ B > 0).astype(int)
    np.fill_diagonal(C, 0)
    return C


def decompose(A: np.ndarray) -> dict:
    """Components of G_ZZ, and a check that the cost operator really is a direct sum."""
    C = interaction_graph(A)
    n_b = A.shape[1]
    n_zz = int(C.sum() // 2)
    n_comp, lab = connected_components(csr_matrix(C), directed=False)
    sizes = sorted(Counter(lab.tolist()).values(), reverse=True)

    # Direct-sum condition: each target's cover-set must lie inside one component.
    split = [int(t) for t in range(A.shape[0])
             if len({int(lab[b]) for b in np.where(A[t] == 1)[0]}) > 1]

    return dict(n_b=n_b, n_ZZ=n_zz, n_components=int(n_comp),
                lower_bound=max(n_b - n_zz, 1),
                block_sizes=sizes, largest_block=sizes[0],
                singletons=int(sum(1 for s in sizes if s == 1)),
                enumeration_cost=int(sum(2 ** s for s in sizes)),
                naive_cost=int(2 ** n_b),  # exact: float 2.0**n_b overflows past n_b=1023
                log2_naive_cost=int(n_b),
                log2_enumeration_cost=round(math.log2(sum(2 ** s for s in sizes)), 3),
                targets_spanning_blocks=split,
                direct_sum=len(split) == 0)


def main() -> None:
    out = {"instances": {}}
    print(f"{'instance':20} {'n_B':>4} {'n_ZZ':>5} {'blocks':>7} {'largest':>8} "
          f"{'enum cost':>12} {'vs 2^n_B':>12}")
    for tag, f in (("rung1", "frozen_rung1.json"), ("arco", "frozen_arco.json"),
                   ("overshoot", "frozen_overshoot.json")):
        d = json.load(open(DATA_DIR / f))
        r = decompose(np.asarray(d["A"], int))
        r["m"] = d["m"]
        out["instances"][tag] = r
        print(f"{tag:20} {r['n_b']:>4} {r['n_ZZ']:>5} {r['n_components']:>7} "
              f"{r['largest_block']:>8} {r['enumeration_cost']:>12,} "
              f"{r['naive_cost']:>12.3g}")
        assert r["direct_sum"], f"{tag}: cost operator is not a direct sum"

    # Does the decomposition generalise?  Re-derive it across the published ensemble.
    from calc_ensemble_squeeze import haversine, load_sites, make_instance
    sites = load_sites()
    rows = []
    for n_b in (20, 40, 80, 160):
        rng0 = np.random.default_rng(0)
        sub = sites[rng0.choice(len(sites), n_b, replace=False)]
        Ds = haversine(sub[:, 0], sub[:, 1], sub[:, 0], sub[:, 1])
        np.fill_diagonal(Ds, np.inf)
        delta = float(np.median(Ds.min(1)))
        for f in (0.30, 0.45, 0.60, 0.80, 1.00, 1.30):
            R = f * delta * 3.0
            for s in range(25):
                rng = np.random.default_rng(10_000 * n_b + 100 * int(f * 100) + s)
                A = make_instance(sites, n_b, n_b, rng, R)
                if A.shape[0] < 5:
                    continue
                r = decompose(A)
                rows.append(dict(n_b=n_b, R_over_delta=round(R / delta, 3), seed=s,
                                 n_ZZ=r["n_ZZ"], n_components=r["n_components"],
                                 lower_bound=r["lower_bound"],
                                 largest_block=r["largest_block"],
                                 log2_enum=math.log2(max(r["enumeration_cost"], 1)),
                                 direct_sum=r["direct_sum"]))
    out["ensemble"] = rows
    out["ensemble_all_direct_sum"] = all(r["direct_sum"] for r in rows)
    print(f"\nensemble: {len(rows)} instances, cost operator is a direct sum in all of them: "
          f"{out['ensemble_all_direct_sum']}")
    print(f"\n{'n_B':>5} {'R/delta':>8} {'blocks':>8} {'largest':>8} {'log2(enum)':>11} {'n_B':>6}")
    for n_b in (20, 40, 80, 160):
        for rd in (0.90, 1.35, 1.80, 2.40, 3.00, 3.90):
            sel = [r for r in rows if r["n_b"] == n_b and abs(r["R_over_delta"] - rd) < 0.02]
            if sel:
                print(f"{n_b:>5} {rd:>8.2f} {np.mean([r['n_components'] for r in sel]):>8.1f} "
                      f"{np.mean([r['largest_block'] for r in sel]):>8.1f} "
                      f"{np.mean([r['log2_enum'] for r in sel]):>11.1f} {n_b:>6}")
    json.dump(out, open(RESULTS_DIR / "block_decomposition.json", "w"), indent=1)
    print("\n-> results/block_decomposition.json")


if __name__ == "__main__":
    main()
