#!/usr/bin/env python3
"""The faithful inequality encoding, and what it costs.

The cost operator used for the hardware runs is the two-sided quadratic surrogate

    H = sum_b x_b + lam sum_t (1 - sum_b A_tb x_b)^2 ,

which penalises over-coverage as well as under-coverage and therefore encodes *exact* cover
rather than the inequality Ax >= 1.  Its ground state is consequently not the minimum cover,
as Sec. IV reports directly from enumeration.  The faithful encoding introduces one slack
register per target,

    sum_b A_tb x_b - 1 - s_t = 0 ,   s_t in {0, ..., deg_t - 1} ,

with s_t written in binary.  This script measures what that costs and -- the question that
matters for the manuscript -- whether the block decomposition survives it.

It does, for a structural reason: the slack bits of target t couple only to the bases that
cover t, and those already lie in a single component, so the ancillas join that block rather
than bridging two.  The register grows, the couplings grow, the block count does not.

Writes results/exact_encoding.json.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from _paths import DATA_DIR, RESULTS_DIR
from calc_block_decomposition import decompose, interaction_graph


def slack_widths(A):
    """Bits needed for s_t in {0, ..., deg_t - 1}.  A degree-one target needs none: its
    single base is forced, and the constraint is already an equality."""
    deg = A.sum(1)
    return [int(np.ceil(np.log2(d))) if d > 1 else 0 for d in deg]


def extended_graph(A):
    """Interaction graph of the faithful encoding, on bases followed by slack bits."""
    n_t, n_b = A.shape
    w = slack_widths(A)
    n = n_b + sum(w)
    C = np.zeros((n, n), np.int8)
    off = n_b
    groups = []
    for t in range(n_t):
        bits = list(range(off, off + w[t]))
        off += w[t]
        # squaring the linear form makes every pair inside {bases of t} U {slacks of t} couple
        groups.append(list(np.where(A[t] == 1)[0]) + bits)
    for grp in groups:
        for i in grp:
            for j in grp:
                if i != j:
                    C[i, j] = 1
    np.fill_diagonal(C, 0)
    return C, w


def verify_faithful_ground_space(A, lam=10):
    """Enumerate base assignments component by component for the slack penalty.

    After minimizing over the nonnegative slack register, a covered target has zero
    equality penalty and an uncovered target contributes ``lam``.  Coverage terms never
    span components of the base interaction graph, so the global ground energy and
    degeneracy are products/sums of independently enumerated block results.  For every
    minimizing base assignment the corresponding slack value is unique.
    """
    C = interaction_graph(A)
    n_components, labels = connected_components(csr_matrix(C), directed=False)
    total_energy = 0
    total_degeneracy = 1
    all_ground_states_feasible = True
    block_records = []

    for component in range(n_components):
        bases = np.where(labels == component)[0]
        tasks = np.where(A[:, bases].sum(axis=1) > 0)[0]
        # Every task's cover set is confined to one component.
        assert np.all(A[tasks][:, labels != component].sum(axis=1) == 0)
        minimum = None
        minimizers = []
        for mask in range(1 << len(bases)):
            x = np.array([(mask >> j) & 1 for j in range(len(bases))], dtype=int)
            cover = A[tasks][:, bases] @ x
            energy = int(x.sum() + lam * np.count_nonzero(cover == 0))
            feasible = bool(np.all(cover >= 1))
            if minimum is None or energy < minimum:
                minimum = energy
                minimizers = [(mask, feasible)]
            elif energy == minimum:
                minimizers.append((mask, feasible))
        block_records.append({
            "n_bases": int(len(bases)),
            "n_tasks": int(len(tasks)),
            "ground_energy": int(minimum),
            "ground_degeneracy": int(len(minimizers)),
            "all_ground_states_feasible": bool(all(flag for _, flag in minimizers)),
        })
        total_energy += int(minimum)
        total_degeneracy *= len(minimizers)
        all_ground_states_feasible &= all(flag for _, flag in minimizers)

    return {
        "penalty_lambda": int(lam),
        "method": "exact_base_block_enumeration_after_analytic_slack_minimization",
        "ground_energy": int(total_energy),
        "ground_degeneracy": int(total_degeneracy),
        "all_ground_states_feasible": bool(all_ground_states_feasible),
        "blocks": block_records,
    }


def analyse(tag, fn):
    d = json.load(open(DATA_DIR / fn))
    A = np.asarray(d["A"], int)
    n_t, n_b = A.shape
    base = decompose(A)
    C, w = extended_graph(A)
    n = C.shape[0]
    nc, lab = connected_components(csr_matrix(C), directed=False)
    sizes = sorted(np.bincount(lab).tolist(), reverse=True)
    assert sum(sizes) == n
    return dict(
        tag=tag, n_b=n_b, n_t=n_t,
        surrogate=dict(n_qubits=n_b, n_ZZ=base["n_ZZ"], blocks=base["n_components"],
                       largest_block=base["largest_block"],
                       enumeration_cost=base["enumeration_cost"]),
        exact=dict(n_slack=int(sum(w)), n_qubits=int(n), n_ZZ=int(C.sum() // 2),
                   blocks=int(nc), largest_block=int(sizes[0]), block_sizes=sizes,
                   enumeration_cost=int(sum(2 ** s for s in sizes)),
                   log2_naive=int(n)),
        blocks_preserved=bool(nc == base["n_components"]),
        faithfulness_lambda_10=verify_faithful_ground_space(A, lam=10))


def main() -> None:
    out = {"instances": {}}
    print(f"{'instance':>10}  {'encoding':>10}{'qubits':>8}{'n_ZZ':>7}{'blocks':>8}"
          f"{'largest':>9}{'exact cost':>13}")
    for tag, fn in (("rung1", "frozen_rung1.json"), ("arco", "frozen_arco.json")):
        r = analyse(tag, fn)
        out["instances"][tag] = r
        for name in ("surrogate", "exact"):
            s = r[name]
            print(f"{tag if name == 'surrogate' else '':>10}  {name:>10}"
                  f"{s['n_qubits']:>8}{s['n_ZZ']:>7}{s['blocks']:>8}"
                  f"{s['largest_block']:>9}{s['enumeration_cost']:>13,}")
        assert r["blocks_preserved"], f"{tag}: the faithful encoding changed the block count"
        assert r["faithfulness_lambda_10"]["all_ground_states_feasible"], (
            f"{tag}: lambda=10 has an infeasible ground state"
        )
    print("\nthe faithful encoding preserves the block count on every instance: "
          f"{all(v['blocks_preserved'] for v in out['instances'].values())}")
    json.dump(out, open(RESULTS_DIR / "exact_encoding.json", "w"), indent=1)
    print("-> results/exact_encoding.json")


if __name__ == "__main__":
    main()
