#!/usr/bin/env python3
"""Does the hardware correlate the logical blocks that routing joins in the 51-qubit run?

The faithful 51-qubit operator is block-local over 13 logical blocks, and its ideal p=1 state
is a product across them, so every correlation between block observables vanishes ideally.
On the device, the transpiler's routing places two-qubit gates between qubits of different
logical blocks (the blocks of 13, 9 and 5 variables).  For each executed job we compute, from
the archived 10^5-shot counts, the Pearson correlation between the per-shot energies of every
pair of logical blocks, and classify the pairs causally, from the archived transpiled circuit
(results/ibm_circuits/exact51_*.ops.json): the backward light cone of a block is the set of
instructions that can influence its measured qubits, followed through every routing SWAP.

  cone      the two blocks' light cones share at least one instruction,
  adjacent  disjoint cones, but a chip coupler joins their measured qubits (readout-crosstalk
            control),
  distant   neither.

Noise can correlate two blocks through gates only if their cones overlap.  The earlier
classification by the measurement-time map ("routed": a gate joins the qubits that carry the
two blocks at measurement) ignores that SWAPs move logical information during the circuit; it
is kept as a secondary field for comparison.

Significance: z = r sqrt(S) against a Bonferroni threshold for the 78 pairs, a permutation
test (10^4 shuffles of one block across shots) for the cone pairs, and the probability, if pairs
were exchangeable, that the two largest |r| both fall among the cone pairs.

Writes results/exact51_cross_block.json.  Offline.
"""
from __future__ import annotations

import itertools
import json
import math

import numpy as np
from scipy.stats import norm

from _paths import DATA_DIR, RESULTS_DIR
from build_exact_circuit import blocks_from_edges, exact_qubo

LAM = 10.0
N_PERM = 10000
SEED = 20260929
COUNTS = {"defaults": "exact51_counts.json", "matched": "exact51_matched_counts.json"}


def light_cones(ops_file, blocks: list) -> list:
    """Indices of the instructions in the backward light cone of each block's measured qubits."""
    ops = json.load(open(ops_file))
    meas = {c[0]: q[0] for name, q, c in ops if name == "measure"}
    cones = []
    for b in blocks:
        wires, cone = {meas[q] for q in b}, set()
        for k in range(len(ops) - 1, -1, -1):
            name, qs, _ = ops[k]
            if name == "measure" or not set(qs) & wires:
                continue
            cone.add(k)
            wires |= set(qs)
        cones.append(cone)
    return cones, ops


def rung1_reference(z_bonf: float) -> dict:
    """The same statistic on the 24-qubit p=1 run, where no gate crosses a block."""
    from calc_block_factorisation import blocks_of
    d = json.load(open(DATA_DIR / "frozen_rung1.json"))
    A = np.asarray(d["A"], int)
    lam = float(d["lam"])
    Jd = {tuple(int(x) for x in k.split(",")): v for k, v in d["J"].items()}
    blocks = [sorted(int(q) for q in b) for b in blocks_of(d["n_b"], Jd)]
    lab = {q: i for i, b in enumerate(blocks) for q in b}
    lin = 1.0 - lam * A.sum(0)                                     # Eq. (3), q_b
    quad = {(i, j): 2 * lam * int((A[:, i] & A[:, j]).sum())
            for i in range(A.shape[1]) for j in range(i + 1, A.shape[1])
            if (A[:, i] & A[:, j]).any()}
    counts = json.load(open(RESULTS_DIR / "kingston_100k_rung1_counts.json"))["circuits"]["rung1_qaoa_p1"]["counts"]
    keys = list(counts)
    assert all(len(k.replace(" ", "")) == A.shape[1] for k in keys)
    X = np.repeat(np.array([[int(k.replace(" ", "")[::-1][q]) for q in range(A.shape[1])]
                            for k in keys], float),
                  [counts[k] for k in keys], axis=0)
    S = len(X)
    E = np.zeros((S, len(blocks)))
    for i, b in enumerate(blocks):
        e = X[:, b] @ lin[b]
        for (a, c), v in quad.items():
            if lab[a] == i:
                e = e + v * X[:, a] * X[:, c]
        E[:, i] = e
    R = np.corrcoef(E.T)
    z = [float(R[i, j] * math.sqrt(S)) for i, j in itertools.combinations(range(len(blocks)), 2)]
    cones, _ = light_cones(RESULTS_DIR / "ibm_circuits" / "rung1_original.ops.json", blocks)
    overlapping = sum(bool(cones[i] & cones[j])
                      for i, j in itertools.combinations(range(len(blocks)), 2))
    return dict(job_id="dakjch39k43c73aik9rg", shots=S, n_pairs=len(z), z=z,
                max_abs_z=float(max(abs(v) for v in z)),
                n_above_bonferroni=int(sum(abs(v) > z_bonf for v in z)),
                cross_block_two_qubit_gates=0, overlapping_cone_pairs=int(overlapping))


def main() -> None:
    d = json.load(open(DATA_DIR / "frozen_rung1.json"))
    A = np.asarray(d["A"], int)
    lin, quad, n, n_b, _, _, _ = exact_qubo(A, LAM)
    blocks = sorted([sorted(int(q) for q in b) for b in blocks_from_edges(n, quad.keys())],
                    key=lambda b: (-len(b), b))
    lab = {q: i for i, b in enumerate(blocks) for q in b}
    M = len(blocks)
    pairs = list(itertools.combinations(range(M), 2))
    z_bonf = float(norm.isf(0.05 / (2 * len(pairs))))
    layout = json.load(open(RESULTS_DIR / "exact51_physical_layout.json"))
    coupling = {tuple(sorted(e)) for e in
                json.load(open(RESULTS_DIR / "rung1_physical_layout.json"))["coupling_map"]}
    rng = np.random.default_rng(SEED)
    out = {"blocks": [{"index": i, "size": len(b), "qubits": b,
                       "posts": [q for q in b if q < n_b]} for i, b in enumerate(blocks)],
           "n_pairs": len(pairs), "z_bonferroni_5pct": z_bonf, "permutations": N_PERM,
           "runs": {}}
    for tag, fname in COUNTS.items():
        arc = json.load(open(RESULTS_DIR / fname))
        counts = next(iter(arc["circuits"].values()))["counts"]
        keys = list(counts)
        mult = np.array([counts[k] for k in keys])
        assert all(len(k.replace(" ", "")) == n for k in keys)
        bits = np.array([[int(k.replace(" ", "")[::-1][q]) for q in range(n)] for k in keys],
                        dtype=np.int8)
        X = np.repeat(bits, mult, axis=0).astype(float)
        S = len(X)
        E = np.zeros((S, M))
        for i, b in enumerate(blocks):
            e = X[:, b] @ lin[b]
            for (a, c), v in quad.items():
                if lab[a] == i:
                    e = e + v * X[:, a] * X[:, c]
            E[:, i] = e
        lay = layout[tag]
        assert lay["job_id"] == arc["job_id"]
        p2l = {v: int(k) for k, v in lay["logical_to_physical"].items()}
        routed, cross_gates = set(), 0
        for g in lay["two_qubit_edges"]:
            a, c = g["edge"]
            if lab[p2l[a]] != lab[p2l[c]]:
                routed.add(tuple(sorted((lab[p2l[a]], lab[p2l[c]]))))
                cross_gates += g["count"]
        cones, ops = light_cones(RESULTS_DIR / "ibm_circuits" / f"exact51_{tag}.ops.json", blocks)
        shared = {(i, j): cones[i] & cones[j] for i, j in pairs}
        cone_pairs = {p for p, sh in shared.items() if sh}
        coupled = {tuple(sorted((lab[p2l[a]], lab[p2l[c]]))) for a, c in coupling
                   if a in p2l and c in p2l and lab[p2l[a]] != lab[p2l[c]]}
        adjacent = coupled - cone_pairs
        R = np.corrcoef(E.T)
        rows = []
        for i, j in pairs:
            r = float(R[i, j])
            cls = "cone" if (i, j) in cone_pairs else "adjacent" if (i, j) in adjacent else "distant"
            row = dict(blocks=[i, j], sizes=[len(blocks[i]), len(blocks[j])], r=r,
                       z=r * math.sqrt(S), cls=cls, routed_at_measurement=(i, j) in routed,
                       shared_instructions=len(shared[(i, j)]),
                       shared_two_qubit_gates=sum(len(ops[k][1]) == 2 for k in shared[(i, j)]))
            if cls == "cone":
                ej = E[:, j].copy()
                exceed = 0
                for _ in range(N_PERM):
                    rng.shuffle(ej)
                    exceed += abs(np.corrcoef(E[:, i], ej)[0, 1]) >= abs(r)
                row["permutation_p_two_sided"] = (exceed + 1) / (N_PERM + 1)
            rows.append(row)
        ranks = sorted(rows, key=lambda x: -abs(x["r"]))
        top = {tuple(x["blocks"]) for x in ranks[:len(routed)]}
        summary = {}
        for cls in ("cone", "adjacent", "distant"):
            zs = [abs(x["z"]) for x in rows if x["cls"] == cls]
            summary[cls] = dict(n=len(zs), max_abs_z=max(zs) if zs else None,
                                n_above_bonferroni=int(sum(z > z_bonf for z in zs)))
        total_var = float(E.sum(1).var())
        cov_sum = float((total_var - E.var(0).sum()) / 2)
        out["runs"][tag] = dict(
            job_id=arc["job_id"], shots=S, cross_block_two_qubit_gates=cross_gates,
            routed_pairs=sorted([list(p) for p in routed]),
            adjacent_only_pairs=sorted([list(p) for p in adjacent]),
            routed_pairs_are_top_ranked=bool(top == routed),
            cone_pairs=sorted([list(p) for p in cone_pairs]),
            cone_blocks=sorted({i for p in cone_pairs for i in p}),
            cone_wires=len({q for i in {i for p in cone_pairs for i in p}
                            for k in cones[i] for q in ops[k][1]}),
            p_routed_top_ranked_if_exchangeable=1 / math.comb(len(pairs), len(routed)),
            top_two_in_cone_pairs=bool({tuple(x["blocks"]) for x in ranks[:2]} <= cone_pairs),
            p_top_two_in_cone_pairs_if_exchangeable=math.comb(len(cone_pairs), 2)
            / math.comb(len(pairs), 2),
            summary=summary, sum_cross_block_energy_covariance=cov_sum,
            total_energy_variance=total_var, pairs=rows)
        print(f"{tag}: routed {sorted(routed)} ({cross_gates} gates); "
              + "; ".join(f"{c}: n={v['n']}, max|z|={v['max_abs_z']:.1f}, "
                          f">{z_bonf:.2f}: {v['n_above_bonferroni']}"
                          for c, v in summary.items() if v["n"])
              + "; cone perm p = "
              + ", ".join(f"{x['permutation_p_two_sided']:.1e}" for x in rows if x["cls"] == "cone")
              + f"; routed top-ranked: {top == routed}; cone pairs {sorted(cone_pairs)}")
    out["rung1_reference"] = rung1_reference(z_bonf)
    json.dump(out, open(RESULTS_DIR / "exact51_cross_block.json", "w"), indent=1)
    print("-> results/exact51_cross_block.json")


if __name__ == "__main__":
    main()
