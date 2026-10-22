#!/usr/bin/env python3
"""Is the Rung-1 hardware loss local to each block, or does the device correlate blocks?

The ideal p-layer QAOA state of the frozen Rung-1 instance is an exact product over the 13
blocks of G_ZZ, and so is the set of 288 minimum covers (a free product of per-block optimal
sets).  Hence, for the ideal state, P(opt) = prod_i P_i(block i in Opt_i) exactly.  Applied to
the measured counts, the same product of per-block marginals is a null with no cross-block
correlation: whatever the device does inside a block is kept, correlations between blocks are
removed.  If the measured P(opt) matches it, the loss is local and independent per block; if
not, the hardware has produced cross-block correlation that the ideal circuit cannot.

Four analyses per circuit (p = 1, 2, 3):
  1. point null  prod_i P_hw(Opt_i) and the measured-to-null ratio;
  2. permutation test: every block's columns are permuted independently across shots, which
     preserves each block's joint distribution exactly and destroys only cross-block
     dependence (two-sided Monte Carlo p value);
  3. per-block loss: P_hw(Opt_i) / P_ideal(Opt_i), block by block;
  4. cross-block structure: pairwise phi coefficients of the block indicators, and the
     covariance of block weights (weight-preserving exchanges between blocks would make the
     total weight less variable than the product of block-weight marginals predicts).

Writes results/block_product_null.json.  Offline; no IBM access.
"""
from __future__ import annotations

import json
import math

import numpy as np

from _paths import DATA_DIR, RESULTS_DIR
from calc_block_factorisation import blocks_of, load

N_PERMUTATIONS = 20000
N_BOOT = 4000
SEED = 20260929


def block_state(qubits, h, J, gammas, betas):
    """Ideal p-layer QAOA state on one block (local order: qubits[0] is the MSB)."""
    m = len(qubits)
    pos = {q: t for t, q in enumerate(qubits)}
    z = 1 - 2 * ((np.arange(2 ** m)[:, None] >> np.arange(m)[None, ::-1]) & 1)
    e = z @ h[qubits]
    for (i, j), v in J.items():
        if i in pos and j in pos:
            e = e + v * z[:, pos[i]] * z[:, pos[j]]
    psi = np.full(2 ** m, 1 / math.sqrt(2 ** m), dtype=complex)
    for g, b in zip(gammas, betas):
        t = (np.exp(-1j * g * e) * psi).reshape([2] * m)
        c, s = np.cos(b), -1j * np.sin(b)
        for q in range(m):
            t = np.moveaxis(t, q, 0)
            t = np.stack([c * t[0] + s * t[1], s * t[0] + c * t[1]])
            t = np.moveaxis(t, 0, q)
        psi = t.reshape(-1)
    return np.abs(psi) ** 2


def optimal_block_sets(A, blocks, n, k_opt):
    idx = np.arange(2 ** n)
    bits = ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)
    cand = np.where(bits.sum(1) == k_opt)[0]
    opt = bits[cand[(A @ bits[cand].T >= 1).all(0)]]
    return [{tuple(r) for r in opt[:, list(b)]} for b in blocks], len(opt)


def analyse(tag, counts, params, A, h, J, blocks, opt_sets, n, rng):
    keys = list(counts)
    mult = np.array([counts[k] for k in keys], dtype=np.int64)
    bits = np.array([[int(k.replace(" ", "")[::-1][q]) for q in range(n)] for k in keys],
                    dtype=np.int8)
    shots = np.repeat(np.arange(len(keys)), mult)             # one row per shot
    S = len(shots)
    ind = np.empty((len(blocks), S), dtype=bool)
    wts = np.empty((len(blocks), S), dtype=np.int16)
    for i, b in enumerate(blocks):
        sub = bits[:, list(b)]
        hit = np.array([tuple(r) in opt_sets[i] for r in sub])
        ind[i] = hit[shots]
        wts[i] = sub.sum(1)[shots]
    observed = int(ind.all(0).sum())
    p_block_hw = ind.mean(1)
    point_null = float(np.prod(p_block_hw))

    perm_hits = np.empty(N_PERMUTATIONS, dtype=np.int64)
    for r in range(N_PERMUTATIONS):
        acc = ind[0].copy()
        for i in range(1, len(blocks)):
            acc &= ind[i][rng.permutation(S)]
        perm_hits[r] = acc.sum()
    mean_null = float(perm_hits.mean())
    # bootstrap interval for the cross-block factor (observed / block-product null)
    ind_u = np.array([[tuple(r) in opt_sets[i] for r in bits[:, list(b)]]
                      for i, b in enumerate(blocks)])
    boot = []
    for _ in range(N_BOOT):
        m_ = rng.multinomial(S, mult / S)
        boot.append((m_ * ind_u.all(0)).sum() / S / np.prod((ind_u * m_).sum(1) / S))
    cross_ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    lo = float(np.mean(perm_hits <= observed))
    hi = float(np.mean(perm_hits >= observed))

    p = len(params) // 2
    p_block_id = []
    for i, b in enumerate(blocks):
        pr = block_state(list(b), h, J, params[:p], params[p:])
        m = len(b)
        local = ((np.arange(2 ** m)[:, None] >> np.arange(m)[None, ::-1]) & 1)
        p_block_id.append(float(sum(pr[j] for j in range(2 ** m)
                                    if tuple(local[j]) in opt_sets[i])))
    p_opt_ideal = float(np.prod(p_block_id))

    phi = np.corrcoef(ind.astype(float))
    iu = np.triu_indices(len(blocks), 1)
    z = phi[iu] * math.sqrt(S)
    total_w = wts.sum(0).astype(float)
    var_total = float(total_w.var())
    var_indep = float(sum(w.astype(float).var() for w in wts))
    cov_sum = (var_total - var_indep) / 2

    # the k=10 sector mass implied by independent blocks, against the measured one
    conv = np.array([1.0])
    for w in wts:
        conv = np.convolve(conv, np.bincount(w, minlength=w.max() + 1) / S)
    return dict(
        circuit=tag, shots=S, observed_opt_hits=observed, observed_p_opt=observed / S,
        block_product_null_p_opt=point_null, block_product_null_hits=point_null * S,
        observed_over_null=observed / (point_null * S),
        permutation=dict(n=N_PERMUTATIONS, mean_hits=mean_null,
                         sd_hits=float(perm_hits.std()),
                         p_lower=lo, p_upper=hi, p_two_sided=min(1.0, 2 * min(lo, hi))),
        ideal_p_opt_product=p_opt_ideal,
        blocks=[dict(qubits=[int(q) for q in b], size=len(b), n_opt=len(opt_sets[i]),
                     p_hw=float(p_block_hw[i]), p_ideal=p_block_id[i],
                     hw_over_ideal=float(p_block_hw[i] / p_block_id[i]))
                for i, b in enumerate(blocks)],
        product_of_block_ratios=float(np.prod(p_block_hw / np.array(p_block_id))),
        retention_by_block_size={
            str(size): dict(
                n_blocks=int(sum(len(b) == size for b in blocks)),
                min=float(min(p_block_hw[i] / p_block_id[i] for i, b in enumerate(blocks) if len(b) == size)),
                max=float(max(p_block_hw[i] / p_block_id[i] for i, b in enumerate(blocks) if len(b) == size)),
                product=float(np.prod([p_block_hw[i] / p_block_id[i] for i, b in enumerate(blocks) if len(b) == size])))
            for size in sorted({len(b) for b in blocks})},
        cross_block_factor=float((observed / S) / point_null) if point_null else None,
        cross_block_factor_ci95=cross_ci,
        cross_block=dict(max_abs_phi=float(np.max(np.abs(phi[iu]))),
                         n_pairs=int(len(iu[0])),
                         n_pairs_abs_z_gt_3=int(np.sum(np.abs(z) > 3)),
                         mean_phi=float(np.mean(phi[iu])),
                         var_total_weight=var_total,
                         var_total_weight_if_independent=var_indep,
                         sum_cross_block_weight_covariance=cov_sum,
                         p_k10_measured=float(np.mean(total_w == 10)),
                         p_k10_independent_blocks=float(conv[10])),
    )


def main() -> None:
    d, A, h, J = load()
    n, k_opt = d["n_b"], d["k_opt"]
    blocks = blocks_of(n, J)
    opt_sets, n_opt = optimal_block_sets(A, blocks, n, k_opt)
    assert n_opt == 288 and math.prod(len(s) for s in opt_sets) == 288
    plan = {j["tag"]: j for j in json.load(open(RESULTS_DIR / "kingston_rerun_plan_rung1.json"))["jobs"]}
    archive = json.load(open(RESULTS_DIR / "kingston_100k_rung1_counts.json"))["circuits"]
    rng = np.random.default_rng(SEED)
    out = {"description": __doc__.split("\n\n")[0], "n_blocks": len(blocks),
           "permutations": N_PERMUTATIONS, "seed": SEED, "circuits": []}
    for tag in ("rung1_qaoa_p1", "rung1_qaoa_p2", "rung1_qaoa_p3"):
        row = analyse(tag, archive[tag]["counts"], plan[tag]["params"], A, h, J, blocks,
                      opt_sets, n, rng)
        out["circuits"].append(row)
        pm = row["permutation"]
        print(f"{tag}: observed {row['observed_opt_hits']} vs block-product null "
              f"{row['block_product_null_hits']:.1f} (perm mean {pm['mean_hits']:.1f} "
              f"+/- {pm['sd_hits']:.1f}, two-sided p={pm['p_two_sided']:.4f}); "
              f"ideal check {row['ideal_p_opt_product']:.4e}; "
              f"max|phi|={row['cross_block']['max_abs_phi']:.4f}, "
              f"|z|>3 pairs {row['cross_block']['n_pairs_abs_z_gt_3']}/{row['cross_block']['n_pairs']}; "
              f"P(k=10) meas {row['cross_block']['p_k10_measured']:.4f} vs indep "
              f"{row['cross_block']['p_k10_independent_blocks']:.4f}")
    json.dump(out, open(RESULTS_DIR / "block_product_null.json", "w"), indent=1)
    print("-> results/block_product_null.json")


if __name__ == "__main__":
    main()
