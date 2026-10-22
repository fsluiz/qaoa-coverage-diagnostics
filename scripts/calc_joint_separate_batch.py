#!/usr/bin/env python3
"""The cost of executing all blocks at once, measured in one batch on the same qubits.

Job dau04mahcrkc73dur170 runs a new transpilation of the joint Rung-1 circuit (1e5 shots; not
the compiled circuit of the principal run dakjch39k43c73aik9rg) and, after it in the same job,
each of its 13 block slices alone (1e4 shots each): the same physical gates on the same
physical qubits as the joint pub.  Order, shot budget and drift are not controlled here; the
interleaved replication is calc_interleaved_batch.py.

For each block i we compare P(Opt_i) measured in the joint run (the block marginal) with the
same probability measured when the block runs alone, both relative to the exact ideal P(Opt_i).
We report the per-block and product retentions, bootstrap 95% intervals for the joint/separate
ratio, the block-product null of the joint run (cross-block factor with a bootstrap interval),
and the job-time calibration proxy per block (identical gates and qubits, so it must agree).

Writes results/joint_separate_batch.json.  Offline.
"""
from __future__ import annotations

import json
import math

import numpy as np

from _paths import RESULTS_DIR
from calc_block_factorisation import blocks_of, load
from calc_block_product_null import block_state, optimal_block_sets

N_BOOT = 4000
SEED = 20260930


def main() -> None:
    d, A, h, J = load()
    n, k_opt = d["n_b"], d["k_opt"]
    blocks = [sorted(int(q) for q in b) for b in blocks_of(n, J)]
    opt_sets, _ = optimal_block_sets(A, blocks, n, k_opt)
    plan = json.load(open(RESULTS_DIR / "joint_separate_batch_plan.json"))
    arc = json.load(open(RESULTS_DIR / "joint_separate_batch_counts.json"))
    assert arc["job_id"] == plan["job_id"]
    g, b_ = plan["angles"]["gamma"], plan["angles"]["beta"]
    rng = np.random.default_rng(SEED)

    # joint run: per-shot block indicators
    cj = arc["pubs"]["joint"]["counts"]
    keys = list(cj)
    mult = np.array([cj[k] for k in keys])
    bits = np.array([[int(k.replace(" ", "")[::-1][q]) for q in range(n)] for k in keys], np.int8)
    ind = np.array([[tuple(r) in opt_sets[i] for r in bits[:, blk]] for i, blk in enumerate(blocks)])
    S = int(mult.sum())
    pj = (ind * mult).sum(1) / S
    obs_hits = int((mult * ind.all(0)).sum())

    rows = []
    for i, blk in enumerate(blocks):
        meta = next(p for p in plan["pubs"] if p["tag"] != "joint" and p["logical_qubits"] == blk)
        cs = arc["pubs"][meta["tag"]]["counts"]
        m = len(blk)
        hit = sum(v for k, v in cs.items()
                  if tuple(int(k.replace(" ", "")[::-1][t]) for t in range(m)) in opt_sets[i])
        ss = sum(cs.values())
        pr = block_state(blk, h, J, [g], [b_])
        local = ((np.arange(2 ** m)[:, None] >> np.arange(m)[None, ::-1]) & 1)
        pid = float(sum(pr[t] for t in range(2 ** m) if tuple(local[t]) in opt_sets[i]))
        phys = meta["measured_physical"]
        F = np.prod([1 - arc["readout_errors"][str(q)] for q in phys.values()])
        for e in meta["two_qubit_edges"]:
            err = arc["two_qubit_errors"][f"{e['edge'][0]},{e['edge'][1]}"]
            F *= (1 - err) ** e["count"]
        rows.append(dict(qubits=blk, size=m, physical=sorted(phys.values()),
                         n_2q=meta["n_2q"], p_ideal=pid,
                         p_joint=float(pj[i]), p_separate=hit / ss, shots_separate=ss,
                         hits_separate=hit, retention_joint=float(pj[i] / pid),
                         retention_separate=hit / ss / pid, F_proxy=float(F)))

    # bootstrap: the joint arm resamples whole shots (multinomial), which keeps the correlation
    # between blocks measured in the same shot; each separate block ran in its own pub, so it
    # has no cross-block correlation to keep and is resampled binomially.  The asymmetry is
    # deliberate.
    boot_ratio, boot_cross, boot_ratio_blk = [], [], {i: [] for i in range(len(blocks))}
    boot_ratio_2q = []
    two_q = np.array([len(b_) > 1 for b_ in blocks])
    p_mult = mult / S
    for _ in range(N_BOOT):
        m_ = rng.multinomial(S, p_mult)
        pj_b = (ind * m_).sum(1) / S
        ps_b = np.array([rng.binomial(r["shots_separate"], r["p_separate"]) / r["shots_separate"]
                         for r in rows])
        rj = np.prod(pj_b / [r["p_ideal"] for r in rows])
        rs = np.prod(ps_b / [r["p_ideal"] for r in rows])
        boot_ratio.append(rj / rs)
        boot_ratio_2q.append(float(np.prod((pj_b / np.maximum(ps_b, 1e-12))[two_q])))
        boot_cross.append((m_ * ind.all(0)).sum() / S / np.prod(pj_b))
        for i in range(len(blocks)):
            boot_ratio_blk[i].append(pj_b[i] / max(ps_b[i], 1e-12))
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    for i, r in enumerate(rows):
        r["joint_over_separate"] = r["p_joint"] / r["p_separate"]
        r["joint_over_separate_ci95"] = ci(boot_ratio_blk[i])
    prod_j = float(np.prod([r["retention_joint"] for r in rows]))
    prod_s = float(np.prod([r["retention_separate"] for r in rows]))
    null = float(np.prod(pj))
    ideal = float(np.prod([r["p_ideal"] for r in rows]))
    out = dict(
        job_id=plan["job_id"], calibration_date=arc["calibration_date"],
        joint_shots=S, joint_opt_hits=obs_hits, joint_p_opt=obs_hits / S,
        ideal_p_opt=ideal, joint_over_ideal=obs_hits / S / ideal,
        within_block_retention_joint=prod_j, within_block_retention_separate=prod_s,
        joint_over_separate=prod_j / prod_s, joint_over_separate_ci95=ci(boot_ratio),
        cross_block_factor=obs_hits / S / null, cross_block_factor_ci95=ci(boot_cross),
        block_product_null_hits=null * S,
        F_proxy_product=float(np.prod([r["F_proxy"] for r in rows])),
        joint_eta_all=obs_hits / S / (288 / 2 ** n),
        joint_over_separate_two_qubit_blocks=float(np.prod(
            [r["joint_over_separate"] for r in rows if r["size"] > 1])),
        joint_over_separate_two_qubit_blocks_ci95=ci(boot_ratio_2q),
        joint_over_separate_single_qubit_blocks=float(np.prod(
            [r["joint_over_separate"] for r in rows if r["size"] == 1])),
        blocks=rows)
    json.dump(out, open(RESULTS_DIR / "joint_separate_batch.json", "w"), indent=1)
    print(f"joint P(opt) = {obs_hits}/{S} = {obs_hits / S:.2e} ({out['joint_over_ideal']:.3f} of ideal)")
    print(f"within-block retention: joint {prod_j:.3f}, separate {prod_s:.3f}; "
          f"joint/separate = {prod_j / prod_s:.3f} {np.round(ci(boot_ratio), 3)}")
    print(f"blocks with two-qubit gates: {out['joint_over_separate_two_qubit_blocks']:.3f} "
          f"{np.round(out['joint_over_separate_two_qubit_blocks_ci95'], 3)}; single-qubit blocks "
          f"{out['joint_over_separate_single_qubit_blocks']:.3f}; joint eta_all {out['joint_eta_all']:.2f}")
    print(f"cross-block factor {out['cross_block_factor']:.3f} {np.round(ci(boot_cross), 3)}")
    for r in sorted(rows, key=lambda r: -r["size"]):
        print(f"  {str(r['qubits']):14} n2q={r['n_2q']:2d}  joint {r['retention_joint']:.3f}  "
              f"separate {r['retention_separate']:.3f}  ratio {r['joint_over_separate']:.3f} "
              f"{np.round(r['joint_over_separate_ci95'], 3)}")
    print("-> results/joint_separate_batch.json")


if __name__ == "__main__":
    main()
