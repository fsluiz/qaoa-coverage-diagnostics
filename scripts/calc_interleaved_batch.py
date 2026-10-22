#!/usr/bin/env python3
"""Analysis of the interleaved, randomised joint/separate job (daul9oqhcrkc73dvsqj0).

Three circuit families, all archived ISA circuits resubmitted unchanged (submit_interleaved_batch.py):

  A   the joint circuit of batch job dau04mahcrkc73dur170 (J_A) and its 13 slices (S_A)
  B   the joint circuit of the principal run dakjch39k43c73aik9rg (J_B) and its 13 slices (S_B)
  C   the six multi-qubit block circuits of the first separate job dan62g5r85ps73fdlp00

Each of the 34 circuits ran in 4 rounds of 1e4 shots, in a seeded random order.  For every
block i we estimate P(Opt_i) in each pub, relative to the exact ideal P(Opt_i), and report

  * the joint/alone ratio of within-block retention for A and for B (all 13 blocks, the six
    with two-qubit gates, the seven single-qubit ones), pooled over rounds, with a shot
    bootstrap interval and, separately, the spread over the four rounds (drift);
  * the compilation contrast J_A versus J_B in the same job: P(opt), eta_all, within-block
    retention and cross-block factor;
  * the retention of the four-qubit block on its three executions (A and B: qubits 2, 3, 4, 16
    with different assignments and routing; C: qubits 122-124, 136);
  * the replication of the same compiled circuits A across days (against dau04mahcrkc73dur170).

Writes results/interleaved_batch.json.  Offline.
"""
from __future__ import annotations

import json

import numpy as np

from _paths import RESULTS_DIR
from calc_block_factorisation import blocks_of, load
from calc_block_product_null import block_state, optimal_block_sets

N_BOOT = 4000
SEED = 20260930


def decode(counts: dict, width: int) -> tuple[np.ndarray, np.ndarray]:
    keys = list(counts)
    for k in keys:
        assert len(k.replace(" ", "")) == width, (k, width)
    bits = np.array([[int(k.replace(" ", "")[::-1][q]) for q in range(width)] for k in keys],
                    np.int8)
    return bits, np.array([counts[k] for k in keys])


def main() -> None:
    d, A, h, J = load()
    n, k_opt = d["n_b"], d["k_opt"]
    plan = json.load(open(RESULTS_DIR / "interleaved_batch_plan.json"))
    arc = json.load(open(RESULTS_DIR / "interleaved_batch_counts.json"))
    assert arc["job_id"] == plan["job_id"]
    blocks = plan["blocks"]
    assert sorted(map(tuple, blocks)) == sorted(tuple(sorted(int(q) for q in b))
                                                  for b in blocks_of(n, J))
    opt_sets, _ = optimal_block_sets(A, blocks, n, k_opt)
    g, b_ = d["qaoa_angles"]["1"]["gamma"][0], d["qaoa_angles"]["1"]["beta"][0]
    p_ideal = []
    for i, blk in enumerate(blocks):
        m = len(blk)
        pr = block_state(blk, h, J, [g], [b_])
        local = ((np.arange(2 ** m)[:, None] >> np.arange(m)[None, ::-1]) & 1)
        p_ideal.append(float(sum(pr[t] for t in range(2 ** m) if tuple(local[t]) in opt_sets[i])))
    p_ideal = np.array(p_ideal)
    gated = np.array([len(b) > 1 for b in blocks])
    M = len(blocks)
    rounds = plan["rounds"]

    # hits[tag][round] = per-block optimal hits (joint: 13 entries + all-block hits), shots
    hits: dict = {}
    for meta in plan["pubs"]:
        tag, r = meta["tag"], meta["round"]
        counts = arc["pubs"][str(meta["position"])]["counts"]
        width = len(meta["logical_qubits"])
        bits, mult = decode(counts, width)
        if tag.startswith("J_"):
            ind = np.array([[tuple(x) in opt_sets[i] for x in bits[:, blk]]
                            for i, blk in enumerate(blocks)])
            rec = dict(block=(ind * mult).sum(1), all=int((mult * ind.all(0)).sum()),
                       shots=int(mult.sum()), ind=ind, mult=mult)
        else:
            i = blocks.index(meta["logical_qubits"])
            ind = np.array([tuple(x) in opt_sets[i] for x in bits])
            rec = dict(block=int((ind * mult).sum()), shots=int(mult.sum()), index=i)
        hits.setdefault(tag, {})[r] = rec

    def joint_p(tag, rs=None):
        rs = range(rounds) if rs is None else rs
        S = sum(hits[tag][r]["shots"] for r in rs)
        return np.sum([hits[tag][r]["block"] for r in rs], 0) / S, S

    def alone_p(fam, rs=None):
        rs = range(rounds) if rs is None else rs
        p = np.full(M, np.nan)
        for i in range(M):
            tag = f"{fam}_{i:02d}"
            if tag in hits:
                p[i] = sum(hits[tag][r]["block"] for r in rs) / sum(hits[tag][r]["shots"] for r in rs)
        return p

    out = dict(job_id=plan["job_id"], calibration_date=arc["calibration_date"],
               creation_date=arc["creation_date"], usage_seconds=arc["usage_seconds"],
               rounds=rounds, shots_per_pub=plan["shots_per_pub"], blocks=blocks,
               p_ideal=p_ideal.tolist(), families={})
    rng = np.random.default_rng(SEED)
    for fam in ("A", "B"):
        pj, S = joint_p(f"J_{fam}")
        ps = alone_p(f"S_{fam}")
        ratio = pj / ps
        allhits = sum(hits[f"J_{fam}"][r]["all"] for r in range(rounds))
        null = float(np.prod(pj))
        per_round = []
        for r in range(rounds):
            pj_r, _ = joint_p(f"J_{fam}", [r])
            ps_r = alone_p(f"S_{fam}", [r])
            per_round.append(dict(
                all=float(np.prod(pj_r / ps_r)), gated=float(np.prod((pj_r / ps_r)[gated])),
                retention_joint=float(np.prod(pj_r / p_ideal)),
                retention_alone=float(np.prod(ps_r / p_ideal))))
        # shot bootstrap: multinomial within each joint pub, binomial within each slice pub
        boot = {"all": [], "gated": [], "single": [], "cross": []}
        for _ in range(N_BOOT):
            bj = np.zeros(M)
            ball = 0
            for r in range(rounds):
                rec = hits[f"J_{fam}"][r]
                m_ = rng.multinomial(rec["shots"], rec["mult"] / rec["shots"])
                bj += (rec["ind"] * m_).sum(1)
                ball += (m_ * rec["ind"].all(0)).sum()
            bj /= S
            bs = np.array([sum(rng.binomial(hits[f"S_{fam}_{i:02d}"][r]["shots"],
                                            hits[f"S_{fam}_{i:02d}"][r]["block"]
                                            / hits[f"S_{fam}_{i:02d}"][r]["shots"])
                               for r in range(rounds)) / (rounds * plan["shots_per_pub"])
                           for i in range(M)])
            rr = bj / np.maximum(bs, 1e-12)
            boot["all"].append(np.prod(rr))
            boot["gated"].append(np.prod(rr[gated]))
            boot["single"].append(np.prod(rr[~gated]))
            boot["cross"].append(ball / S / np.prod(bj))
        ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
              for k, v in boot.items()}
        logs = np.log([x["gated"] for x in per_round])
        half = 3.182 * logs.std(ddof=1) / np.sqrt(rounds)          # t_{0.975, 3}
        out["families"][fam] = dict(
            joint_shots=S, joint_opt_hits=int(allhits), joint_p_opt=allhits / S,
            joint_over_ideal=allhits / S / float(np.prod(p_ideal)),
            joint_eta_all=allhits / S / (288 / 2 ** n),
            within_block_retention_joint=float(np.prod(pj / p_ideal)),
            within_block_retention_alone=float(np.prod(ps / p_ideal)),
            joint_over_alone=float(np.prod(ratio)), joint_over_alone_ci95=ci["all"],
            joint_over_alone_gated=float(np.prod(ratio[gated])),
            joint_over_alone_gated_ci95=ci["gated"],
            joint_over_alone_single=float(np.prod(ratio[~gated])),
            joint_over_alone_single_ci95=ci["single"],
            joint_over_alone_gated_round_t_ci95=[float(np.exp(logs.mean() - half)),
                                                 float(np.exp(logs.mean() + half))],
            cross_block_factor=allhits / S / null, cross_block_factor_ci95=ci["cross"],
            per_round=per_round,
            per_block=[dict(qubits=blocks[i], p_ideal=float(p_ideal[i]),
                            retention_joint=float(pj[i] / p_ideal[i]),
                            retention_alone=float(ps[i] / p_ideal[i]),
                            joint_over_alone=float(ratio[i])) for i in range(M)])
    pc = alone_p("C")
    out["families"]["C"] = dict(per_block=[dict(qubits=blocks[i], retention_alone=float(pc[i] / p_ideal[i]))
                                           for i in range(M) if not np.isnan(pc[i])])
    # compilation contrast in the same job
    a_, b2 = out["families"]["A"], out["families"]["B"]
    hA, hB, SA, SB = a_["joint_opt_hits"], b2["joint_opt_hits"], a_["joint_shots"], b2["joint_shots"]
    pool = (hA + hB) / (SA + SB)
    z = (hA / SA - hB / SB) / np.sqrt(pool * (1 - pool) * (1 / SA + 1 / SB))
    out["compilation_contrast"] = dict(p_opt_A=hA / SA, p_opt_B=hB / SB, z=float(z),
                                       retention_A=a_["within_block_retention_joint"],
                                       retention_B=b2["within_block_retention_joint"])
    # the four-qubit block on its three placements
    k4 = next(i for i, b in enumerate(blocks) if len(b) == 4)
    out["four_qubit_block"] = dict(
        qubits=blocks[k4],
        A_joint=a_["per_block"][k4]["retention_joint"], A_alone=a_["per_block"][k4]["retention_alone"],
        B_joint=b2["per_block"][k4]["retention_joint"], B_alone=b2["per_block"][k4]["retention_alone"],
        C_alone=next(x["retention_alone"] for x in out["families"]["C"]["per_block"]
                     if x["qubits"] == blocks[k4]),
        C_physical=next(p["physical"] for p in plan["pubs"] if p["tag"] == f"C_{k4:02d}"))
    # first-order proxy of Eq. (resource_exact) on this job's calibration, per circuit
    def proxy(tag):
        meta = next(p for p in plan["pubs"] if p["tag"] == tag)
        F = np.prod([1 - arc["readout_errors"][str(q)] for q in meta["measured_physical"].values()])
        for e in arc["circuit_two_qubit_edges"][tag]:
            F *= (1 - arc["two_qubit_errors"][f"{e['edge'][0]},{e['edge'][1]}"]) ** e["count"]
        return float(F)
    out["F_proxy"] = {t: proxy(t) for t in sorted({p["tag"] for p in plan["pubs"]})}
    out["four_qubit_block"].update(F_proxy_A=out["F_proxy"][f"S_A_{k4:02d}"],
                                   F_proxy_B=out["F_proxy"][f"S_B_{k4:02d}"],
                                   F_proxy_C=out["F_proxy"][f"C_{k4:02d}"])
    # the same compiled circuits A, one day earlier
    prev = json.load(open(RESULTS_DIR / "joint_separate_batch.json"))
    out["replication_of_dau04"] = dict(
        dau04_joint_over_ideal=prev["joint_over_ideal"], now_joint_over_ideal=a_["joint_over_ideal"],
        dau04_retention_joint=prev["within_block_retention_joint"],
        now_retention_joint=a_["within_block_retention_joint"],
        dau04_retention_alone=prev["within_block_retention_separate"],
        now_retention_alone=a_["within_block_retention_alone"],
        dau04_gated_ratio=prev["joint_over_separate_two_qubit_blocks"],
        now_gated_ratio=a_["joint_over_alone_gated"])
    json.dump(out, open(RESULTS_DIR / "interleaved_batch.json", "w"), indent=1)

    for fam in ("A", "B"):
        f = out["families"][fam]
        print(f"{fam}: P(opt) {f['joint_opt_hits']}/{f['joint_shots']} ({f['joint_over_ideal']:.3f} of ideal, "
              f"eta_all {f['joint_eta_all']:.1f}); retention joint {f['within_block_retention_joint']:.3f} "
              f"alone {f['within_block_retention_alone']:.3f}")
        print(f"   joint/alone all {f['joint_over_alone']:.3f} {np.round(f['joint_over_alone_ci95'], 3)}; "
              f"gated {f['joint_over_alone_gated']:.3f} {np.round(f['joint_over_alone_gated_ci95'], 3)} "
              f"(rounds t-CI {np.round(f['joint_over_alone_gated_round_t_ci95'], 3)}); single "
              f"{f['joint_over_alone_single']:.3f} {np.round(f['joint_over_alone_single_ci95'], 3)}; "
              f"cross {f['cross_block_factor']:.2f} {np.round(f['cross_block_factor_ci95'], 2)}")
        print("   per round gated:", [round(x["gated"], 3) for x in f["per_round"]])
    print("compilation:", {k: round(v, 5) for k, v in out["compilation_contrast"].items()})
    print("four-qubit block:", {k: (round(v, 3) if isinstance(v, float) else v)
                               for k, v in out["four_qubit_block"].items()})
    print("replication A:", {k: round(v, 3) for k, v in out["replication_of_dau04"].items()})
    for fam in ("A", "B"):
        for x in out["families"][fam]["per_block"]:
            if len(x["qubits"]) > 1:
                print(f"  {fam} {str(x['qubits']):14} joint {x['retention_joint']:.3f} alone "
                      f"{x['retention_alone']:.3f} ratio {x['joint_over_alone']:.3f}")
    for x in out["families"]["C"]["per_block"]:
        print(f"  C {str(x['qubits']):14} alone {x['retention_alone']:.3f}")
    print("-> results/interleaved_batch.json")


if __name__ == "__main__":
    main()
