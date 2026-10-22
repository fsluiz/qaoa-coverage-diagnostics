#!/usr/bin/env python3
"""Where the hardware lost its concentration, and what the energy reductions are worth.

Two offline analyses of archived counts, added for version 4.

1. Rung-1 sector decomposition.  The measured optimum probability is 43% of the ideal.
   Because every minimum cover has k = 10, the ratio factorises exactly as

       P_hw(opt) / P_id(opt) = [P_hw(k=10) / P_id(k=10)] x [P_hw(opt|k=10) / P_id(opt|k=10)],

   i.e. loss of cardinality-sector mass times loss of selection inside the sector.  The
   ideal factors come from the exact 24-qubit p=1 state vector.

2. Energy nulls.  An energy reduction relative to uniform bitstrings is dominated by the
   linear terms and can be earned by a sampler with the right single-bit marginals and
   no correlations at all.  For each circuit we evaluate, exactly,
     E_prod   the mean energy of the product distribution with the measured marginals;
     E_card   (faithful encoding) posts uniform within the measured post-cardinality
              histogram, slack bits at their measured marginals;
   and report the fraction of the uniform-to-measured reduction that each null already
   achieves.  Everything the circuit does beyond E_prod is attributable to correlations.

Writes results/hardware_decomposition.json.
"""
from __future__ import annotations

import json
import math

import numpy as np

from _paths import DATA_DIR, RESULTS_DIR
from build_exact_circuit import exact_qubo
from calc_block_factorisation import load, qaoa_state

LAM = 10.0


def counts_to_bits(counts: dict, n: int) -> tuple[np.ndarray, np.ndarray]:
    keys = list(counts)
    # qiskit prints qubit 0 rightmost
    bits = np.array([[int(k.replace(" ", "")[::-1][q]) for q in range(n)] for k in keys],
                    dtype=np.int8)
    mult = np.array([counts[k] for k in keys], dtype=float)
    return bits, mult


def qubo_energy(bits, lin, quad, const):
    e = bits @ lin + const
    for (i, j), v in quad.items():
        e = e + v * bits[:, i] * bits[:, j]
    return e


def product_null(marg, lin, quad, const):
    return float(const + marg @ lin + sum(v * marg[i] * marg[j] for (i, j), v in quad.items()))


def sector_decomposition() -> dict:
    d, A, h, J = load()
    n, k = d["n_b"], d["k_opt"]
    g = d["qaoa_angles"]["1"]["gamma"][0]
    b = d["qaoa_angles"]["1"]["beta"][0]
    p = np.abs(qaoa_state(list(range(n)), h, J, g, b)) ** 2
    idx = np.arange(2 ** n)
    bits = ((idx[:, None] >> np.arange(n - 1, -1, -1)[None, :]) & 1).astype(np.int8)
    w = bits.sum(1)
    cand = np.where(w == k)[0]
    cov = cand[(A @ bits[cand].T >= 1).all(0)]
    p_k_id, p_opt_id = float(p[w == k].sum()), float(p[cov].sum())
    card_id = [float(p[w == j].sum()) for j in range(n + 1)]

    arc = json.load(open(RESULTS_DIR / "kingston_100k_rung1_counts.json"))
    circ = arc["circuits"]
    tag = next(t for t in circ if t.endswith("p1"))
    hb, hm = counts_to_bits(circ[tag]["counts"], n)
    shots = hm.sum()
    hw_w = hb.sum(1)
    covered = (A @ hb.T >= 1).all(0)
    p_k_hw = float(hm[hw_w == k].sum() / shots)
    p_opt_hw = float(hm[(hw_w == k) & covered].sum() / shots)
    card_hw = [float(hm[hw_w == j].sum() / shots) for j in range(n + 1)]
    null_k = 288 / math.comb(n, k)
    sector_ratio = p_k_hw / p_k_id
    within_ratio = (p_opt_hw / p_k_hw) / (p_opt_id / p_k_id)
    # success proxy of this very circuit (Eq. resource_exact), read rather than copied
    F = next(c["F_exact"] for c in json.load(open(RESULTS_DIR / "resource_model_exact.json"))["circuits"]
             if c["job_id"] == "dakjch39k43c73aik9rg" and c["tag"] == tag)
    return dict(
        circuit=tag, shots=int(shots),
        p_k10_ideal=p_k_id, p_k10_hw=p_k_hw, p_k10_uniform=math.comb(n, k) / 2 ** n,
        p_opt_ideal=p_opt_id, p_opt_hw=p_opt_hw,
        p_opt_given_k10_ideal=p_opt_id / p_k_id, p_opt_given_k10_hw=p_opt_hw / p_k_hw,
        eta_k10_ideal=(p_opt_id / p_k_id) / null_k, eta_k10_hw=(p_opt_hw / p_k_hw) / null_k,
        sector_mass_ratio=sector_ratio, within_sector_ratio=within_ratio,
        total_ratio=sector_ratio * within_ratio,
        log_share_within_sector=math.log(within_ratio) / math.log(sector_ratio * within_ratio),
        mean_k_ideal=float(np.dot(np.arange(n + 1), card_id)),
        mean_k_hw=float(np.dot(np.arange(n + 1), card_hw)),
        cardinality_ideal=card_id, cardinality_hw=card_hw,
        global_depolarising_eta_all=F * (p_opt_id / (288 / 2 ** n)) + (1 - F),
    )


def rung1_energy_null() -> dict:
    d, A, h, J = load()
    n = d["n_b"]
    lam = d["lam"]
    # QUBO of Eq. (2): sum x + lam sum_t (1 - A x)^2
    lin = 1.0 - lam * A.sum(0).astype(float)
    quad = {}
    for i in range(n):
        for j in range(i + 1, n):
            s = int((A[:, i] & A[:, j]).sum())
            if s:
                quad[(i, j)] = 2 * lam * s
    const = lam * A.shape[0]
    arc = json.load(open(RESULTS_DIR / "kingston_100k_rung1_counts.json"))["circuits"]
    tag = next(t for t in arc if t.endswith("p1"))
    bits, mult = counts_to_bits(arc[tag]["counts"], n)
    shots = mult.sum()
    e_hw = float(qubo_energy(bits.astype(float), lin, quad, const) @ mult / shots)
    e_rand = product_null(np.full(n, 0.5), lin, quad, const)
    marg = (bits * mult[:, None]).sum(0) / shots
    e_prod = product_null(marg, lin, quad, const)
    return dict(circuit=tag, energy_random=e_rand, energy_hw=e_hw, energy_product_null=e_prod,
                reduction_hw_pct=100 * (1 - e_hw / e_rand),
                reduction_product_null_pct=100 * (1 - e_prod / e_rand),
                product_null_share_of_hw_reduction=(e_rand - e_prod) / (e_rand - e_hw))


def faithful_energy_nulls(counts_file: str, plan_file: str) -> dict:
    d = json.load(open(DATA_DIR / "frozen_rung1.json"))
    A = np.asarray(d["A"], int)
    lin, quad, n, n_b, _, _, const = exact_qubo(A, LAM)
    plan = json.load(open(RESULTS_DIR / plan_file))
    arc = json.load(open(RESULTS_DIR / counts_file))
    if arc["job_id"] != plan["job_id"]:
        raise ValueError("counts and plan refer to different jobs")
    counts = next(iter(arc["circuits"].values()))["counts"]
    bits, mult = counts_to_bits(counts, n)
    shots = mult.sum()
    bf = bits.astype(float)
    e_hw = float(qubo_energy(bf, lin, quad, const) @ mult / shots)
    e_rand = product_null(np.full(n, 0.5), lin, quad, const)
    marg = (bf * mult[:, None]).sum(0) / shots
    e_prod = product_null(marg, lin, quad, const)

    # cardinality-matched null: posts uniform within the measured weight histogram,
    # slack bits independent at their measured marginals
    kpost = bits[:, :n_b].sum(1)
    hist = np.bincount(kpost, weights=mult, minlength=n_b + 1) / shots
    e_card = 0.0
    for kk, pk in enumerate(hist):
        if pk == 0:
            continue
        m1 = marg.copy()
        m1[:n_b] = kk / n_b
        e = const + m1 @ lin
        for (i, j), v in quad.items():
            if i < n_b and j < n_b:
                e += v * kk * (kk - 1) / (n_b * (n_b - 1))
            else:
                e += v * m1[i] * m1[j]
        e_card += pk * e
    e_ideal = plan["energy_ideal"]
    red = lambda e: 100 * (1 - e / e_rand)
    return dict(
        job_id=plan["job_id"], mitigation=plan.get("mitigation", "sampler defaults"),
        energy_random=e_rand, energy_hw=e_hw, energy_ideal=e_ideal,
        energy_product_null=e_prod, energy_cardinality_null=e_card,
        reduction_hw_pct=red(e_hw), reduction_ideal_pct=red(e_ideal),
        reduction_product_null_pct=red(e_prod), reduction_cardinality_null_pct=red(e_card),
        gap_closed_vs_ideal=(e_rand - e_hw) / (e_rand - e_ideal),
        product_null_share_of_hw_reduction=(e_rand - e_prod) / (e_rand - e_hw),
        cardinality_null_share_of_hw_reduction=(e_rand - e_card) / (e_rand - e_hw),
        correlation_contribution=(e_prod - e_hw) / (e_rand - e_hw),
    )


def faithful_ideal_product_null() -> dict:
    """The same product-marginal null applied to the ideal p=1 state (exact, blockwise)."""
    from build_exact_circuit import blocks_from_edges
    from submit_exact51 import block_energy_table
    d = json.load(open(DATA_DIR / "frozen_rung1.json"))
    A = np.asarray(d["A"], int)
    lin, quad, n, _, _, _, const = exact_qubo(A, LAM)
    plan = json.load(open(RESULTS_DIR / "exact51_plan.json"))
    g, b = plan["gamma"], plan["beta"]
    marg, energy = np.zeros(n), const
    for blk in blocks_from_edges(n, quad.keys()):
        bits, e = block_energy_table(blk, lin, quad)
        m = len(blk)
        t = (np.exp(-1j * g * e) / np.sqrt(2 ** m)).reshape([2] * m)
        c, s = np.cos(b), -1j * np.sin(b)
        for q in range(m):
            t = np.moveaxis(t, q, 0)
            t = np.stack([c * t[0] + s * t[1], s * t[0] + c * t[1]])
            t = np.moveaxis(t, 0, q)
        p = np.abs(t.reshape(-1)) ** 2
        marg[list(blk)] = p @ bits
        energy += float(p @ e)
    e_rand = product_null(np.full(n, 0.5), lin, quad, const)
    e_prod = product_null(marg, lin, quad, const)
    return dict(energy_random=e_rand, energy_ideal=energy, energy_product_null_ideal=e_prod,
                product_null_share_of_ideal_reduction=(e_rand - e_prod) / (e_rand - energy),
                correlation_energy_ideal=e_prod - energy)


def main() -> None:
    out = {"rung1_sector_decomposition": sector_decomposition(),
           "rung1_energy_null": rung1_energy_null(),
           "faithful_default": faithful_energy_nulls("exact51_counts.json", "exact51_plan.json")}
    out["faithful_ideal"] = faithful_ideal_product_null()
    fd, fi = out["faithful_default"], out["faithful_ideal"]
    fd["correlation_energy_hw"] = fd["energy_product_null"] - fd["energy_hw"]
    fd["correlation_energy_retained_vs_ideal"] = (fd["correlation_energy_hw"]
                                                  / fi["correlation_energy_ideal"])
    try:
        out["faithful_matched"] = faithful_energy_nulls("exact51_matched_counts.json",
                                                        "exact51_matched_plan.json")
        fm = out["faithful_matched"]
        fm["correlation_energy_hw"] = fm["energy_product_null"] - fm["energy_hw"]
        fm["correlation_energy_retained_vs_ideal"] = (fm["correlation_energy_hw"]
                                                      / fi["correlation_energy_ideal"])
    except FileNotFoundError:
        out["faithful_matched"] = None
    for key, val in out.items():
        print(f"== {key}")
        if val:
            for k, v in val.items():
                if not isinstance(v, list):
                    print(f"   {k:>40}  {v}")
    json.dump(out, open(RESULTS_DIR / "hardware_decomposition.json", "w"), indent=1)
    print("-> results/hardware_decomposition.json")


if __name__ == "__main__":
    main()
