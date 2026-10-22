#!/usr/bin/env python3
"""Statistical audit of the archived IBM Kingston measurements.

This program deliberately keeps three quantities separate:

* an unconditional probability on the full measured register;
* a probability conditional on the observed Hamming-weight sector;
* the legacy paper-v3 ratio, whose numerator is unconditional but whose
  denominator is conditional and is therefore retained only for diagnosis.

It also checks the measured slack equalities of the 51-qubit encoding and builds
an exact cardinality-matched null for the probability that its 24 post bits form
a cover.  No IBM access is used.

Writes results/statistical_audit.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import binom, norm

from _paths import DATA_DIR, RESULTS_DIR
from analyze_kingston import energy as surrogate_energy
from build_exact_circuit import exact_qubo
from calc_block_factorisation import blocks_of


Z95 = float(norm.ppf(0.975))


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def wilson(successes: int, trials: int, z: float = Z95) -> list[float]:
    """Two-sided Wilson score interval for a binomial proportion."""
    if trials <= 0:
        return [float("nan"), float("nan")]
    p = successes / trials
    den = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / den
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials**2)) / den
    return [max(0.0, centre - half), min(1.0, centre + half)]


def mean_and_se(values: np.ndarray, multiplicities: np.ndarray) -> tuple[float, float]:
    n = int(multiplicities.sum())
    mean = float(values @ multiplicities / n)
    if n <= 1:
        return mean, float("nan")
    ss = float(((values - mean) ** 2) @ multiplicities)
    sample_var = ss / (n - 1)
    return mean, math.sqrt(sample_var / n)


def exact_feasible_counts_by_weight(A: np.ndarray) -> np.ndarray:
    """Count covers at every weight by convolving independent base blocks."""
    graph = (A.T.astype(np.int64) @ A.astype(np.int64)) > 0
    np.fill_diagonal(graph, False)
    n_components, labels = connected_components(csr_matrix(graph), directed=False)
    total = np.array([1], dtype=object)
    for component in range(n_components):
        bases = np.where(labels == component)[0]
        targets = np.where(A[:, bases].any(axis=1))[0]
        states = ((np.arange(1 << len(bases))[:, None] >> np.arange(len(bases))) & 1)
        valid = ((A[np.ix_(targets, bases)] @ states.T) >= 1).all(axis=0)
        local = np.zeros(len(bases) + 1, dtype=object)
        for k in range(len(local)):
            local[k] = int(np.sum(valid & (states.sum(axis=1) == k)))
        total = np.convolve(total, local)
    return np.asarray([int(v) for v in total], dtype=object)


def ideal_qaoa_weight_distribution(instance: dict, parameters: list[float]) -> np.ndarray:
    """Exact ideal Hamming-weight distribution, evolved block by block."""
    h = np.asarray(instance["h"], dtype=float)
    couplings = {
        tuple(int(index) for index in key.split(",")): float(value)
        for key, value in instance["J"].items()
    }
    n = int(instance["n_b"])
    depth = len(parameters) // 2
    gammas = parameters[:depth]
    betas = parameters[depth:]
    total = np.asarray([1.0])
    for block in blocks_of(n, couplings):
        m = len(block)
        local_index = {int(qubit): position for position, qubit in enumerate(block)}
        z = 1 - 2 * ((np.arange(1 << m)[:, None] >> np.arange(m)[None, ::-1]) & 1)
        diagonal = z @ h[block]
        for (left, right), value in couplings.items():
            if left in local_index and right in local_index:
                diagonal += value * z[:, local_index[left]] * z[:, local_index[right]]
        state = np.full(1 << m, 1.0 / math.sqrt(1 << m), dtype=complex)
        for gamma, beta in zip(gammas, betas):
            state *= np.exp(-1j * gamma * diagonal)
            tensor = state.reshape([2] * m)
            cosine, sine = np.cos(beta), -1j * np.sin(beta)
            for position in range(m):
                tensor = np.moveaxis(tensor, position, 0)
                tensor = np.stack([
                    cosine * tensor[0] + sine * tensor[1],
                    sine * tensor[0] + cosine * tensor[1],
                ])
                tensor = np.moveaxis(tensor, 0, position)
            state = tensor.reshape(-1)
        weight = np.asarray([index.bit_count() for index in range(1 << m)])
        local = np.bincount(weight, weights=np.abs(state) ** 2, minlength=m + 1)
        total = np.convolve(total, local)
    return total


def classify_rung1_counts(counts: dict[str, int], instance: dict) -> dict:
    A = np.asarray(instance["A"], dtype=int)
    target = np.minimum(instance["m"], A.sum(axis=1))
    k_opt = int(instance["k_opt"])
    keys = list(counts)
    mult = np.asarray([counts[key] for key in keys], dtype=np.int64)
    bits = np.asarray([[int(c) for c in key[::-1]] for key in keys], dtype=np.int8)
    card = bits.sum(axis=1)
    feasible = ((A @ bits.T) >= target[:, None]).all(axis=0)
    optimal = feasible & (card == k_opt)
    energies = np.asarray([surrogate_energy(x, {**instance, "A": A, "target": target})
                           for x in bits], dtype=float)
    mean_e, se_e = mean_and_se(energies, mult)
    return {
        "shots": int(mult.sum()),
        "n_opt": int(mult[optimal].sum()),
        "n_feasible": int(mult[feasible].sum()),
        "n_in_optimal_sector": int(mult[card == k_opt].sum()),
        "mean_cardinality": float(card @ mult / mult.sum()),
        "mean_energy": mean_e,
        "mean_energy_se": se_e,
        "mean_energy_ci95_normal": [mean_e - Z95 * se_e, mean_e + Z95 * se_e],
    }


def ideal_null_test(observed: int, shots: int, p0: float) -> dict:
    expected = shots * p0
    if observed <= expected:
        direction = "deficit"
        p_tail = float(binom.cdf(observed, shots, p0))
    else:
        direction = "excess"
        p_tail = float(binom.sf(observed - 1, shots, p0))
    z_equivalent = float(norm.isf(p_tail)) if 0.0 < p_tail < 1.0 else float("inf")
    return {
        "null_probability": p0,
        "expected_count": expected,
        "null_binomial_sd_count": math.sqrt(shots * p0 * (1.0 - p0)),
        "direction": direction,
        "one_sided_exact_binomial_p": p_tail,
        "one_sided_normal_equivalent_z": z_equivalent,
        "legacy_difference_over_sqrt_observed": (
            abs(expected - observed) / math.sqrt(observed) if observed else None
        ),
    }


def audit_rung1() -> dict:
    instance = load(DATA_DIR / "frozen_rung1.json")
    archive = load(RESULTS_DIR / "kingston_100k_rung1_counts.json")["circuits"]
    aggregate = {r["tag"]: r for r in load(RESULTS_DIR / "kingston_100k_rung1.json")["circuits"]}
    resource = {r["tag"]: r for r in load(RESULTS_DIR / "resource_model_exact.json")["circuits"]}
    plan = {r["tag"]: r for r in load(RESULTS_DIR / "kingston_rerun_plan_rung1.json")["jobs"]}

    n_b = int(instance["n_b"])
    k_opt = int(instance["k_opt"])
    n_optimal_strings = int(load(RESULTS_DIR / "block_factorisation.json")["n_optimal_covers"])
    baseline_all = n_optimal_strings / 2**n_b
    baseline_sector = n_optimal_strings / math.comb(n_b, k_opt)
    rows = []
    for tag, payload in archive.items():
        raw = classify_rung1_counts(payload["counts"], instance)
        shots = raw["shots"]
        n_opt = raw["n_opt"]
        n_sector = raw["n_in_optimal_sector"]
        p_opt = n_opt / shots
        p_sector = n_sector / shots
        p_opt_given_sector = n_opt / n_sector if n_sector else 0.0
        p_ci = wilson(n_opt, shots)
        cond_ci = wilson(n_opt, n_sector)
        p_ideal = float(aggregate[tag]["p_opt_nl"])
        ideal_weight_distribution = ideal_qaoa_weight_distribution(instance, plan[tag]["params"])
        ideal_p_sector = float(ideal_weight_distribution[k_opt])
        ideal_p_opt_given_sector = p_ideal / ideal_p_sector
        F = float(resource[tag]["F_exact"])
        row = {
            "tag": tag,
            **raw,
            "p_opt_unconditional": p_opt,
            "p_opt_unconditional_wilson95": p_ci,
            "p_opt_given_k_opt": p_opt_given_sector,
            "p_opt_given_k_opt_wilson95": cond_ci,
            "p_k_opt": p_sector,
            "uniform_all_register_baseline": baseline_all,
            "uniform_k_opt_sector_baseline": baseline_sector,
            "eta_all_register": p_opt / baseline_all,
            "eta_all_register_wilson95": [v / baseline_all for v in p_ci],
            "eta_conditional_k_opt": p_opt_given_sector / baseline_sector,
            "eta_conditional_k_opt_wilson95": [v / baseline_sector for v in cond_ci],
            "legacy_hybrid_eta_deprecated": p_opt / baseline_sector,
            "ideal_p_opt": p_ideal,
            "ideal_p_k_opt": ideal_p_sector,
            "ideal_p_opt_given_k_opt": ideal_p_opt_given_sector,
            "ideal_eta_all_register": p_ideal / baseline_all,
            "ideal_eta_conditional_k_opt": ideal_p_opt_given_sector / baseline_sector,
            "ideal_null_test": ideal_null_test(n_opt, shots, p_ideal),
            "hardware_to_ideal_absolute_probability_ratio": p_opt / p_ideal,
            "hardware_to_ideal_conditional_probability_ratio": (
                p_opt_given_sector / ideal_p_opt_given_sector
            ),
            "resource_proxy": {
                "F_est": F,
                "qualification": "resource proxy, not a calibrated output-noise probability",
                "legacy_erasure_like_reference_F_times_pideal": F * p_ideal,
                "illustrative_global_depolarizing_mixture": F * p_ideal + (1.0 - F) * baseline_all,
            },
        }
        rows.append(row)
    return {
        "definition": {
            "eta_all_register": "P(opt)/(N_opt/2^n)",
            "eta_conditional_k_opt": "P(opt|k=k_opt)/(N_opt/C(n,k_opt))",
            "legacy_hybrid_eta": "P(opt)/(N_opt/C(n,k_opt)); mixes sample spaces and is deprecated",
        },
        "n_qubits": n_b,
        "k_opt": k_opt,
        "n_optimal_strings": n_optimal_strings,
        "circuits": rows,
    }


def audit_exact51(counts_file: str = "exact51_counts.json",
                  results_file: str = "exact51_results.json") -> dict:
    instance = load(DATA_DIR / "frozen_rung1.json")
    A = np.asarray(instance["A"], dtype=int)
    archive = load(RESULTS_DIR / counts_file)["circuits"]["faithful_exact51_p1"]
    counts = archive["counts"]
    keys = list(counts)
    mult = np.asarray([counts[key] for key in keys], dtype=np.int64)
    shots = int(mult.sum())

    lin, quad, n, n_b, widths, offsets, const = exact_qubo(A, float(instance["lam"]))
    bits = np.asarray([[int(key[::-1][q]) for q in range(n)] for key in keys], dtype=np.int8)
    posts = bits[:, :n_b]
    cardinality = posts.sum(axis=1)
    covered = ((A @ posts.T) >= 1).all(axis=0)

    equalities = np.ones(len(keys), dtype=bool)
    for t in range(A.shape[0]):
        slack = np.zeros(len(keys), dtype=int)
        for j in range(int(widths[t])):
            slack += (2**j) * bits[:, n_b + int(offsets[t]) + j]
        equalities &= ((posts @ A[t]) - 1 - slack) == 0

    energy = bits.astype(float) @ lin + const
    for (i, j), coefficient in quad.items():
        energy += coefficient * bits[:, i] * bits[:, j]
    mean_e, se_e = mean_and_se(energy, mult)

    cover_by_weight = exact_feasible_counts_by_weight(A)
    observed_weight = np.bincount(cardinality, weights=mult, minlength=n_b + 1)
    matched_null_terms = []
    matched_null = 0.0
    for k in range(n_b + 1):
        sector_size = math.comb(n_b, k)
        cover_fraction = int(cover_by_weight[k]) / sector_size
        observed_mass = float(observed_weight[k] / shots)
        matched_null += observed_mass * cover_fraction
        matched_null_terms.append({
            "k": k,
            "observed_probability": observed_mass,
            "n_covers": int(cover_by_weight[k]),
            "sector_size": sector_size,
            "uniform_within_sector_cover_probability": cover_fraction,
        })

    n_cover = int(mult[covered].sum())
    n_equalities = int(mult[equalities].sum())
    uniform_cover = int(sum(cover_by_weight)) / 2**n_b
    strict_uniform = int(sum(cover_by_weight)) / 2**n
    return {
        "job_id": archive.get("job_id", load(RESULTS_DIR / results_file)["job_id"]),
        "shots": shots,
        "n_qubits": n,
        "n_post_qubits": n_b,
        "mean_post_cardinality": float(cardinality @ mult / shots),
        "mean_energy": mean_e,
        "mean_energy_se": se_e,
        "mean_energy_ci95_normal": [mean_e - Z95 * se_e, mean_e + Z95 * se_e],
        "post_bits_form_cover": {
            "count": n_cover,
            "probability": n_cover / shots,
            "wilson95": wilson(n_cover, shots),
            "uniform_24_bit_baseline": uniform_cover,
            "legacy_enhancement": (n_cover / shots) / uniform_cover,
            "cardinality_matched_uniform_within_sector_baseline": matched_null,
            "cardinality_matched_enhancement": (n_cover / shots) / matched_null,
        },
        "all_measured_slack_equalities_satisfied": {
            "count": n_equalities,
            "probability": n_equalities / shots,
            "wilson95": wilson(n_equalities, shots),
            "uniform_51_bit_baseline": strict_uniform,
            "qualification": "this, not post coverage alone, is measured feasibility of the encoded equality constraints",
        },
        "exact_cover_counts_by_post_weight": [int(v) for v in cover_by_weight],
        "cardinality_matched_null_terms": matched_null_terms,
    }


def main() -> None:
    out = {
        "description": "Offline statistical audit; no IBM access",
        "confidence_level": 0.95,
        "rung1": audit_rung1(),
        "faithful_51q": audit_exact51(),
        "faithful_51q_matched_mitigation": audit_exact51("exact51_matched_counts.json",
                                                         "exact51_matched_results.json"),
        "interpretive_decisions": [
            "Use all-register or conditional concentration, never the legacy hybrid ratio as a concentration claim.",
            "Use the binomial variance under the tested ideal null.",
            "Wilson/binomial calculations assume exchangeable independent shots; aggregated counts cannot diagnose drift or overdispersion.",
            "Do not attach statistical significance to F_est without a calibrated stochastic output model.",
            "Call 51q post-bit coverage a marginal cover rate, not encoded-feasibility probability.",
        ],
    }
    path = RESULTS_DIR / "statistical_audit.json"
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")

    print("Rung-1 coherent concentration measures")
    for row in out["rung1"]["circuits"]:
        test = row["ideal_null_test"]
        print(
            f"  {row['tag']}: eta_all={row['eta_all_register']:.2f}, "
            f"eta_cond={row['eta_conditional_k_opt']:.2f}, "
            f"legacy={row['legacy_hybrid_eta_deprecated']:.2f}, "
            f"ideal-null {test['direction']} z={test['one_sided_normal_equivalent_z']:.2f}"
        )
    exact = out["faithful_51q"]
    print(
        "51q: post-cover P={:.5f}, cardinality-matched null={:.5f}, "
        "strict measured equalities={}/{}".format(
            exact["post_bits_form_cover"]["probability"],
            exact["post_bits_form_cover"]["cardinality_matched_uniform_within_sector_baseline"],
            exact["all_measured_slack_equalities_satisfied"]["count"],
            exact["shots"],
        )
    )
    print(f"-> {path}")


if __name__ == "__main__":
    main()
