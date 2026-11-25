"""Exact output-distribution controls for the v7 diagnostics; no hardware noise fit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from independent_encoding import components

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
# Declared grids: symmetric flips and bias at fixed total transition probability.
FLIP_GRID = np.linspace(0.0, 0.2, 11)
BIAS_GRID = np.linspace(0.0, 0.2, 11)
TRANSITION_SUM = 0.2


def moments(probability, bits):
    return probability @ bits, bits.T @ (probability[:, None] * bits)


def readout_moments(mean, second, p01, p10):
    """Independent stochastic output channels, including their diagonal shot variance."""
    a = np.broadcast_to(np.asarray(p01, float), mean.shape)
    loss = np.broadcast_to(np.asarray(p10, float), mean.shape)
    if np.any(a < 0) or np.any(a > 1) or np.any(loss < 0) or np.any(loss > 1):
        raise ValueError("Transition probabilities must lie in [0, 1].")
    b = 1 - a - loss
    transformed = a + b * mean
    covariance = second - np.outer(mean, mean)
    result = np.outer(transformed, transformed) + np.outer(b, b) * covariance
    np.fill_diagonal(result, transformed)
    return transformed, result


def energy(mean, second, linear, quadratic, constant):
    return float(constant + linear @ mean + np.sum(quadratic * second))


def faithful_reference():
    """Reconstruct the 51-variable QUBO and simulate its components from equalities."""
    frozen = json.loads((PROJECT / "data/frozen_rung1.json").read_text())
    incidence = np.asarray(frozen["A"], dtype=np.int64)
    nt, nb = incidence.shape
    widths = [int(np.ceil(np.log2(d))) for d in incidence.sum(axis=1)]
    n = nb + sum(widths)
    coefficients = np.zeros((nt, n), dtype=np.int64)
    coefficients[:, :nb] = incidence
    cursor = nb
    for target, width in enumerate(widths):
        coefficients[target, cursor:cursor+width] = -(2 ** np.arange(width))
        cursor += width
    penalty = float(frozen["lam"])
    gram = coefficients.T @ coefficients
    linear = np.r_[np.ones(nb), np.zeros(n-nb)] + penalty * (
        np.diag(gram) - 2 * coefficients.sum(axis=0))
    quadratic = np.triu(2 * penalty * gram, 1)
    constant = penalty * nt
    plan = json.loads((PROJECT / "results/exact51_plan.json").read_text())
    adjacency = gram.copy()
    np.fill_diagonal(adjacency, 0)
    blocks = components(adjacency != 0)
    mean, second = np.zeros(n), np.zeros((n, n))
    local_seconds = []
    for block in blocks:
        integers = np.arange(2 ** len(block))
        bits = ((integers[:, None] >> np.arange(len(block))) & 1).astype(float)
        local_q = quadratic[np.ix_(block, block)]
        e = bits @ linear[block] + np.einsum("bi,ij,bj->b", bits, local_q, bits)
        state = np.exp(-1j * plan["gamma"] * e) / np.sqrt(len(bits))
        for q in range(len(block)):
            state = (np.cos(plan["beta"]) * state
                     - 1j * np.sin(plan["beta"]) * state[integers ^ (1 << q)])
        local_mean, local_second = moments(abs(state)**2, bits)
        mean[block] = local_mean
        local_seconds.append(local_second)
    second[:] = np.outer(mean, mean)
    for block, local_second in zip(blocks, local_seconds):
        second[np.ix_(block, block)] = local_second
    return mean, second, linear, quadratic, constant, blocks


def dependence_control():
    """Two two-bit blocks: identical block marginals, different global optimum mass."""
    bits = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(float)
    independent, coupled = np.zeros(16), np.zeros(16)
    independent[[0, 3, 12, 15]] = 0.25
    coupled[[0, 15]] = 0.5
    linear = np.zeros(4)
    quadratic = np.zeros((4, 4))
    quadratic[0, 1] = quadratic[2, 3] = -1
    rows = []
    for label, probability in (("independent", independent), ("coupled", coupled)):
        mean, second = moments(probability, bits)
        e = energy(mean, second, linear, quadratic, 0)
        ep = energy(mean, np.outer(mean, mean), linear, quadratic, 0)
        local_optima = [float(probability[(bits[:, block] == 1).all(axis=1)].sum())
                        for block in ([0, 1], [2, 3])]
        rows.append(dict(distribution=label, energy=e, product_energy=ep,
                         correlation_energy=ep-e, correlation_retention=(ep-e)/0.5,
                         block_optimum_probabilities=local_optima,
                         global_optimum_probability=float(probability[15]),
                         cross_block_factor=float(probability[15]/np.prod(local_optima))))
    return rows


def analyze():
    mean, second, linear, quadratic, constant, blocks = faithful_reference()
    uniform = np.full(len(mean), 0.5)
    erandom = energy(uniform, np.outer(uniform, uniform), linear, quadratic, constant)
    eideal = energy(mean, second, linear, quadratic, constant)
    eproduct = energy(mean, np.outer(mean, mean), linear, quadratic, constant)
    denominator = eproduct - eideal
    if abs(denominator) < 1e-10:
        raise ValueError("The ideal covariance reference is too small for this normalization.")

    def record(p01, p10):
        mu, pair = readout_moments(mean, second, p01, p10)
        e = energy(mu, pair, linear, quadratic, constant)
        ep = energy(mu, np.outer(mu, mu), linear, quadratic, constant)
        return dict(p01=float(p01), p10=float(p10), energy=e, product_energy=ep,
                    mean_effective_flip_probability=float(np.mean((1-mean)*p01+mean*p10)),
                    uniform_reference_reduction=1-e/erandom,
                    correlation_retention=(ep-e)/denominator)

    return dict(
        scope="Exact synthetic output-distribution controls; not a fitted hardware noise model.",
        grids=dict(symmetric_flip=FLIP_GRID.tolist(), fixed_transition_sum=TRANSITION_SUM,
                   p01_at_fixed_total=BIAS_GRID.tolist()),
        reference=dict(n_qubits=len(mean), block_sizes=list(map(len, blocks)),
                       represented_dimension=sum(2**len(b) for b in blocks),
                       energy_random=erandom, energy_ideal=eideal, energy_product=eproduct,
                       ideal_correlation_energy=denominator,
                       ideal_reduction=1-eideal/erandom,
                       product_reduction=1-eproduct/erandom),
        symmetric=[record(p, p) for p in FLIP_GRID],
        biased=[record(a, TRANSITION_SUM-a) for a in BIAS_GRID],
        dependence=dependence_control())


def write_artifacts(result, output):
    (output / "diagnostic_controls.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7))
    for ax, key, xlabel, title in zip(
        axes, ("symmetric", "biased"),
        (r"symmetric flip probability $p$", r"$p_{01}$ at $p_{01}+p_{10}=0.2$"),
        ("(a) Independent symmetric output flips", "(b) Bias at fixed covariance attenuation")):
        rows = result[key]
        ax.plot([r["p01"] for r in rows], [r["uniform_reference_reduction"] for r in rows],
                "o-", ms=3, color="#4477AA", label="uniform-reference energy reduction")
        ax.plot([r["p01"] for r in rows], [r["correlation_retention"] for r in rows],
                "s--", ms=3, color="#228833", label=r"covariance ratio $C$")
        ax.set(xlabel=xlabel, ylabel="dimensionless diagnostic", title=title)
        ax.grid(axis="y", color="#dddddd", lw=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.14, 1, 1), w_pad=1.5)
    fig.savefig(output / "fig_diagnostic_controls.pdf", bbox_inches="tight")
    plt.close(fig)
    biased = result["biased"]
    macros = {
        "ControlBiasReductionMin": f"{100*min(r['uniform_reference_reduction'] for r in biased):.1f}",
        "ControlBiasReductionMax": f"{100*max(r['uniform_reference_reduction'] for r in biased):.1f}",
        "ControlProductReduction": f"{100*result['reference']['product_reduction']:.1f}",
    }
    (output / "diagnostic_numbers.tex").write_text("\n".join(
        rf"\newcommand{{\{key}}}{{{value}}}" for key, value in macros.items())+"\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=HERE/"generated")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = analyze()
    write_artifacts(result, args.out_dir)
    print(json.dumps(result["reference"], indent=2))


if __name__ == "__main__":
    main()
