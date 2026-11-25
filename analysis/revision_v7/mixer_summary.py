"""Reconstruct mixer-family metadata and compare archived unit-weight endpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT/"scripts"))
from benchmark_qwoa_qaoa import build_instance, cooperation_edges


def analyze():
    archive = json.loads((PROJECT/"results/qwoa_20seeds.json").read_text())
    rows = archive["rows"]
    metadata, effects = [], []
    for seed in sorted({r["seed"] for r in rows}):
        local = [r for r in rows if r["seed"] == seed]
        instance = build_instance(min_task_degree=3, rng_seed=seed)
        matrix = instance["A_q"]
        degree = matrix.sum(axis=0)
        edges = cooperation_edges(matrix)
        assert all(r["range_km"] == instance["R_chosen"] and r["n_support_edges"] == len(edges)
                   for r in local)
        metadata.append(dict(seed=seed, base_indices=instance["q_idx"],
                             n_targets=int(matrix.shape[1]), radius_km=instance["R_chosen"],
                             mean_target_degree=float(degree.mean()), min_target_degree=int(degree.min()),
                             support_edges=len(edges), added_bridges=sum(v == 0.1 for v in edges.values()),
                             alternative_radius_rule=bool(degree.mean() > 6)))
        delta = np.mean([r["searches"]["support"]["expected_energy"]
                         - r["searches"]["complete"]["expected_energy"] for r in local])
        log_ratio = np.mean([np.log(r["searches"]["support"]["metrics"]["p_opt"]
                                    / r["searches"]["complete"]["metrics"]["p_opt"]) for r in local])
        effects.append(dict(seed=seed, mean_energy_difference=float(delta), mean_log_ratio=float(log_ratio)))
    de = np.array([r["mean_energy_difference"] for r in effects])
    lr = np.array([r["mean_log_ratio"] for r in effects])
    # Deterministic seed bootstrap and explicit zero handling for the signed-rank tests.
    rng = np.random.default_rng(20261021)
    indices = rng.integers(0, len(lr), size=(10000, len(lr)))
    ci = np.exp(np.percentile(np.median(lr[indices], axis=1), [2.5, 97.5]))
    energy_ci = np.percentile(np.median(de[indices], axis=1), [2.5, 97.5])
    def signed_rank(values):
        return float(wilcoxon(values, zero_method="wilcox", correction=False, method="approx").pvalue)
    return dict(scope="Exploratory topology-only contrast U versus K; no new optimizations.",
                metadata=metadata, seed_effects=effects,
                summary=dict(n_seeds=len(metadata),
                             distinct_base_sets=len({tuple(r["base_indices"]) for r in metadata}),
                             complete_supports=sum(r["support_edges"] == 66 for r in metadata),
                             alternative_rule_seeds=sum(r["alternative_radius_rule"] for r in metadata),
                             median_energy_difference=float(np.median(de)), energy_ci95=energy_ci.tolist(),
                             median_probability_ratio=float(np.exp(np.median(lr))), ratio_ci95=ci.tolist(),
                             energy_p=signed_rank(de), probability_p=signed_rank(lr),
                             exact_zero_energy_seeds=int((de == 0).sum()), exact_zero_probability_seeds=int((lr == 0).sum())),
                inference=dict(unit="seed: mean over its three multiplicities",
                               tests="two-sided Wilcoxon, zero_method=wilcox, normal approximation, no continuity correction",
                               family="two exploratory endpoint tests; raw p values and Holm-adjusted values",
                               bootstrap_replicates=10000, bootstrap_seed=20261021,
                               interval="pointwise percentile interval for the median seed effect"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=HERE/"generated")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = analyze()
    summary = result["summary"]
    p = np.array([summary["energy_p"], summary["probability_p"]])
    order = np.argsort(p)
    adjusted = np.empty(2)
    adjusted[order] = np.maximum.accumulate(np.minimum(1, p[order]*np.array([2, 1])))
    summary["energy_p_holm"], summary["probability_p_holm"] = adjusted.tolist()
    (args.out_dir/"mixer_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    lo, hi = summary["ratio_ci95"]
    elo, ehi = summary["energy_ci95"]
    lines = [r"\begin{table}[t]",
             r"\caption{Exploratory topology-only contrast $U-K$ from the 20 archived seeds. Effects are seed-level medians; probability is $P_U/P_K$. Brackets are pointwise 95\% seed-bootstrap intervals (10,000 draws). Signed-rank tests discard exact zeros and use the normal approximation. Holm adjustment covers these two endpoints only.}",
             r"\label{tab:sm_topology}", r"\begin{ruledtabular}",
             r"\begin{tabular}{lrrr}", r"Metric & Effect [interval] & $p$ & $p_{\rm Holm}$ \\", r"\hline",
             f"Energy & ${summary['median_energy_difference']:.3f}$ $[{elo:.3f},{ehi:.3f}]$ & {p[0]:.2f} & {adjusted[0]:.2f}"+r" \\",
             f"Probability & ${summary['median_probability_ratio']:.3f}$ $[{lo:.3f},{hi:.3f}]$ & {p[1]:.2f} & {adjusted[1]:.2f}"+r" \\",
             r"\end{tabular}", r"\end{ruledtabular}", r"\end{table}"]
    (args.out_dir/"topology_table.tex").write_text("\n".join(lines)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
