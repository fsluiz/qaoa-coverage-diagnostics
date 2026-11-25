"""Exploratory sensitivity of the descriptive three-job variance subtraction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from combine_epochs import ENDPOINTS

HERE = Path(__file__).resolve().parent
# Fixed before the sensitivity calculations; these are hypothetical multipliers.
INFLATION_GRID = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0)


def analyze():
    epochs = [json.loads((HERE/f"results/e0{i}_analysis.json").read_text()) for i in range(1, 4)]
    result = {}
    for key, (variance_key, log_scale) in ENDPOINTS.items():
        values = np.array([e["epoch_summary"][key]["estimate"] for e in epochs])
        y = np.log(values) if log_scale else values
        variance = [e["epoch_summary"]["variance_description"][variance_key] for e in epochs]
        rounds = np.array([v["observed_round_variance"] for v in variance])
        shot = np.array([v["mean_conditional_shot_variance"] for v in variance])

        def summary(keep, inflation=1.0):
            yy = y[keep]
            observed = float(yy.var(ddof=1))
            working = float(np.mean(np.maximum(rounds[keep], inflation*shot[keep])/4))
            return dict(estimate=float(np.exp(yy.mean()) if log_scale else yy.mean()),
                        observed_variance=observed, mean_working_variance=working,
                        excess=max(0.0, observed-working), ratio=observed/working)

        all_jobs = np.arange(3)
        base = summary(all_jobs)
        threshold = None
        if base["excess"] > 0 and summary(all_jobs, INFLATION_GRID[-1])["excess"] == 0:
            target = base["observed_variance"]
            threshold = float(brentq(
                lambda d: np.mean(np.maximum(rounds, d*shot)/4)-target,
                1, INFLATION_GRID[-1]))
        result[key] = dict(
            scale="log" if log_scale else "identity", baseline=base,
            leave_one_out=[dict(omitted=f"E0{i+1}", **summary(np.delete(all_jobs, i)))
                           for i in range(3)],
            inflation=[dict(factor=d, **summary(all_jobs, d)) for d in INFLATION_GRID],
            zero_excess_threshold_within_grid=threshold)
    return dict(scope="Exploratory model sensitivity, not estimation of shot dependence or population variance.",
                inflation_grid=INFLATION_GRID, endpoints=result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=HERE/"generated")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = analyze()
    (args.out_dir/"campaign_sensitivity.json").write_text(json.dumps(result, indent=2)+"\n")
    lines = [r"\begin{table}[t]",
             r"\caption{Exploratory omission of one job. $s^2/\bar v$ compares observed dispersion with the mean working estimation variance, on the log scale for $R$ and identity scale for $C$. Each row has two jobs and is a sensitivity description, not population inference.}",
             r"\label{tab:sm_sensitivity}", r"\begin{ruledtabular}",
             r"\begin{tabular}{lrrrr}",
             r"Omitted & Mean $R$ & $s_R^2/\bar v_R$ & Mean $C$ & $s_C^2/\bar v_C$ \\",
             r"\hline"]
    rr, cc = [result["endpoints"][k] for k in ENDPOINTS]
    for r, c in zip(rr["leave_one_out"], cc["leave_one_out"]):
        lines.append(f"{r['omitted']} & {r['estimate']:.4f} & {r['ratio']:.2f} & "
                     f"{c['estimate']:.4f} & {c['ratio']:.3f}"+r" \\")
    lines += [r"\end{tabular}", r"\end{ruledtabular}", r"\end{table}"]
    (args.out_dir/"sensitivity_table.tex").write_text("\n".join(lines)+"\n")
    threshold = rr["zero_excess_threshold_within_grid"]
    (args.out_dir/"sensitivity_numbers.tex").write_text(
        rf"\newcommand{{\RInflationThreshold}}{{{threshold:.1f}}}"+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
