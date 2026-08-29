"""Generate the corrected deterministic 3x3 Rydberg result figure."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _paths import FIGURES_DIR, RESULTS_DIR


def main() -> None:
    with (RESULTS_DIR / "rydberg_3x3_exact.json").open(encoding="utf-8") as stream:
        result = json.load(stream)

    alpha_scan = result["alpha_scan_at_T_2p6_us"]
    time_scan = result["time_scan_at_alpha_0p5"]
    baseline = 100.0 * result["uniform_random_baseline"]
    alpha = np.array([row["alpha"] for row in alpha_scan])
    p_alpha = 100.0 * np.array([row["P_opt"] for row in alpha_scan])
    duration = np.array([row["T_us"] for row in time_scan])
    p_time = 100.0 * np.array([row["P_opt"] for row in time_scan])

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
        }
    )
    blue = "#4477AA"
    red = "#CC6677"
    grey = "#777777"
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))

    axes[0].plot(alpha, p_alpha, "-o", color=blue, ms=4.5)
    axes[0].axhline(baseline, color=grey, ls="--", lw=1.0, label="uniform baseline")
    selected = int(np.flatnonzero(np.isclose(alpha, 0.5))[0])
    axes[0].scatter(alpha[selected], p_alpha[selected], s=50, facecolors="none", edgecolors=red)
    axes[0].annotate(
        "$P_\\mathrm{opt}=14.80\\%$\n$\\bar{k}=1.765$; $7.58\\times$",
        (alpha[selected], p_alpha[selected]),
        xytext=(0.515, 11.5),
        fontsize=8,
        arrowprops={"arrowstyle": "->", "color": grey, "lw": 0.7},
    )
    axes[0].set(xlabel="local-detuning scale $\\alpha$", ylabel="$P(\\mathrm{opt})$ (%)")
    axes[0].set_title(
        "(a) Exact parameter scan at $T=2.6~\\mu\\mathrm{s}$", loc="left"
    )
    axes[0].set_xticks(alpha)
    axes[0].set_ylim(0, 18)
    axes[0].legend(loc="upper left")

    axes[1].plot(duration, p_time, "-o", color=blue, ms=4.5)
    axes[1].axhline(baseline, color=grey, ls="--", lw=1.0)
    optimum = int(np.argmax(p_time))
    axes[1].scatter(
        duration[optimum], p_time[optimum], s=50, facecolors="none", edgecolors=red
    )
    axes[1].annotate(
        "best scanned $T=3.0~\\mu\\mathrm{s}$\n$P_\\mathrm{opt}=36.33\\%$",
        (duration[optimum], p_time[optimum]),
        xytext=(3.15, 29.0),
        fontsize=8,
        arrowprops={"arrowstyle": "->", "color": grey, "lw": 0.7},
    )
    axes[1].set(
        xlabel="sweep duration $T$ ($\\mu\\mathrm{s}$)",
        ylabel="$P(\\mathrm{opt})$ (%)",
    )
    axes[1].set_title("(b) Exact time scan at $\\alpha=0.5$", loc="left")
    axes[1].set_xticks(duration)
    axes[1].tick_params(axis="x", labelrotation=45)
    axes[1].set_ylim(0, 40)

    for axis in axes:
        axis.tick_params(direction="in", width=0.6)
        axis.spines["left"].set_color(grey)
        axis.spines["bottom"].set_color(grey)

    figure.tight_layout(w_pad=2.0)
    for extension in ("pdf", "png"):
        figure.savefig(FIGURES_DIR / f"fig4_rydberg_results.{extension}")
    plt.close(figure)
    print("Wrote corrected figures/fig4_rydberg_results.{pdf,png}")


if __name__ == "__main__":
    main()
