#!/usr/bin/env python3
"""Regenerate fig2: where the regular-grid dominating-set ILP fails, and why.

The shipped fig2 asserted "discrete degree plateaus (source of LP symmetry)", a mechanism
the manuscript now retracts.  This version plots what the data support: solve time against
R/delta with the timeouts marked, and the absolute LP-integer gap that separates the ratios
the solver closes from the first window.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _paths import DATA_DIR, FIGURES_DIR, RESULTS_DIR

bench = json.load(open(DATA_DIR / "ilp_ratio_benchmark.json"))
gap = {r["ratio"]: r for r in json.load(open(RESULTS_DIR / "ilp_symmetry_corrected.json"))["lp_gap_sweep"]}

plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "savefig.dpi": 600, "savefig.bbox": "tight"})
blue, red, grey = "#4477AA", "#CC6677", "#777777"
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))

ratios = [r["ratio"] for r in bench]
times = [r["t_ilp_s"] for r in bench]
to = [r["status"] == "TIMEOUT" for r in bench]
integral = [gap[r]["abs_gap"] == 0.0 for r in ratios]

ax[0].plot(ratios, times, "-", color=grey, lw=0.8, zorder=1)
for x, y, t, i in zip(ratios, times, to, integral):
    ax[0].scatter(x, y, s=42, zorder=3, marker="v" if t else ("o" if i else "s"),
                  color=red if t else (blue if i else "white"),
                  edgecolor=red if t else blue, linewidth=1.1)
ax[0].axhline(60, color=red, ls=":", lw=1.0)
ax[0].text(5.9, 68, "60 s limit", ha="right", fontsize=7, color=red)
ax[0].set_yscale("log")
ax[0].set(xlabel=r"$R/\delta$", ylabel="ILP solve time (s)")
ax[0].set_title("(a) Two graphs fail; the rest close at the root", loc="left")
ax[0].scatter([], [], marker="o", color=blue, edgecolor=blue, s=42, label="LP integral")
ax[0].scatter([], [], marker="s", color="white", edgecolor=blue, s=42, label="LP fractional")
ax[0].scatter([], [], marker="v", color=red, edgecolor=red, s=42, label="timeout")
ax[0].legend(loc="lower right", fontsize=7)

g = [gap[r]["abs_gap"] for r in ratios]
for x, y, t in zip(ratios, g, to):
    if y is None:
        continue
    ax[1].bar(x, y, width=0.16, color=red if t else blue,
              edgecolor=red if t else blue, alpha=0.9 if t else 0.55)
ax[1].axhline(2.59, color=grey, ls="--", lw=0.9)
ax[1].text(5.95, 2.9, "largest gap the solver closes", ha="right", fontsize=6.8, color=grey)
ax[1].annotate(r"$\gamma=141$ vs LP $130.83$", xy=(1.2, 10.17), xytext=(2.1, 8.4),
               fontsize=7, arrowprops={"arrowstyle": "->", "color": grey, "lw": 0.7})
ax[1].set(xlabel=r"$R/\delta$", ylabel=r"$k_{\mathrm{opt}}-\mathrm{LP}$ (units)")
ax[1].set_title("(b) Absolute LP--integer gap", loc="left")

for a in ax:
    a.tick_params(direction="in", width=0.6)
    a.spines["left"].set_color(grey); a.spines["bottom"].set_color(grey)
fig.tight_layout(w_pad=2.0)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGURES_DIR / "fig2_ilp_hardness.pdf")
print("wrote figures/fig2_ilp_hardness.pdf")
