#!/usr/bin/env python3
"""Central figure: circuit cost and exact classical cost rise together.

(a) The exact cost of solving each instance by enumerating the connected components of the
    interaction graph, against the two-qubit count of the circuit that would run it, over the
    600-instance ensemble.  Colour is register size, marker shape is R/delta, so the
    radius-driven trend inside each register size is visible rather than hidden.
(b) Why the block cost is the right observable: at fixed geometry it still carries an
    instance-level signal, where the forced fraction phi does not.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter

from _paths import FIGURES_DIR, RESULTS_DIR

blocks = {(r["n_b"], r["R_over_delta"], r["seed"]): r
          for r in json.load(open(RESULTS_DIR / "block_decomposition.json"))["ensemble"]}
phi = {(r["n_b"], r["R_over_delta"], r["seed"]): r
       for r in json.load(open(RESULTS_DIR / "ensemble_squeeze.json"))["rows"]}
keys = sorted(set(blocks) & set(phi))

plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "savefig.dpi": 600, "savefig.bbox": "tight"})
grey = "#777777"
cols = {20: "#88CCEE", 40: "#44AA99", 80: "#CC6677", 160: "#332288"}
marks = {0.90: "o", 1.35: "s", 1.80: "^", 2.40: "D", 3.00: "v", 3.90: "P"}
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))

for nb, c in cols.items():
    for rd, mk in marks.items():
        sel = [blocks[k] for k in keys if k[0] == nb and abs(k[1] - rd) < 0.02]
        if sel:
            ax[0].scatter([r["n_ZZ"] for r in sel], [r["log2_enum"] for r in sel],
                          s=11, alpha=0.6, color=c, marker=mk, edgecolor="none")
ax[0].set_xscale("log")
ax[0].set(xlabel=r"$n_{ZZ}$ (circuit cost: two-qubit interactions)",
          ylabel=r"$\log_2$ (exact classical cost)")
ax[0].set_title("(a) Both costs rise together", loc="left")
ax[0].legend(handles=[Patch(color=c, label=rf"$n_B={nb}$") for nb, c in cols.items()]
             + [Line2D([], [], ls="", marker=mk, color=grey, label=rf"$R/\delta={rd}$")
                for rd, mk in marks.items()],
             loc="upper left", fontsize=6, ncol=2, handletextpad=0.2, columnspacing=0.8)

nbs = sorted(cols)
rng = np.random.default_rng(0)


def within(metric):
    out = []
    for nb in nbs:
        per = []
        for rd in marks:
            sel = [k for k in keys if k[0] == nb and abs(k[1] - rd) < 0.02]
            if len(sel) > 5:
                x = np.log10([max(blocks[k]["n_ZZ"], 1) for k in sel])
                y = [metric(k) for k in sel]
                c = st.spearmanr(x, y)[0]
                if not np.isnan(c):
                    per.append(c)
        out.append(per)
    return out


series = [("exact classical cost", lambda k: blocks[k]["log2_enum"], "#332288", "-o"),
          (r"forced fraction $\phi$", lambda k: phi[k]["phi"], "#CC6677", "--s")]
for lab, f, c, fmt in series:
    per = within(f)
    mu = [np.mean(p) for p in per]
    lo = [np.percentile([np.mean(rng.choice(p, len(p))) for _ in range(2000)], 2.5) for p in per]
    hi = [np.percentile([np.mean(rng.choice(p, len(p))) for _ in range(2000)], 97.5) for p in per]
    ax[1].errorbar(nbs, mu, yerr=[np.array(mu) - lo, np.array(hi) - np.array(mu)],
                   fmt=fmt, color=c, ms=4.5, capsize=3, lw=1.2, elinewidth=0.9, label=lab)
ax[1].axhline(0, color=grey, lw=0.8)
ax[1].set_xscale("log"); ax[1].set_xticks(nbs); ax[1].set_xticklabels(nbs)
ax[1].xaxis.set_minor_formatter(NullFormatter())
ax[1].set(xlabel=r"register size $n_B$",
          ylabel=r"within-cell Spearman vs $\log n_{ZZ}$")
ax[1].set_title("(b) At fixed geometry, only one survives", loc="left")
ax[1].legend(loc="center left", fontsize=7)

for a in ax:
    a.tick_params(direction="in", width=0.6)
    a.spines["left"].set_color(grey); a.spines["bottom"].set_color(grey)
fig.tight_layout(w_pad=2.0)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGURES_DIR / "fig5_squeeze.pdf")
print("wrote figures/fig5_squeeze.pdf")
