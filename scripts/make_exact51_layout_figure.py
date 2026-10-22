#!/usr/bin/env python3
"""Fig. 4: the faithful 51-qubit circuit on the chip, and what routing does.

(a, b) The two executions on the Kingston coupling map, qubits coloured by logical block at
measurement (circles: base qubits; squares: slack qubits).  A two-qubit gate is drawn in block
colour when it lies in the backward light cone of that block only, and dashed black when it
lies in the light cones of two or more logical blocks (routing SWAPs followed exactly).
(c) Correlation z = r sqrt(S) between the per-shot energies of every pair of logical blocks,
grouped by overlapping light cones / disjoint cones but adjacent at readout / distant, for both
runs, next to the same statistic for the 24-qubit run (all cones disjoint), with the
Bonferroni line.

Inputs: results/exact51_physical_layout.json, results/ibm_circuits/exact51_*.ops.json, results/exact51_cross_block.json,
results/rung1_physical_layout.json (coupling map), data/frozen_rung1.json.
Writes figures/fig_exact51_routing.pdf.  Offline.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from _paths import DATA_DIR, FIGURES_DIR, RESULTS_DIR
from build_exact_circuit import blocks_from_edges, exact_qubo
from calc_exact51_cross_block import light_cones
from make_block_figure import GREY, PALETTE, heavy_hex_positions


def main() -> None:
    d = json.load(open(DATA_DIR / "frozen_rung1.json"))
    A = np.asarray(d["A"], int)
    lin, quad, n, n_b, _, _, _ = exact_qubo(A, 10.0)
    blocks = sorted([sorted(int(q) for q in b) for b in blocks_from_edges(n, quad.keys())],
                    key=lambda b: (-len(b), b))
    col, lab, k = {}, {}, 0
    for i, b in enumerate(blocks):
        c = PALETTE[k] if len(b) > 1 else GREY
        k += len(b) > 1
        for q in b:
            col[q], lab[q] = c, i
    lay = json.load(open(RESULTS_DIR / "exact51_physical_layout.json"))
    cb = json.load(open(RESULTS_DIR / "exact51_cross_block.json"))
    coupling = [tuple(e) for e in json.load(open(RESULTS_DIR / "rung1_physical_layout.json"))["coupling_map"]]
    pos = heavy_hex_positions(coupling)

    plt.rcParams.update({"font.family": "serif", "font.size": 7,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 600, "savefig.bbox": "tight"})
    fig = plt.figure(figsize=(7.1, 2.65))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.3], wspace=0.32,
                          left=0.04, right=0.99, bottom=0.25, top=0.89)
    axes = []
    for k_, (tag, title) in enumerate((("defaults", "(a) sampler defaults"),
                                       ("matched", "(b) XY4 and twirling"))):
        ax = fig.add_subplot(gs[k_])
        p2l = {v: int(q) for q, v in lay[tag]["logical_to_physical"].items()}
        for a, b in coupling:
            ax.plot(*zip(pos[a], pos[b]), color="#e3e3e3", lw=0.5, zorder=1)
        xs, ys = zip(*pos.values())
        ax.scatter(xs, ys, s=3, color="#cccccc", zorder=2)
        cones, ops = light_cones(RESULTS_DIR / "ibm_circuits" / f"exact51_{tag}.ops.json", blocks)
        owner = {}                                   # edge -> set of blocks whose cones hold it
        n_one = n_shared = 0
        for k_op, (name, qs, _) in enumerate(ops):
            if len(qs) != 2 or name == "measure":
                continue
            who = {i for i, c in enumerate(cones) if k_op in c}
            e = tuple(sorted(qs))
            owner.setdefault(e, [set(), 0])
            owner[e][0] |= who
            owner[e][1] += 1
            n_one += len(who) == 1
            n_shared += len(who) > 1
        for (a, b), (who, cnt) in owner.items():
            same = len(who) == 1
            c = col[blocks[next(iter(who))][0]] if same else "k"
            ax.plot(*zip(pos[a], pos[b]), color=c,
                    lw=0.6 + 0.2 * min(cnt, 10) if same else 1.0,
                    ls="-" if same else (0, (1.5, 1.0)),
                    solid_capstyle="round", zorder=3 if same else 4)
        for p, q in p2l.items():
            ax.scatter(*pos[p], s=13, marker="o" if q < n_b else "s", color=col[q],
                       edgecolors="k", linewidths=0.3, zorder=5)
        ax.set_aspect("equal")
        ax.set_anchor("N")
        ax.axis("off")
        ax.text(0.5, -0.02, f"{n_one} gates in one block's light cone,\n{n_shared} shared by two or three",
                transform=ax.transAxes, ha="center", va="top", fontsize=6)
        axes.append((ax, title))

    ax = fig.add_subplot(gs[2])
    xpos = {"cone": 1, "adjacent": 2, "distant": 3}
    ref = cb["rung1_reference"]["z"]
    rng = np.random.default_rng(0)
    ax.scatter(0 + rng.uniform(-0.12, 0.12, len(ref)), ref, s=5, marker="^", color="#4477AA",
               alpha=0.6, lw=0)
    for off, (tag, mk) in zip((-0.15, 0.15), (("defaults", "o"), ("matched", "s"))):
        for row in cb["runs"][tag]["pairs"]:
            x = xpos[row["cls"]] + off + (rng.uniform(-0.07, 0.07) if row["cls"] == "distant" else 0)
            c = "k" if row["cls"] == "cone" else "#999999"
            ax.scatter(x, row["z"], s=10 if row["cls"] == "cone" else 5, marker=mk,
                       color=c, alpha=1 if row["cls"] == "cone" else 0.6, lw=0)
    zb = cb["z_bonferroni_5pct"]
    for s_ in (1, -1):
        ax.axhline(s_ * zb, color="#CC6677", lw=0.6, ls=":")
    ax.axhline(0, color="#555555", lw=0.4)
    ax.text(3.45, zb + 0.15, "Bonferroni, 78 pairs", color="#CC6677", fontsize=5.5, ha="right")
    ax.axvline(0.5, color="#bbbbbb", lw=0.5)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["24q:\ndisjoint\ncones", "51q:\noverlapping\ncones", "51q:\nadjacent\nat readout",
                        "51q:\ndistant"], fontsize=5.8)
    ax.set_ylabel(r"$z=r\sqrt{S}$", fontsize=7)
    ax.set_xlim(-0.5, 3.5)
    panel_legend = ax.legend(handles=[Line2D([], [], marker="^", ls="", ms=3.5, color="#4477AA", label="24q joint run"),
                       Line2D([], [], marker="o", ls="", ms=3.5, color="k", label="51q defaults"),
                       Line2D([], [], marker="s", ls="", ms=3.5, color="k", label="51q XY4 + twirling")],
              loc="lower right", bbox_to_anchor=(1, 0.01), fontsize=5.8, frameon=False)
    ax.set_ylim(-7.0, 7.0)
    # Reserve space below the lower threshold for all three legend rows.
    for _ in range(8):
        fig.canvas.draw()
        threshold_y = ax.transData.transform((0, -zb))[1]
        if panel_legend.get_window_extent(fig.canvas.get_renderer()).y1 < threshold_y - 4:
            break
        ax.set_ylim(ax.get_ylim()[0] - 0.5, 7.0)
    else:
        raise RuntimeError("Panel legend overlaps the lower Bonferroni threshold.")
    axes.append((ax, "(c) correlation between logical blocks"))

    handles = [Line2D([], [], marker="o", ls="", ms=4, mfc="#bbbbbb", mec="k", mew=0.3, label="base qubit"),
               Line2D([], [], marker="s", ls="", ms=4, mfc="#bbbbbb", mec="k", mew=0.3, label="slack qubit"),
               Line2D([], [], color="k", lw=1.0, ls=(0, (1.2, 0.8)), label="gate in two or more blocks' light cones")]
    handles += [Line2D([], [], marker="o", ls="", ms=4.5, mfc=PALETTE[i], mec="k", mew=0.3,
                       label=f"block of {len(b)}")
                for i, b in enumerate([b for b in blocks if len(b) > 1])]
    handles += [Line2D([], [], marker="o", ls="", ms=4.5, mfc=GREY, mec="k", mew=0.3,
                       label="7 single-base blocks")]
    fig.canvas.draw()
    # Place the shared legend below every panel's labels, including multiline ticks.
    renderer = fig.canvas.get_renderer()
    legend_top = min(a.get_tightbbox(renderer).y0 for a, _ in axes) / fig.bbox.height - 0.04
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=5.8, frameon=False,
               bbox_to_anchor=(0.5, legend_top), columnspacing=1.0, handletextpad=0.4)
    top = max(a.get_position().y1 for a, _ in axes) + 0.015
    for a, t in axes:
        fig.text(a.get_position().x0, top, t, fontsize=7, ha="left", va="bottom")
    fig.savefig(FIGURES_DIR / "fig_exact51_routing.pdf")
    print(f"-> {FIGURES_DIR / 'fig_exact51_routing.pdf'}")


if __name__ == "__main__":
    main()
