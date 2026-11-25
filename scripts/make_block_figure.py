#!/usr/bin/env python3
"""Figure: the Rung-1 block structure, from the map to the chip to the counts.

(a) The 24 PEF bases and 23 waypoints, bases coloured by the component of G_ZZ they belong
    to (six multi-qubit blocks in colour, seven singletons in grey) and every base-waypoint
    link drawn in its block colour.  The closest pair of bases in different blocks is marked:
    blocks separate because they share no waypoint, not because they are more than 2R apart.
(b) IBM Kingston's heavy-hex coupling map with the 24 physical qubits of the executed p=1
    circuit in the same colours and its two-qubit gates drawn as edges (width ~ count).
    Inset: the logical circuit of the four-qubit block.
(c) Per-block retention P_hw(Opt_i)/P_ideal(Opt_i) in the joint 24-qubit execution and in
    the separate execution of the 13 block circuits, with binomial standard errors.

Inputs (all archived): data/frozen_rung1.json, results/rung1_physical_layout.json,
results/block_product_null.json, results/ibm_block_results/kingston_blocks_analysis.json.
Writes figures/fig_blocks.pdf.  Offline; no IBM access and no qiskit needed.
"""
from __future__ import annotations

import json
import math

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from scipy.optimize import Bounds, LinearConstraint, milp

from _paths import DATA_DIR, FIGURES_DIR, RESULTS_DIR
from calc_block_factorisation import blocks_of, load

# Paul Tol "bright" (colour-blind safe) for the six multi-qubit blocks; grey for singletons
PALETTE = ["#4477AA", "#228833", "#EE6677", "#AA3377", "#CCBB44", "#66CCEE"]
GREY = "#9A9A9A"


def haversine(la1, lo1, la2, lo2, r=6371.0):
    p1, p2 = np.radians(la1), np.radians(la2)
    dl = np.radians(lo2 - lo1)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def ilp_cover(A):
    res = milp(np.ones(A.shape[1]), integrality=np.ones(A.shape[1]), bounds=Bounds(0, 1),
               constraints=[LinearConstraint(A, 1, np.inf)])
    return np.flatnonzero(res.x > 0.5)


def heavy_hex_positions(coupling):
    """Kingston (Heron r2): rows of 16 qubits at 20r..20r+15, bridges at 20r+16..20r+19."""
    nbr = {}
    for a, b in coupling:
        nbr.setdefault(a, set()).add(b)
        nbr.setdefault(b, set()).add(a)
    pos = {}
    for q in nbr:
        r, i = divmod(q, 20)
        if i < 16:
            pos[q] = (i, -2 * r)
    for q in nbr:
        r, i = divmod(q, 20)
        if i >= 16:
            upper = min(nbr[q])
            pos[q] = (pos[upper][0], -2 * r - 1)
    return pos


def draw_block_circuit(ax, color, qubits, pairs):
    """Logical p=1 QAOA circuit of one block: H, R_z, one ZZ rotation per coupling, R_x, M."""
    m = len(qubits)
    xz = [2.6 + 0.62 * k for k in range(len(pairs))]
    xr = xz[-1] + 0.9
    ax.set_xlim(-0.1, xr + 1.4)
    ax.set_ylim(-0.7, m - 0.3)
    ax.axis("off")
    y = {q: m - 1 - k for k, q in enumerate(qubits)}
    for q in qubits:
        ax.plot([0.6, xr + 1.2], [y[q], y[q]], color="#444444", lw=0.6, zorder=0)
        ax.text(0.45, y[q], f"$q_{{{q}}}$", ha="right", va="center", fontsize=6)

    def box(x, yy, text, fc="white"):
        ax.add_patch(FancyBboxPatch((x - 0.27, yy - 0.25), 0.54, 0.5,
                                    boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=fc, ec=color, lw=0.7, zorder=2))
        ax.text(x, yy, text, ha="center", va="center", fontsize=5, zorder=3)

    for q in qubits:
        box(1.0, y[q], "H")
        box(1.75, y[q], r"$R_z$")
        box(xr, y[q], r"$R_x$")
        box(xr + 0.8, y[q], "M", fc="#f0f0f0")
    for x, (a, b) in zip(xz, pairs):
        ax.plot([x, x], sorted((y[a], y[b])), color=color, lw=1.0, zorder=1)
        ax.plot([x, x], [y[a], y[b]], "o", ms=2.8, color=color, zorder=3)
    ax.text((xz[0] + xz[-1]) / 2, -0.62, r"$e^{-i\gamma J_{bb'}Z_bZ_{b'}}$ on each coupling",
            ha="center", va="center", fontsize=5.5, color=color)


def main() -> None:
    d, A, h, J = load()
    n = d["n_b"]
    blocks = [sorted(int(q) for q in b) for b in blocks_of(n, J)]
    multi = sorted([b for b in blocks if len(b) > 1], key=lambda b: (-len(b), b))
    color = {}
    for k, b in enumerate(multi):
        for q in b:
            color[q] = PALETTE[k]
    for b in blocks:
        if len(b) == 1:
            color[b[0]] = GREY
    block_of = {q: tuple(b) for b in blocks for q in b}

    blat, blon = np.asarray(d["base_lats"]), np.asarray(d["base_lons"])
    tlat, tlon = np.asarray(d["task_lats"]), np.asarray(d["task_lons"])
    cover = ilp_cover(A)

    lay = json.load(open(RESULTS_DIR / "rung1_physical_layout.json"))
    l2p = {int(k): v for k, v in lay["circuits"]["rung1_qaoa_p1"]["logical_to_physical"].items()}
    p2l = {v: k for k, v in l2p.items()}
    gates = lay["circuits"]["rung1_qaoa_p1"]["two_qubit_edges"]
    coupling = [tuple(e) for e in lay["coupling_map"]]
    pos = heavy_hex_positions(coupling)

    joint = {tuple(sorted(b["qubits"])): b for b in
             next(c for c in json.load(open(RESULTS_DIR / "block_product_null.json"))["circuits"]
                  if c["circuit"] == "rung1_qaoa_p1")["blocks"]}
    sep_an = json.load(open(RESULTS_DIR / "ibm_block_results" / "kingston_blocks_analysis.json"))
    sep = {tuple(sorted(b["original_qubits"])): b for b in sep_an["blocks"]}
    assert set(joint) == set(sep) == {tuple(b) for b in blocks}

    hw_sep = {b["tag"]: b for b in sep_an["hardware_job"]["blocks"]}
    sep_phys = []   # (logical block, physical qubits, two-qubit edges) of the separate run
    for b in sep_an["blocks"]:
        hb = hw_sep[b["tag"]]
        sep_phys.append((tuple(sorted(b["original_qubits"])), hb["measured_physical_qubits"],
                         hb["two_qubit_edges"]))

    plt.rcParams.update({"font.family": "serif", "font.size": 7,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 600, "savefig.bbox": "tight"})
    fig = plt.figure(figsize=(7.1, 5.4))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.66], hspace=0.62)
    top_gs = outer[0].subgridspec(1, 3, width_ratios=[0.95, 1.0, 1.0], wspace=0.12)
    bot_gs = outer[1].subgridspec(1, 2, width_ratios=[0.85, 2.15], wspace=0.22)

    # ---- (a) the map ---------------------------------------------------------------------
    ax = fig.add_subplot(top_gs[0])
    ax.set_facecolor("#f1f1f1")   # outside the Legal Amazon: neighbouring countries
    outline = gpd.read_file(DATA_DIR / "legal_amazon_outline_2024.geojson")
    outline.plot(ax=ax, color="#eef5ee", edgecolor="#555555", lw=0.6, zorder=1)
    from shapely.geometry import Point
    poly = outline.geometry.iloc[0]
    n_out = sum(not poly.contains(Point(lo, la)) for la, lo in zip(tlat, tlon))
    for t in range(A.shape[0]):
        bs = np.flatnonzero(A[t])
        for b in bs:
            ax.plot([blon[b], tlon[t]], [blat[b], tlat[t]], color=color[b], lw=0.5,
                    alpha=0.8, zorder=2)
        ax.plot(tlon[t], tlat[t], "o", ms=1.8, color=color[bs[0]], mec="none", zorder=3)
    for b in range(n):
        ax.plot(blon[b], blat[b], "s", ms=3.6, color=color[b], mec="k", mew=0.3, zorder=4)
    ax.scatter(blon[cover], blat[cover], s=38, facecolors="none", edgecolors="k", lw=0.5,
               zorder=5)
    pairs = [(haversine(blat[i], blon[i], blat[j], blon[j]), i, j)
             for i in range(n) for j in range(i + 1, n) if block_of[i] != block_of[j]]
    dmin, i, j = min(pairs)
    mid = ((blon[i] + blon[j]) / 2, (blat[i] + blat[j]) / 2)
    ax.annotate(f"{dmin:.0f} km apart,\ndifferent blocks", xy=mid,
                xytext=(mid[0] + 3.2, mid[1] - 2.6), fontsize=5.5, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.5, color="k", shrinkB=2))
    ax.set_xlim(blon.min() - 2.5, blon.max() + 2.5)
    ax.set_ylim(blat.min() - 2.0, blat.max() + 2.0)
    ax.set_aspect("equal")
    ax.set_anchor("N")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="s", ls="", ms=4, mfc="#bbbbbb", mec="k", mew=0.3,
               label="candidate site (PEF position)"),
        Line2D([], [], marker="o", ls="", ms=2.5, mfc="#bbbbbb", mec="none",
               label="synthetic target"),
        Line2D([], [], color="#888888", lw=0.8, label=r"link, distance $\leq R$"),
        Line2D([], [], marker="o", ls="", ms=6, mfc="none", mec="k", mew=0.5,
               label=r"ILP cover ($k_{\rm opt}=10$)"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.02, -0.01), fontsize=5.8,
              frameon=False, ncol=1, handlelength=1.4, handletextpad=0.5, labelspacing=0.35)
    ax_a = ax

    def chip_background(ax):
        for a_, b_ in coupling:
            (x1, y1), (x2, y2) = pos[a_], pos[b_]
            ax.plot([x1, x2], [y1, y2], color="#dddddd", lw=0.5, zorder=1)
        xs, ys = zip(*pos.values())
        ax.scatter(xs, ys, s=3, color="#cccccc", zorder=2)
        ax.set_aspect("equal")
        ax.set_anchor("N")
        ax.axis("off")

    # ---- (b) the joint 24-qubit execution ------------------------------------------------
    ax = fig.add_subplot(top_gs[1])
    chip_background(ax)
    for g in gates:
        a_, b_ = g["edge"]
        (x1, y1), (x2, y2) = pos[a_], pos[b_]
        ax.plot([x1, x2], [y1, y2], color=color[p2l[a_]], lw=0.6 + 0.35 * g["count"],
                solid_capstyle="round", zorder=2)
    for q, lq in p2l.items():
        ax.scatter(*pos[q], s=18, color=color[lq], edgecolors="k", linewidths=0.3, zorder=3)
    ax.text(0.5, -0.03, "one circuit: 24 qubits,\n49 two-qubit gates", transform=ax.transAxes,
            ha="center", va="top", fontsize=5.8)
    ax_b = ax

    # ---- (c) the 13 blocks executed separately -------------------------------------------
    ax = fig.add_subplot(top_gs[2])
    chip_background(ax)
    from matplotlib.patches import Wedge
    users, edge_count, edge_cols = {}, {}, {}
    for blk_, qs, edges in sep_phys:
        for q in qs:
            users.setdefault(q, [])
            if color[blk_[0]] not in users[q]:
                users[q].append(color[blk_[0]])
        for e in edges:
            key = tuple(sorted(e["edge"]))
            edge_count[key] = edge_count.get(key, 0) + e["count"]
            edge_cols.setdefault(key, set()).add(color[blk_[0]])
    for (a_, b_), cnt in edge_count.items():
        (x1, y1), (x2, y2) = pos[a_], pos[b_]
        cs = edge_cols[(a_, b_)]
        ax.plot([x1, x2], [y1, y2], color=next(iter(cs)) if len(cs) == 1 else "#555555",
                lw=0.6 + 0.35 * min(cnt, 7), solid_capstyle="round", zorder=2)
    for q, cs in users.items():
        x0, y0 = pos[q]
        r = 0.34
        for k, c in enumerate(cs):
            ax.add_patch(Wedge((x0, y0), r, 90 + 360 * k / len(cs), 90 + 360 * (k + 1) / len(cs),
                               fc=c, ec="none", zorder=3))
        ax.add_patch(plt.Circle((x0, y0), r, fc="none", ec="k", lw=0.3, zorder=4))
    shared = sorted(q for q, cs in users.items() if len(cs) > 1)
    four = sorted(q for q, cs in users.items() if cs == [PALETTE[0]])
    ax.annotate("12 blocks,\nrun in turn", xy=pos[shared[-1]], xytext=(pos[shared[-1]][0] + 2.3,
                pos[shared[-1]][1] + 1.6), fontsize=5.5, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.5, color="k", shrinkB=3))
    ax.annotate("four-qubit\nblock", xy=pos[four[-1]], xytext=(pos[four[-1]][0] + 2.3,
                pos[four[-1]][1] - 1.2), fontsize=5.5, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.5, color="k", shrinkB=3))
    n_sep_2q = sum(edge_count.values())
    ax.text(0.5, -0.03, f"13 circuits in turn: {len(users)} qubits reused,\n{n_sep_2q} two-qubit gates",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.8)
    ax_c = ax

    # ---- (d) logical circuit of the four-qubit block -------------------------------------
    ins = fig.add_subplot(bot_gs[0])
    blk = multi[0]
    draw_block_circuit(ins, PALETTE[0], blk, [e for e in sorted(J) if set(e) <= set(blk)])
    ins.set_anchor("N")
    ax_d = ins

    # ---- (e) retention per block ---------------------------------------------------------
    ax = fig.add_subplot(bot_gs[1])
    ordered = multi + sorted([b for b in blocks if len(b) == 1])
    x = np.arange(len(ordered))
    wj, ws = [], []
    ej, es = [], []
    for b in ordered:
        bj, bs_ = joint[tuple(b)], sep[tuple(b)]
        pj, pid = bj["p_hw"], bj["p_ideal"]
        wj.append(pj / pid)
        ej.append(math.sqrt(pj * (1 - pj) / 1e5) / pid)
        ps, pid2 = bs_["p_opt"], bs_["p_opt_ideal"]
        ws.append(ps / pid2)
        es.append(math.sqrt(ps * (1 - ps) / bs_["shots"]) / pid2)
    cols = [color[b[0]] for b in ordered]
    batch = {tuple(r["qubits"]): r for r in
             json.load(open(RESULTS_DIR / "joint_separate_batch.json"))["blocks"]}
    ax.axhline(1.0, color="#555555", lw=0.6, ls=":")
    for k, b in enumerate(ordered):
        rb = batch[tuple(b)]
        ebj = math.sqrt(rb["p_joint"] * (1 - rb["p_joint"]) / 1e5) / rb["p_ideal"]
        ebs = math.sqrt(rb["p_separate"] * (1 - rb["p_separate"]) / rb["shots_separate"]) / rb["p_ideal"]
        ax.errorbar(x[k] - 0.27, wj[k], yerr=ej[k], fmt="o", ms=3.6, color=cols[k],
                    mec="k", mew=0.3, elinewidth=0.6, capsize=0, zorder=3)
        ax.errorbar(x[k] - 0.09, ws[k], yerr=es[k], fmt="o", ms=3.6, mfc="white",
                    mec=cols[k], mew=1.0, color=cols[k], elinewidth=0.6, capsize=0, zorder=3)
        ax.errorbar(x[k] + 0.09, rb["retention_joint"], yerr=ebj, fmt="s", ms=3.4, color=cols[k],
                    mec="k", mew=0.3, elinewidth=0.6, capsize=0, zorder=3)
        ax.errorbar(x[k] + 0.27, rb["retention_separate"], yerr=ebs, fmt="s", ms=3.4, mfc="white",
                    mec=cols[k], mew=1.0, color=cols[k], elinewidth=0.6, capsize=0, zorder=3)
    ax.plot([], [], "o", ms=4, color="#555555", mec="k", mew=0.3,
            label="joint run, panel (b)")
    ax.plot([], [], "o", ms=4, mfc="white", mec="#555555", mew=1.0,
            label="separately compiled blocks, panel (c)")
    ax.plot([], [], "s", ms=4, color="#555555", mec="k", mew=0.3,
            label="same batch, joint")
    ax.plot([], [], "s", ms=4, mfc="white", mec="#555555", mew=1.0,
            label="same batch, isolated slices")
    ax.set_xticks(x)
    ax.set_xticklabels([str(len(b)) for b in ordered], fontsize=6)
    ax.set_xlabel("block size (qubits)", fontsize=7)
    ax.set_ylabel(r"$P_{\rm hw}(\mathrm{Opt}_i)\,/\,P_{\rm ideal}(\mathrm{Opt}_i)$", fontsize=7)
    ax.set_ylim(0.66, 1.04)
    ax.set_xlim(-0.6, len(ordered) - 0.4)
    pj_prod, ps_prod = float(np.prod(wj)), float(np.prod(ws))
    bt = json.load(open(RESULTS_DIR / "joint_separate_batch.json"))
    ax.legend(loc="lower right", fontsize=5.8, frameon=False, handlelength=1.0,
              title=(f"product over blocks: {pj_prod:.3f} / {ps_prod:.3f} (circles), "
                     f"{bt['within_block_retention_joint']:.3f} / "
                     f"{bt['within_block_retention_separate']:.3f} (squares)"),
              title_fontsize=5.8)
    ax_e = ax

    fig.canvas.draw()
    top = max(a.get_position().y1 for a in (ax_a, ax_b, ax_c)) + 0.012
    for a, lab in ((ax_a, "(a) sites, targets and blocks"),
                   (ax_b, "(b) Kingston: 24 qubits in one run"),
                   (ax_c, "(c) Kingston: 13 blocks run separately")):
        fig.text(a.get_position().x0, top, lab, fontsize=7, ha="left", va="bottom")
    top2 = max(a.get_position().y1 for a in (ax_d, ax_e)) + 0.012
    for a, lab in ((ax_d, "(d) four-qubit block, logical circuit"),
                   (ax_e, "(e) retention per block")):
        fig.text(a.get_position().x0, top2, lab, fontsize=7, ha="left", va="bottom")

    fig.savefig(FIGURES_DIR / "fig_blocks.pdf")
    print(f"waypoints outside the IBGE outline: {n_out}/{len(tlat)}")
    print(f"closest inter-block pair {dmin:.1f} km (bases {i}, {j}); "
          f"retention products joint {pj_prod:.3f}, separate {ps_prod:.3f}")
    print(f"-> {FIGURES_DIR / 'fig_blocks.pdf'}")


if __name__ == "__main__":
    main()
