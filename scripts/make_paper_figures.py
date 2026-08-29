#!/usr/bin/env python3
"""
Paper figures for the QWOA-vs-QAOA results.

Fig A (compare_strong_results.json, n=14 overshoot instance, m=3, k_opt=9):
  (a) P(opt) vs circuit depth p at k_opt for QAOA / QWOA-direct / QWOA-descent.
  (b) Feasibility funnel: QWOA P(feasible) vs cardinality k during the descent,
      against the random in-sector baseline.

Fig B (benchmark_qwoa_qaoa_results.json, n=12 instance):
  Best P(opt): QWOA vs QAOA at m=1 and m=2 -> the advantage grows with coverage.

Outputs PDF (600 dpi, vector) + PNG to figures/. Design: Tol palette, Tufte spines.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _paths import FIGURES_DIR, RESULTS_DIR

# ---- style -----------------------------------------------------------------
try:
    plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
    _ = plt.figure(); plt.close(_)            # trigger a usetex probe later
    USETEX = True
except Exception:
    USETEX = False
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 10,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8,
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.5, "lines.linewidth": 1.2, "savefig.dpi": 600,
    "savefig.bbox": "tight",
})

C = dict(qaoa="#EE6677", direct="#228833", descent="#4477AA", rand="#BBBBBB")


def apply_tufte(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.5); ax.spines[s].set_color("#888888")
    ax.tick_params(direction="in", length=3, width=0.5, colors="#555555")
    ax.grid(False)
    leg = ax.get_legend()
    if leg:
        leg.set_frame_on(False)


def lbl(s):
    return s if USETEX else s.replace("\\,", " ").replace("$", "")


# ============================================================================
# FIGURE A
# ============================================================================
def figure_A():
    R = json.load(open(RESULTS_DIR / "compare_strong_results.json"))
    k_opt = R["meta"]["k_opt"]
    qaoa = sorted(R["qaoa"], key=lambda r: r["p"])
    direct9 = sorted(R["qwoa_direct"][str(k_opt)]["p_sweep"], key=lambda r: r["p"])
    desc9 = sorted(R["qwoa_descent"][str(k_opt)]["p_sweep"], key=lambda r: r["p"])
    ps = [r["p"] for r in direct9]

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    # --- (a) P(opt) vs p ---
    FLOOR = 1e-7
    pa = np.array([r["p_opt"] for r in qaoa])
    pd_ = np.array([r["p_opt"] for r in direct9])
    pde = np.array([r["p_opt"] for r in desc9])
    axa.plot(ps, np.clip(pd_, FLOOR, 1), "-o", color=C["direct"], ms=4.5,
             label=lbl(r"QWOA-direct"), zorder=3)
    axa.plot(ps, np.clip(pde, FLOOR, 1), "--s", color=C["descent"], ms=4,
             label=lbl(r"QWOA-descent"), zorder=3)
    # QAOA: clip off-scale points to the floor and mark with a down-triangle
    pac = np.clip(pa, FLOOR, 1)
    axa.plot(ps, pac, ":", color=C["qaoa"], lw=1.0, zorder=2)
    for p, raw, cl in zip(ps, pa, pac):
        if raw < FLOOR:
            axa.plot(p, cl, marker="v", color=C["qaoa"], ms=5, mfc="white",
                     mec=C["qaoa"], mew=0.9, zorder=4)
        else:
            axa.plot(p, cl, marker="x", color=C["qaoa"], ms=5, mew=1.2, zorder=4)
    axa.plot([], [], "x", color=C["qaoa"], label=lbl(r"QAOA"))
    axa.set_yscale("log"); axa.set_ylim(FLOOR, 1.0)
    axa.set_xticks(ps)
    axa.set_xlabel(lbl(r"circuit depth $p$"))
    axa.set_ylabel(lbl(r"$P(\mathrm{optimum})$"))
    axa.set_title(lbl(r"\textbf{QWOA reaches the optimum; QAOA cannot}")
                  if USETEX else "QWOA reaches the optimum; QAOA cannot",
                  loc="left", fontsize=9.5)
    axa.annotate(lbl(r"QAOA selects all 14 bases"),
                 xy=(2.0, FLOOR*1.5), xytext=(1.6, 1.2e-6),
                 color=C["qaoa"], fontsize=7.2)
    axa.text(-0.16, 1.06, "(a)", transform=axa.transAxes, fontsize=11,
             fontweight="bold", va="top")
    apply_tufte(axa); axa.legend(loc="center right")

    # --- (b) feasibility funnel ---
    ks = [12, 11, 10, 9]
    def bestfeas(block, k): return max(r["p_feas"] for r in block[str(k)]["p_sweep"])
    direct_f = [bestfeas(R["qwoa_direct"], k) for k in ks]
    desc_f = [bestfeas(R["qwoa_descent"], k) for k in ks]
    rand_f = [R["qwoa_direct"][str(k)]["n_feas"] / R["qwoa_direct"][str(k)]["dim"] for k in ks]
    x = np.arange(len(ks))
    axb.plot(x, direct_f, "-o", color=C["direct"], ms=4.5, label=lbl(r"QWOA-direct"))
    axb.plot(x, desc_f, "--s", color=C["descent"], ms=4, label=lbl(r"QWOA-descent"))
    axb.plot(x, rand_f, "-^", color=C["rand"], ms=4, label=lbl(r"random in sector"))
    axb.set_yscale("log")
    axb.set_xticks(x); axb.set_xticklabels([str(k) for k in ks])
    axb.set_xlabel(lbl(r"cardinality $k$ (descending)"))
    axb.set_ylabel(lbl(r"$P(\mathrm{feasible})$"))
    axb.set_title(lbl(r"\textbf{QWOA stays feasible as it descends}")
                  if USETEX else "QWOA stays feasible as it descends",
                  loc="left", fontsize=9.5)
    # annotate greedy start and optimum
    axb.annotate(lbl(r"greedy stops"), xy=(0, direct_f[0]), xytext=(0.05, 0.35),
                 fontsize=7.2, color="#555555",
                 arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6))
    axb.annotate(lbl(r"optimum $k^\star=9$"), xy=(3, direct_f[3]),
                 xytext=(1.65, direct_f[3]*1.4), fontsize=7.2, color="#555555",
                 arrowprops=dict(arrowstyle="->", color="#999999", lw=0.6))
    axb.text(-0.16, 1.06, "(b)", transform=axb.transAxes, fontsize=11,
             fontweight="bold", va="top")
    apply_tufte(axb); axb.legend(loc="lower left")

    fig.tight_layout(w_pad=2.0)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"fig_qwoa_vs_qaoa.{ext}")
    plt.close(fig)
    print(f"wrote {FIGURES_DIR}/fig_qwoa_vs_qaoa.pdf (+ .png)")


# ============================================================================
# FIGURE B  — advantage grows with m
# ============================================================================
def figure_B():
    R = json.load(open(RESULTS_DIR / "benchmark_qwoa_qaoa_results.json"))
    ms = sorted(int(k) for k in R["m"].keys())
    qwoa_best, qaoa_best, kopts = [], [], []
    for m in ms:
        blk = R["m"][str(m)]
        qaoa_best.append(max(r["p_opt"] for r in blk["qaoa"]))
        qwoa_best.append(max(r["p_opt"] for r in blk["qwoa"]))
        kopts.append(blk["k_opt"])

    def fmt_ratio(r):
        if r < 1000:
            return rf"$\times{r:.0f}$"
        e = int(np.floor(np.log10(r))); mant = r / 10 ** e
        return rf"$\times{mant:.0f}\times10^{{{e}}}$"

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    x = np.arange(len(ms)); w = 0.36
    FLOOR = 1e-8
    qa = np.clip(qaoa_best, FLOOR, 1); qw = np.clip(qwoa_best, FLOOR, 1)
    ax.bar(x - w/2, qw, w, color=C["direct"])
    ax.bar(x + w/2, qa, w, color=C["qaoa"])
    ax.set_yscale("log"); ax.set_ylim(FLOOR, 12.0)
    # direct labels on the first group (Tufte: no legend box)
    ax.text(x[0] - w/2, qw[0]*1.6, lbl(r"QWOA"), ha="center", va="bottom",
            color=C["direct"], fontsize=8.5, fontweight="bold")
    ax.text(x[0] + w/2, qa[0]*1.7, lbl(r"QAOA"), ha="center", va="bottom",
            color=C["qaoa"], fontsize=8.5, fontweight="bold")
    # ratio annotation centered above each group, with a connecting span
    for xi, (a, b) in enumerate(zip(qwoa_best, qaoa_best)):
        ax.text(xi, 3.0, fmt_ratio(a / b), ha="center", va="bottom",
                fontsize=9, color="#333333")
        ax.annotate("", xy=(xi - w/2, a*1.25), xytext=(xi - w/2, 2.6),
                    arrowprops=dict(arrowstyle="-", color="#cccccc", lw=0.5))
    ax.set_xticks(x)
    ax.set_xticklabels([lbl(rf"$m={m},\ k^\star={kk}$")
                        for m, kk in zip(ms, kopts)])
    ax.set_ylabel(lbl(r"best $P(\mathrm{optimum})$"))
    ax.set_title(lbl(r"\textbf{QWOA's edge grows with coverage $m$}")
                 if USETEX else "QWOA's edge grows with coverage m",
                 loc="left", fontsize=9.5)
    apply_tufte(ax)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"fig_advantage_vs_m.{ext}")
    plt.close(fig)
    print(f"wrote {FIGURES_DIR}/fig_advantage_vs_m.pdf (+ .png)")


if __name__ == "__main__":
    print(f"usetex = {USETEX}")
    figure_A()
    figure_B()
    print("done.")
