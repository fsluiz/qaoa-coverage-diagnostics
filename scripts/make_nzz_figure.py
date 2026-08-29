"""Generate the corrected cross-scale n_ZZ and fidelity figure."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _paths import FIGURES_DIR, RESULTS_DIR


def main() -> None:
    with (RESULTS_DIR / "ilp_geographic_ibge_2024_centroids.json").open(
        encoding="utf-8"
    ) as stream:
        geographic = json.load(stream)
    geo_200 = next(row for row in geographic["sweep"] if row["R_km"] == 200)

    scale_labels = ["Scale 3\nMDS\n(773q)", "Scale 1\nMSC\n(24q)", "Scale 2\nMSC\n(79q)"]
    n_zz = np.array([geo_200["n_ZZ"], 21, 9])
    colors = ["#D6604D", "#F4A582", "#2B6CB0"]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
        }
    )
    figure, (left, right) = plt.subplots(1, 2, figsize=(8.0, 3.1))

    x = np.arange(3)
    left.bar(x, n_zz, width=0.5, color=colors, edgecolor="black", linewidth=0.8)
    left.set_yscale("log")
    left.set_ylim(1, 2e5)
    left.set_xticks(x, scale_labels)
    left.set_ylabel("$n_{ZZ}$ (number of ZZ interactions)")
    left.set_title("(a) Sparsification: $n_{ZZ}$ across scales", loc="left")
    left.grid(axis="y", which="major", linestyle=":", color="#AAAAAA")
    for index, value in enumerate(n_zz):
        left.text(index, value * 1.35, f"{value:,}", ha="center", fontweight="bold")
    left.text(0, 1.55, "$\\mathcal{F}\\ll10^{-100}$", ha="center", color="#444444")
    left.text(1, 12.5, "$\\mathcal{F}\\approx0.73$", ha="center", color="#444444")
    left.text(2, 1.55, "$\\mathcal{F}=0.87$", ha="center", color="#444444")
    left.annotate(
        "$\\sim4{,}400\\times$ fewer interactions",
        xy=(2, 9),
        xytext=(0.55, 850),
        fontsize=9,
        fontstyle="italic",
        arrowprops={"arrowstyle": "->", "color": "#444444", "lw": 1.0},
    )

    qubits = np.array([24, 79])
    fidelity = np.array([0.73, 0.87])
    point_nzz = np.array([21, 9])
    scatter = right.scatter(
        qubits,
        fidelity,
        c=point_nzz,
        cmap="RdYlGn_r",
        vmin=0,
        vmax=25,
        s=280,
        edgecolor="black",
        linewidth=1.0,
        zorder=3,
    )
    right.plot(qubits, fidelity, "--", color="#888888", zorder=1)
    right.annotate("Scale 1\n(24q, $n_{ZZ}=21$)", (24, 0.73), xytext=(32, 0.70))
    right.annotate(
        "Scale 2\n(79q, $n_{ZZ}=9$)",
        (79, 0.87),
        xytext=(55, 0.895),
        arrowprops={"arrowstyle": "-", "color": "#555555"},
    )
    right.text(28, 0.78, "$\\mathcal{F}$ driven by $n_{ZZ}$, not $n_B$", color="#2468A2")
    right.set_xlim(10, 95)
    right.set_ylim(0.65, 0.95)
    right.set_xlabel("Qubit count $n_B$")
    right.set_ylabel("Estimated circuit fidelity $\\mathcal{F}$")
    right.set_title("(b) Fidelity inversion: larger circuit, higher $\\mathcal{F}$", loc="left")
    colorbar = figure.colorbar(scatter, ax=right, pad=0.05)
    colorbar.set_label("$n_{ZZ}$")

    for axis in (left, right):
        axis.tick_params(direction="in")
    figure.tight_layout(w_pad=2.0)
    for extension in ("pdf", "png"):
        figure.savefig(FIGURES_DIR / f"fig3_nzz_fidelity.{extension}")
    plt.close(figure)
    print("Wrote corrected figures/fig3_nzz_fidelity.{pdf,png}")


if __name__ == "__main__":
    main()
