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
    with (RESULTS_DIR / "resource_model_exact.json").open(encoding="utf-8") as stream:
        resource = json.load(stream)
    geo_200 = next(row for row in geographic["sweep"] if row["R_km"] == 200)
    by_tag = {row["tag"]: row for row in resource["circuits"]}
    rung1 = by_tag["rung1_qaoa_p1"]
    rung2 = by_tag["arco_qaoa_p1"]

    scale_labels = ["Rung 3\nMDS\n(773q)", "Rung 1\nMSC\n(24q)", "Rung 2\nMSC\n(79q)"]
    n_zz = np.array([geo_200["n_ZZ"], 17, 9])
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
    left.text(1, 4.5, f"$\\mathcal{{F}}_{{\\rm gate}}={rung1['F_gates']:.2f}$",
              ha="center", color="#444444")
    left.text(2, 2.0, f"$\\mathcal{{F}}_{{\\rm gate}}={rung2['F_gates']:.2f}$",
              ha="center", color="#444444")
    left.annotate(
        "$\\sim4{,}400\\times$ fewer interactions",
        xy=(2, 9),
        xytext=(0.55, 850),
        fontsize=9,
        fontstyle="italic",
        arrowprops={"arrowstyle": "->", "color": "#444444", "lw": 1.0},
    )

    qubits = np.array([24, 79])
    fidelity = np.array([rung1["F_gates"], rung2["F_gates"]])
    point_nzz = np.array([17, 9])
    scatter = right.scatter(
        qubits,
        fidelity,
        c=point_nzz,
        cmap="cividis",
        vmin=0,
        vmax=25,
        s=280,
        edgecolor="black",
        linewidth=1.0,
        zorder=3,
    )
    right.plot(qubits, fidelity, "--", color="#888888", zorder=1)
    right.annotate("Rung 1\n(24q, $n_{ZZ}=17$)", (24, fidelity[0]),
                   xytext=(33, 0.872),
                   arrowprops={"arrowstyle": "-", "color": "#555555"})
    right.annotate(
        "Rung 2\n(79q, $n_{ZZ}=9$)",
        (79, fidelity[1]),
        xytext=(51, 0.970),
        arrowprops={"arrowstyle": "-", "color": "#555555"},
    )
    right.set_xlim(10, 95)
    right.set_ylim(0.84, 0.99)
    right.set_xlabel("Qubit count $n_B$")
    right.set_ylabel("$\\mathcal{F}_{\\mathrm{gate}}$ (gate-only estimate)")
    right.set_title("(b) Gate-only inversion; readout reverses it", loc="left")
    colorbar = figure.colorbar(scatter, ax=right, pad=0.05)
    colorbar.set_label("$n_{ZZ}$")

    for axis in (left, right):
        axis.tick_params(direction="in")
    figure.tight_layout(w_pad=2.0)
    figure.savefig(FIGURES_DIR / "fig3_nzz_fidelity.pdf")
    plt.close(figure)
    print("Wrote figures/fig3_nzz_fidelity.pdf")


if __name__ == "__main__":
    main()
