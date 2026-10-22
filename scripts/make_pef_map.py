#!/usr/bin/env python3
"""Generate Fig. 1 from the frozen PEF-24 instance.

The five plotting sectors follow the contiguous frontier groups in the Army spreadsheet.
The highlighted minimum cover is solved afresh; all 23 frozen waypoints must be covered
at 200 km.
"""
from __future__ import annotations

import json

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from _paths import DATA_DIR, FIGURES_DIR


SECTORS = (
    ("Venezuela--Guyana", range(0, 6), "o", "#2B6CB0"),
    ("Colombia--Amazonas", (6, 7, 9), "s", "#39A824"),
    ("Colombia--Peru", (8, 10, 11, 12, 13, 14), "^", "#F4A582"),
    ("Peru--Acre", range(15, 19), "D", "#762A83"),
    ("Bolivia--Rondônia", range(19, 24), "v", "#D6604D"),
)


def selected_cover(A: np.ndarray) -> list[int]:
    constraints = [LinearConstraint(A, 1, np.inf)]
    result = milp(np.ones(A.shape[1]), integrality=np.ones(A.shape[1]),
                  bounds=Bounds(0, 1), constraints=constraints,
                  options={"presolve": True})
    if not result.success:
        raise RuntimeError(result.message)
    selected = np.flatnonzero(result.x > 0.5).tolist()
    if len(selected) != 10:
        raise RuntimeError(f"expected a 10-post cover, obtained {len(selected)}")
    return selected


def main() -> None:
    frozen = json.load(open(DATA_DIR / "frozen_rung1.json"))
    posts = pd.read_excel(DATA_DIR / "pef_planilha.xlsx")
    A = np.asarray(frozen["A"], int)
    selected = selected_cover(A)

    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    brazil = world[world["name"] == "Brazil"]
    neighbours = world[world["name"].isin(
        ["Venezuela", "Guyana", "Suriname", "Colombia", "Peru", "Bolivia"])]

    plt.rcParams.update({"font.family": "serif", "font.size": 8,
                         "savefig.dpi": 600, "savefig.bbox": "tight"})
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    neighbours.plot(ax=ax, color="#f2f2f2", edgecolor="#cccccc", linewidth=0.5)
    brazil.plot(ax=ax, color="#e8f3e8", edgecolor="#555555", linewidth=1.0)

    for label, indices, marker, colour in SECTORS:
        idx = list(indices)
        ax.scatter(posts.loc[idx, "longitude"], posts.loc[idx, "latitude"], s=42,
                   marker=marker, color=colour, edgecolor="white", linewidth=0.8,
                   label=label, zorder=3)
    ax.scatter(posts.loc[selected, "longitude"], posts.loc[selected, "latitude"],
               s=130, facecolors="none", edgecolors="black", linewidth=1.5,
               label=r"ILP optimal ($k_{\rm opt}=10$)", zorder=4)

    # Label only the selected posts; the complete names and coordinates remain in the input.
    for i in selected:
        ax.annotate(str(posts.loc[i, "localidade"]),
                    (posts.loc[i, "longitude"], posts.loc[i, "latitude"]),
                    xytext=(3, 3), textcoords="offset points", fontsize=6.5)

    ax.set(xlim=(-75, -44), ylim=(-16, 7), xlabel="Longitude (°W)", ylabel="Latitude (°N/S)")
    ax.legend(loc="lower right", fontsize=6.5, frameon=True)
    ax.tick_params(direction="in")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "fig1_pef_map.pdf")
    plt.close(fig)
    print(f"selected post indices: {selected}")
    print("wrote figures/fig1_pef_map.pdf")


if __name__ == "__main__":
    main()
