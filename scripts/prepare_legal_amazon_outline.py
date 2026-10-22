#!/usr/bin/env python3
"""Archive a compact outline of the Legal Amazon from the official IBGE 2024 polygons.

The 773 municipality polygons (the same hash-pinned archive used for the centroids, see
prepare_ibge_amazonia_legal.py) are dissolved into one outline, transformed to EPSG:4326 and
simplified at 0.005 deg (about 0.5 km, below the 0.01 deg precision of the PEF registry
coordinates).  Along the northern and western frontier this outline is the national border
at IBGE resolution; it is used to draw Fig. 3(a) and to count waypoints outside Brazil.

Writes data/legal_amazon_outline_2024.geojson.  Needs the raw IBGE archive once (cached,
not versioned); the figure and the claim checks then use only the GeoJSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd

from _paths import DATA_DIR
from prepare_ibge_amazonia_legal import EXPECTED_SHA256, MUNICIPALITIES_URL, obtain

TOLERANCE_DEG = 0.005
OUT = DATA_DIR / "legal_amazon_outline_2024.geojson"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--municipalities-zip")
    ap.add_argument("--cache-dir", type=Path, default=DATA_DIR / ".cache" / "ibge_2024")
    a = ap.parse_args()
    archive = obtain(a.municipalities_zip, MUNICIPALITIES_URL,
                     a.cache_dir / "Mun_Amazonia_Legal_2024_shp.zip",
                     EXPECTED_SHA256["municipalities"])
    mun = gpd.read_file(f"zip://{archive}")
    outline = mun.to_crs("EPSG:4326").dissolve().geometry.iloc[0]
    outline = outline.simplify(TOLERANCE_DEG, preserve_topology=True)
    gdf = gpd.GeoDataFrame(
        {"name": ["Amazonia Legal 2024 (IBGE)"],
         "source": [MUNICIPALITIES_URL],
         "source_sha256": [EXPECTED_SHA256["municipalities"]],
         "n_municipalities": [len(mun)],
         "simplify_tolerance_deg": [TOLERANCE_DEG]},
        geometry=[outline], crs="EPSG:4326")
    gdf.to_file(OUT, driver="GeoJSON")
    print(f"{len(mun)} municipalities dissolved; -> {OUT} "
          f"({OUT.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
