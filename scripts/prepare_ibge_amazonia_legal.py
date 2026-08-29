"""Prepare the official 2024 IBGE Legal Amazon municipality table.

The raw polygon archive is intentionally not versioned.  This script downloads
the two official IBGE shapefiles, verifies their SHA-256 digests, checks the
municipality codes and writes a compact, deterministic CSV used by the MDS
calculation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

import geopandas as gpd
import pandas as pd

from _paths import DATA_DIR


PRODUCT_YEAR = 2024
MUNICIPALITIES_URL = (
    "https://geoftp.ibge.gov.br/organizacao_do_territorio/estrutura_territorial/"
    "amazonia_legal/2024/Mun_Amazonia_Legal_2024_shp.zip"
)
SEATS_URL = (
    "https://geoftp.ibge.gov.br/organizacao_do_territorio/estrutura_territorial/"
    "amazonia_legal/2024/Sedes_Mun_Amazonia_Legal_2024_shp.zip"
)
EXPECTED_SHA256 = {
    "municipalities": "1af12c71fd1eb33f3eaa3265a26f767ef189efe40272104fd0968cd54385dea4",
    "seats": "3374631a6f8967c3f09d6b93c460254411e363af95b196d7bfb64ce6ce8e2927",
}
EXPECTED_STATE_COUNTS = {
    "AC": 22,
    "AM": 62,
    "AP": 16,
    "MA": 181,
    "MT": 142,
    "PA": 144,
    "RO": 52,
    "RR": 15,
    "TO": 139,
}
EXPECTED_MUNICIPALITIES = sum(EXPECTED_STATE_COUNTS.values())
OFFICIAL_LEGAL_AMAZON_AREA_KM2 = 5_014_107.875
CENTROID_CRS = "EPSG:5880"  # SIRGAS 2000 / Brazil Polyconic


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def obtain(source: str | None, url: str, destination: Path, expected: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source:
        source_path = Path(source).expanduser().resolve()
        if source_path != destination.resolve():
            shutil.copyfile(source_path, destination)
    elif not destination.exists():
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, destination)

    actual = sha256(destination)
    if actual != expected:
        raise ValueError(
            f"SHA-256 mismatch for {destination}: expected {expected}, got {actual}"
        )
    return destination


def prepare(municipalities_zip: Path, seats_zip: Path) -> tuple[pd.DataFrame, dict]:
    municipalities = gpd.read_file(f"zip://{municipalities_zip}")
    seats = gpd.read_file(f"zip://{seats_zip}")

    for label, frame in (("municipalities", municipalities), ("seats", seats)):
        if len(frame) != EXPECTED_MUNICIPALITIES:
            raise ValueError(f"Expected 773 {label}, found {len(frame)}")
        if frame["CD_MUN"].duplicated().any():
            raise ValueError(f"Duplicate municipality codes in {label}")

    municipality_codes = set(municipalities["CD_MUN"].astype(str))
    seat_codes = set(seats["CD_MUN"].astype(str))
    if municipality_codes != seat_codes:
        raise ValueError("Municipality and seat products contain different codes")

    counts = (
        municipalities.groupby("SIGLA_UF")["CD_MUN"]
        .count()
        .sort_index()
        .astype(int)
        .to_dict()
    )
    if counts != EXPECTED_STATE_COUNTS:
        raise ValueError(f"Unexpected state counts: {counts}")
    rounded_area_sum = float(municipalities["AREA_INT"].astype(float).sum())
    if abs(rounded_area_sum - OFFICIAL_LEGAL_AMAZON_AREA_KM2) > 1.0:
        raise ValueError(f"Unexpected Legal Amazon area sum: {rounded_area_sum}")

    # Computing centroids directly in longitude/latitude biases their positions.
    # EPSG:5880 is a projected SIRGAS 2000 CRS covering Brazil.
    centroid_geometries = municipalities.to_crs(CENTROID_CRS).geometry.centroid
    centroids = gpd.GeoSeries(centroid_geometries, crs=CENTROID_CRS).to_crs("EPSG:4326")

    table = pd.DataFrame(
        {
            "cd_mun": municipalities["CD_MUN"].astype(str),
            "nm_mun": municipalities["NM_MUN"].astype(str),
            "sigla_uf": municipalities["SIGLA_UF"].astype(str),
            "nm_uf": municipalities["NM_UF"].astype(str),
            "area_total_km2": municipalities["AREA_TOT"].astype(float),
            "area_legal_amazon_km2": municipalities["AREA_INT"].astype(float),
            "percent_in_legal_amazon": municipalities["PORC_INT"].astype(float),
            "centroid_lat": centroids.y.to_numpy(),
            "centroid_lon": centroids.x.to_numpy(),
        }
    )
    seat_columns = seats[
        ["CD_MUN", "LAT_SEDE", "LNG_SEDE", "SEDE_AL"]
    ].rename(
        columns={
            "CD_MUN": "cd_mun",
            "LAT_SEDE": "seat_lat",
            "LNG_SEDE": "seat_lon",
            "SEDE_AL": "seat_in_legal_amazon",
        }
    )
    seat_columns["cd_mun"] = seat_columns["cd_mun"].astype(str)
    table = table.merge(seat_columns, on="cd_mun", validate="one_to_one")
    table = table.sort_values("cd_mun").reset_index(drop=True)

    provenance = {
        "product": "Amazônia Legal 2024",
        "publisher": "Instituto Brasileiro de Geografia e Estatística (IBGE)",
        "product_year": PRODUCT_YEAR,
        "n_municipalities": EXPECTED_MUNICIPALITIES,
        "state_counts": counts,
        "official_area_product_page_km2": OFFICIAL_LEGAL_AMAZON_AREA_KM2,
        "sum_of_rounded_municipality_AREA_INT_km2": rounded_area_sum,
        "municipality_polygon_source": MUNICIPALITIES_URL,
        "municipality_polygon_sha256": EXPECTED_SHA256["municipalities"],
        "municipal_seat_source": SEATS_URL,
        "municipal_seat_sha256": EXPECTED_SHA256["seats"],
        "source_crs_municipalities": str(municipalities.crs),
        "source_crs_seats": str(seats.crs),
        "centroid_method": (
            "Geometric centroid of each official municipality geometry after "
            f"projection to {CENTROID_CRS}, transformed to EPSG:4326"
        ),
        "raw_files_versioned": False,
    }
    return table, provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--municipalities-zip")
    parser.add_argument("--seats-zip")
    parser.add_argument(
        "--cache-dir", type=Path, default=DATA_DIR / ".cache" / "ibge_2024"
    )
    parser.add_argument(
        "--output", type=Path, default=DATA_DIR / "ibge_amazonia_legal_2024.csv"
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=DATA_DIR / "ibge_amazonia_legal_2024_provenance.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    municipalities_zip = obtain(
        args.municipalities_zip,
        MUNICIPALITIES_URL,
        args.cache_dir / "Mun_Amazonia_Legal_2024_shp.zip",
        EXPECTED_SHA256["municipalities"],
    )
    seats_zip = obtain(
        args.seats_zip,
        SEATS_URL,
        args.cache_dir / "Sedes_Mun_Amazonia_Legal_2024_shp.zip",
        EXPECTED_SHA256["seats"],
    )
    table, provenance = prepare(municipalities_zip, seats_zip)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False, float_format="%.8f", lineterminator="\n")
    args.provenance.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(table)} municipalities to {args.output}")
    print(f"State counts: {provenance['state_counts']}")


if __name__ == "__main__":
    main()
