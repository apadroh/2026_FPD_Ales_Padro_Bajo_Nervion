"""
Station ingestion: metadata + API extract + annual merge.

Steps:
  metadata  -> data/metadata/metadatos_estaciones.csv
  extract   -> data/raw/stations_csv/<station>/<year>.csv
  merge     -> data/intermediate/stations_merged/<station>.csv
  all       -> all three in order

Examples:
  python src/data/ingest_stations.py --step metadata
  python src/data/ingest_stations.py --step extract
  python src/data/ingest_stations.py --step merge
  python src/data/ingest_stations.py --step all
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw" / "stations_csv"
MERGED_DIR = ROOT / "data" / "intermediate" / "stations_merged"
METADATA_PATH = ROOT / "data" / "metadata" / "metadatos_estaciones.csv"
STATIONS_URL = "https://api.euskadi.eus/air-quality/stations"
DEFAULT_YEARS = list(range(2012, 2026))

# Zone / typology manual mapping (API name -> attributes)
STATION_EXTRA = [
    ["ABANTO", 2, "Bajo_nervion", "suburban", "industrial"],
    ["AGURAIN", 7, "Llanada", "suburban", "background"],
    ["ALGORTA (BBIZI2)", 2, "Bajo_nervion", "suburban", "industrial"],
    ["ALONSOTEGI", 2, "Bajo_nervion", "urban", "industrial"],
    ["ANDOAIN", 4, "Donostialdea", "urban", "industrial"],
    ["ANORGA", 4, "Donostialdea", "urban", "industrial"],
    ["ARRAIZ (Monte)", 2, "Bajo_nervion", "rural", "industrial"],
    ["ATEGORRIETA", 4, "Donostialdea", "urban", "traffic"],
    ["AV. GASTEIZ", 7, "Llanada", "urban", "traffic"],
    ["AVDA. TOLOSA", 4, "Donostialdea", "urban", "traffic"],
    ["AZPEITIA", 6, "Goierri", "urban", "traffic"],
    ["BARAKALDO", 2, "Bajo_nervion", "urban", "traffic"],
    ["BASAURI", 2, "Bajo_nervion", "urban", "industrial"],
    ["BEASAIN", 6, "Goierri", "suburban", "traffic"],
    ["BOROA", 5, "Ibaizabal_Alto_Deba", "urban", "industrial"],
    ["CASTREJANA", 2, "Bajo_nervion", "suburban", "industrial"],
    ["DURANGO", 5, "Ibaizabal_Alto_Deba", "urban", "industrial"],
    ["EASO", 4, "Donostialdea", "urban", "traffic"],
    ["ELCIEGO", 8, "Ribera", "suburban", "traffic"],
    ["ERANDIO", 2, "Bajo_nervion", "urban", "traffic"],
    ["EUROPA", 2, "Bajo_nervion", "urban", "background"],
    ["FARMACIA", 7, "Llanada", "suburban", "background"],
    ["HERNANI", 4, "Donostialdea", "urban", "traffic"],
    ["JAIZKIBEL", 4, "Donostialdea", "rural", "background"],
    ["LARRABETZU", 5, "Ibaizabal_Alto_Deba", "suburban", "industrial"],
    ["LAS CARRERAS", 2, "Bajo_nervion", "urban", "industrial"],
    ["LASARTE-ORIA", 4, "Donostialdea", "urban", "traffic"],
    ["LEMONA", 5, "Ibaizabal_Alto_Deba", None, "industrial"],
    ["LEZO", 4, "Donostialdea", "urban", "industrial"],
    ["LLODIO", 1, "Encartaciones", "suburban", "traffic"],
    ["LOS HERRAN", 7, "Llanada", "urban", "traffic"],
    ["MAZARREDO", 2, "Bajo_nervion", "urban", "traffic"],
    ["MONDRAGON", 5, "Ibaizabal_Alto_Deba", "urban", "traffic"],
    ["MONTORRA", 5, "Ibaizabal_Alto_Deba", "suburban", "industrial"],
    ["MUNDAKA", 3, "Kostaldea", "rural", "background"],
    ["MUNOA", 2, "Bajo_nervion", "urban", "industrial"],
    ["MUSKIZ", 2, "Bajo_nervion", "suburban", "industrial"],
    ["PAGOETA", 3, "Kostaldea", "rural", "background"],
    ["PUYO", 4, "Donostialdea", "urban", "background"],
    ["SAN JULIAN", 2, "Bajo_nervion", "suburban", "industrial"],
    ["SANGRONIZ", 2, "Bajo_nervion", "suburban", "traffic"],
    ["SANTURCE", 2, "Bajo_nervion", "suburban", "industrial"],
    ["SERANTES", 2, "Bajo_nervion", "rural", "background"],
    ["SESTAO", 2, "Bajo_nervion", "urban", "industrial"],
    ["TOLOSA", 6, "Goierri", "urban", "traffic"],
    ["URKIOLA", 5, "Ibaizabal_Alto_Deba", "rural", "background"],
    ["USURBIL", 4, "Donostialdea", "urban", "traffic"],
    ["VALDEREJO", 8, "Ribera", "rural", "background"],
    ["ZALLA", 1, "Encartaciones", "urban", "background"],
    ["ZELAIETA PARQUE", 5, "Ibaizabal_Alto_Deba", None, "industrial"],
    ["ZIERBENA (Puerto)", 2, "Bajo_nervion", "urban", "industrial"],
    ["ZUBIETA", 4, "Donostialdea", "urban", "industrial"],
    ["ZUMARRAGA", 6, "Goierri", "urban", "industrial"],
    ["3 DE MARZO", 7, "Llanada", "urban", "traffic"],
]


# =========================
# API helpers
# =========================

def obtener_datos_api(url: str):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.error(f"Error API: {e}")
        return None


def obtener_estaciones(with_location: bool = False) -> list[dict]:
    data = obtener_datos_api(STATIONS_URL)
    if not data or "features" not in data:
        logging.error("No se pudo obtener la lista de estaciones.")
        return []

    rows = []
    for feature in data["features"]:
        props = feature["properties"]
        row = {
            "id": props["id"],
            "nombre": props["name"],
        }
        if with_location:
            row["municipality"] = props["location"]["municipality"]
            row["coordenadas"] = feature["geometry"]["coordinates"]
        rows.append(row)
    return rows


def seleccionar_valor(preferido, nuevo):
    # La API puede devolver el mismo parámetro 2 veces (p.ej. [290.0, 0.0]).
    if preferido is None:
        return nuevo
    if nuevo is None:
        return preferido
    try:
        p = float(preferido)
        n = float(nuevo)
        if p == 0.0 and n != 0.0:
            return nuevo
        if p != 0.0 and n == 0.0:
            return preferido
    except (TypeError, ValueError):
        pass
    return nuevo


def limpiar_nombre(texto: str | None) -> str | None:
    if not texto:
        return texto
    texto = texto.upper().strip()
    texto = "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )
    texto = re.sub(r"\(.*?\)", "", texto).strip()
    return texto


# =========================
# metadata
# =========================

def step_metadata() -> None:
    print("\n=== STEP: metadata ===")
    estaciones = obtener_estaciones(with_location=True)
    if not estaciones:
        raise RuntimeError("Sin estaciones desde la API")

    df = pd.DataFrame(estaciones)
    df[["longitud", "latitud"]] = pd.DataFrame(
        df["coordenadas"].tolist(), index=df.index
    )
    df = df.drop(columns=["coordenadas"])
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df = df.sort_values("id").reset_index(drop=True)
    df["codigo"] = df["id"].apply(lambda x: f"S{int(x):03d}")

    df_extra = pd.DataFrame(
        STATION_EXTRA,
        columns=["nombre", "zone", "zone_des", "area", "type"],
    )
    df["nombre_clean"] = df["nombre"].apply(limpiar_nombre)
    df_extra["nombre_clean"] = df_extra["nombre"].apply(limpiar_nombre)

    df_final = df.merge(
        df_extra.drop(columns=["nombre"]),
        on="nombre_clean",
        how="left",
    )

    no_match = df_final[df_final["zone"].isna()][["nombre", "nombre_clean"]]
    if not no_match.empty:
        print("\nNO MATCH (sin zona):")
        print(no_match.to_string(index=False))

    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(METADATA_PATH, index=False, encoding="utf-8-sig")
    print(f"Guardado: {METADATA_PATH} ({len(df_final)} estaciones)")


# =========================
# extract
# =========================

def procesar_datos_estacion(datos_api: list) -> list[dict]:
    datos = []
    for item in datos_api:
        fecha_dt = datetime.strptime(item["date"], "%Y-%m-%dT%H:%M:%SZ")
        base = {
            "time": fecha_dt,
            "year": fecha_dt.year,
            "month": fecha_dt.month,
            "day": fecha_dt.day,
            "hour": fecha_dt.hour,
        }
        for station_data in item["station"]:
            registro = base.copy()
            registro.update({
                "NO2": None, "SO2": None, "PM10": None, "CO": None,
                "O3": None, "NO": None, "PM25": None, "NOx": None,
                "DV": None, "VV": None, "T": None, "H": None,
                "PR": None, "LL": None, "RA": None,
            })
            for m in station_data["measurements"]:
                nombre = m["name"]
                valor = m.get("value")
                mapping = {
                    "NO2": "NO2", "SO2": "SO2", "PM10": "PM10", "CO": "CO",
                    "O3": "O3", "NO": "NO", "PM2,5": "PM25", "NOX": "NOx",
                    "D.vien": "DV", "V.vien": "VV", "Tº": "T", "H": "H",
                    "P": "PR", "Precipitación": "LL", "R": "RA",
                }
                col = mapping.get(nombre)
                if col is not None:
                    registro[col] = seleccionar_valor(registro[col], valor)
            datos.append(registro)
    return datos


def obtener_rangos_mensuales(year: int) -> list[tuple[str, str]]:
    rangos = []
    for month in range(1, 13):
        start = f"{year}-{month:02d}-01T00:00"
        if month == 12:
            end = f"{year + 1}-01-01T00:00"
        else:
            end = f"{year}-{month + 1:02d}-01T00:00"
        rangos.append((start, end))
    return rangos


def step_extract(years: list[int] | None = None) -> None:
    print("\n=== STEP: extract ===")
    years = years or DEFAULT_YEARS
    estaciones = obtener_estaciones(with_location=False)
    if not estaciones:
        raise RuntimeError("Sin estaciones desde la API")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for estacion in estaciones:
        print(f"Estacion: {estacion['nombre']}")
        for year in years:
            folder = RAW_DIR / estacion["nombre"]
            file_path = folder / f"{year}.csv"

            if file_path.exists():
                file_path.unlink()
                print(f"  Eliminado: {file_path}")

            datos = []
            for start, end in obtener_rangos_mensuales(year):
                url = (
                    "https://api.euskadi.eus/air-quality/measurements/hourly/"
                    f"stations/{estacion['id']}/from/{start}/to/{end}"
                )
                datos_api = obtener_datos_api(url)
                if not datos_api:
                    print(f"  Sin datos {estacion['nombre']} {start} -> {end}")
                    continue
                datos.extend(procesar_datos_estacion(datos_api))

            if not datos:
                continue

            df = pd.DataFrame(datos)
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"], errors="coerce")
                df = df[df["time"].dt.year == year].copy()
                df = df.sort_values("time").drop_duplicates(
                    subset=["time"], keep="last"
                )

            folder.mkdir(parents=True, exist_ok=True)
            df.to_csv(file_path, index=False)
            print(f"  Guardado: {file_path}")


# =========================
# merge
# =========================

def step_merge() -> None:
    print("\n=== STEP: merge ===")
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"No existe {RAW_DIR}")

    MERGED_DIR.mkdir(parents=True, exist_ok=True)

    for station in os.listdir(RAW_DIR):
        station_path = RAW_DIR / station
        if not station_path.is_dir():
            continue

        dfs = []
        for file in os.listdir(station_path):
            if file.endswith(".csv"):
                dfs.append(pd.read_csv(station_path / file))

        if not dfs:
            continue

        df_all = pd.concat(dfs, ignore_index=True)
        df_all["time"] = pd.to_datetime(df_all["time"])
        df_all = df_all.sort_values("time")
        out = MERGED_DIR / f"{station}.csv"
        df_all.to_csv(out, index=False)
        print(f"Merged: {station} -> {out}")


# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingestion: metadata + API extract + merge stations"
    )
    parser.add_argument(
        "--step",
        choices=["metadata", "extract", "merge", "all"],
        default="all",
        help="Step to run. Default: all",
    )
    parser.add_argument(
        "--years",
        type=str,
        default=None,
        help="Anios para extract, ej: 2012-2025 o 2024,2025",
    )
    return parser.parse_args()


def parse_years(text: str | None) -> list[int] | None:
    if not text:
        return None
    years: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            years.extend(range(int(a), int(b) + 1))
        else:
            years.append(int(part))
    return sorted(set(years))


def main() -> None:
    args = parse_args()
    years = parse_years(args.years)

    if args.step in ("metadata", "all"):
        step_metadata()
    if args.step in ("extract", "all"):
        step_extract(years=years)
    if args.step in ("merge", "all"):
        step_merge()

    print("\nIngesta finalizada.")


if __name__ == "__main__":
    main()
