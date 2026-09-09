import pandas as pd
import os
import sys
import unicodedata
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# =========================
# PATHS
# =========================
INPUT_PATH = "data/intermediate/stations_merged"
OUTPUT_PATH = "data/processed"
SAHARIAN_PATH = "data/raw/saharianos.csv"
QUALITY_REPORT_DIR = "data/metadata/data_quality"
SAHARIAN_START_YEAR = 2006

os.makedirs(OUTPUT_PATH, exist_ok=True)

# =========================
# VARIABLES
# =========================
meteo_vars = ["DV", "VV", "T", "H", "PR", "LL", "RA"]
CONTAMINANTS = ["NO2", "SO2", "PM10", "CO", "O3", "NO", "PM25", "NOx"]
METEO_NON_NEGATIVE = ["VV", "PR", "LL", "RA"]


def sanitize_physical_values(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reemplaza valores fisicamente invalidos por NaN.

    Reglas:
      - contaminantes: < 0 -> NaN
      - VV, PR, LL, RA: < 0 -> NaN
      - H: fuera de [0, 100] -> NaN
      - T y DV: no se modifican (T puede ser negativa)
    """
    df = df.copy()
    report_rows: list[dict] = []

    for col in CONTAMINANTS:
        if col not in df.columns:
            continue
        mask = df[col] < 0
        n = int(mask.sum())
        if n:
            df.loc[mask, col] = np.nan
            report_rows.append(
                {
                    "variable": col,
                    "rule": "< 0",
                    "n_replaced": n,
                    "stations_affected": int(
                        df.loc[mask, "station_name"].nunique()
                        if "station_name" in df.columns
                        else 0
                    ),
                }
            )

    for col in METEO_NON_NEGATIVE:
        if col not in df.columns:
            continue
        mask = df[col] < 0
        n = int(mask.sum())
        if n:
            df.loc[mask, col] = np.nan
            report_rows.append(
                {
                    "variable": col,
                    "rule": "< 0",
                    "n_replaced": n,
                    "stations_affected": int(
                        df.loc[mask, "station_name"].nunique()
                        if "station_name" in df.columns
                        else 0
                    ),
                }
            )

    if "H" in df.columns:
        mask = (df["H"] < 0) | (df["H"] > 100)
        n = int(mask.sum())
        if n:
            df.loc[mask, "H"] = np.nan
            report_rows.append(
                {
                    "variable": "H",
                    "rule": "fuera de [0, 100]",
                    "n_replaced": n,
                    "stations_affected": int(
                        df.loc[mask, "station_name"].nunique()
                        if "station_name" in df.columns
                        else 0
                    ),
                }
            )

    return df, pd.DataFrame(report_rows)


# =========================
# LOAD SAHARIAN DAYS
# =========================
def load_saharian_days():

    if not os.path.exists(SAHARIAN_PATH):
        print(f"WARNING No existe el archivo de intrusiones: {SAHARIAN_PATH}")
        return pd.DatetimeIndex([])

    df_sah = pd.read_csv(SAHARIAN_PATH, sep=";")
    df_sah.columns = df_sah.columns.str.strip()

    if "Fecha" in df_sah.columns:

        sah_dates = pd.to_datetime(
            df_sah["Fecha"],
            errors="coerce",
            dayfirst=True
        )

    elif {"aino", "mes", "dia"}.issubset(df_sah.columns):

        sah_dates = pd.to_datetime(
            df_sah[["aino", "mes", "dia"]].rename(
                columns={
                    "aino": "year",
                    "mes": "month",
                    "dia": "day"
                }
            ),
            errors="coerce"
        )

    else:
        print("WARNING saharianos.csv no tiene columnas Fecha ni aino/mes/dia")
        return pd.DatetimeIndex([])

    sah_dates = (
        sah_dates[
            sah_dates.dt.year >= SAHARIAN_START_YEAR
        ]
        .dt.normalize()
        .dropna()
        .drop_duplicates()
    )

    return pd.DatetimeIndex(sah_dates)


SAHARIAN_DAYS = load_saharian_days()

# =========================
# NAME NORMALIZATION
# =========================
def normalize(text):

    text = str(text)
    text = text.strip().lower()

    text = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode()
    )

    text = text.replace(" ", "")

    return text

# =========================
# LOAD METADATA
# =========================
meta = pd.read_csv(
    "data/metadata/metadatos_estaciones.csv"
)

meta.columns = meta.columns.str.strip().str.lower()

rename_dict = {}

if "latitud" in meta.columns:
    rename_dict["latitud"] = "lat"

if "longitud" in meta.columns:
    rename_dict["longitud"] = "lon"

if "zona" in meta.columns:
    rename_dict["zona"] = "zone"

meta = meta.rename(columns=rename_dict)

meta["zone"] = pd.to_numeric(
    meta["zone"],
    errors="coerce"
)

meta["lat"] = pd.to_numeric(
    meta["lat"],
    errors="coerce"
)

meta["lon"] = pd.to_numeric(
    meta["lon"],
    errors="coerce"
)

meta["nombre_clean"] = meta["nombre"].apply(normalize)

# =========================
# CHECK ID COLUMN
# =========================
if "id" not in meta.columns:
    raise ValueError(
        "La metadata debe contener columna 'id'"
    )

# =========================
# MAPPING NOMBRE -> ID
# =========================
station_id_map = dict(
    zip(
        meta["nombre_clean"],
        meta["id"]
    )
)

print("\nMapping estación -> ID cargado")
print(list(station_id_map.items())[:5])


# =========================
# LOAD METEO ASSIGNMENT
# =========================

METEO_ASSIGNMENT_PATH = (
    "data/metadata/stations_meteo_graph_check.csv"
)

meteo_assignment = pd.read_csv(
    METEO_ASSIGNMENT_PATH
)

# =========================
# GET STATIONS BY ZONE
# =========================
def get_stations_by_zone(zones=None):

    print("\nDEBUG ZONES:", meta["zone"].unique())

    if zones is None:
        return meta["nombre"].tolist()

    zones = [float(z) for z in zones]

    selected = meta[
        meta["zone"].isin(zones)
    ]

    print("\nDEBUG seleccion zona:")
    print(selected[["nombre", "zone"]])

    return selected["nombre"].tolist()

# =========================
# FIND FILE
# =========================
def find_station_file(station_name, files):

    station_clean = normalize(station_name)

    for f in files:

        if normalize(f.replace(".csv", "")) == station_clean:
            return f

    return None

def get_source_station(
    station,
    variable
):

    row = meteo_assignment[
        meteo_assignment["station"]
        == station
    ]

    if row.empty:
        return None

    source = row.iloc[0][
        f"{variable}_source"
    ]

    if pd.isna(source):
        return None

    return source

# =========================
# INTEGRATE METEO
# =========================

def integrate_meteo(df, station, files):

    df = df.copy()

    df["time"] = pd.to_datetime(
        df["time"],
        errors="coerce"
    ).dt.floor("h")

    for var in meteo_vars:

        if var not in df.columns:
            df[var] = pd.NA

    for var in meteo_vars:

        before_missing = (
            df[var]
            .isna()
            .sum()
        )

        if before_missing == 0:
            continue

        source_station = (
            get_source_station(
                station,
                var
            )
        )

        if source_station is None:

            print(
                f"WARNING {station}: "
                f"sin proveedor para {var}"
            )

            continue

        candidate_file = (
            find_station_file(
                source_station,
                files
            )
        )

        if candidate_file is None:

            print(
                f"WARNING {station}: "
                f"no encontrado fichero "
                f"para {source_station}"
            )

            continue

        candidate_path = os.path.join(
            INPUT_PATH,
            candidate_file
        )

        df_cand = pd.read_csv(
            candidate_path
        )

        if var not in df_cand.columns:

            print(
                f"WARNING {source_station}: "
                f"no tiene {var}"
            )

            continue

        df_cand["time"] = pd.to_datetime(
            df_cand["time"],
            errors="coerce"
        ).dt.floor("h")

        df_cand = (
            df_cand[
                ["time", var]
            ]
            .rename(
                columns={
                    var: f"{var}_source"
                }
            )
        )

        temp = pd.merge(
            df[["time"]],
            df_cand,
            on="time",
            how="left"
        )

        df[var] = df[var].fillna(
            temp[f"{var}_source"]
        )

        after_missing = (
            df[var]
            .isna()
            .sum()
        )

        print(
            f"{station} | {var} | "
            f"source={source_station} | "
            f"antes={before_missing} | "
            f"después={after_missing}"
        )

    return df
# =========================
# BUILD DATASET LONG FORMAT
# =========================
def build_dataset(zones, name):

    stations = get_stations_by_zone(zones)

    print("\nStations seleccionadas:")
    print(stations)

    df_master = []

    files = os.listdir(INPUT_PATH)

    for station in stations:

        print("\n=========================")
        print(f"Procesando estación: {station}")
        print("=========================")

        matched_file = find_station_file(
            station,
            files
        )

        if matched_file is None:

            print(f"WARNING No encontrado: {station}")

            continue

        path = os.path.join(
            INPUT_PATH,
            matched_file
        )

        print(f"OK Archivo encontrado: {matched_file}")

        df = pd.read_csv(path)

        # =========================
        # TIME
        # =========================
        df["time"] = pd.to_datetime(
            df["time"],
            errors="coerce"
        )

        df["time"] = df["time"].dt.floor("h")

        # =========================
        # GET NUMERIC ID
        # =========================
        station_clean = normalize(station)

        row_meta = meta[
            meta["nombre_clean"] == station_clean
        ]

        if row_meta.empty:

            print(
                f"WARNING No existe metadata para estación {station}"
            )

            continue

        station_id = row_meta.iloc[0]["id"]
        station_lat = row_meta.iloc[0]["lat"]
        station_lon = row_meta.iloc[0]["lon"]
        station_zone = row_meta.iloc[0]["zone"]

        # =========================
        # ADD ID + NAME
        # =========================
        df["ID"] = station_id
        df["station_name"] = station

        df["lat"] = station_lat
        df["lon"] = station_lon
        df["zone"] = station_zone

        # =========================
        # METEO INTEGRATION
        # =========================
        df = integrate_meteo(
            df,
            station,
            files
        )

        # =========================
        # REMOVE AUX COLUMNS
        # =========================
        drop_cols = [
            "year",
            "month",
            "day",
            "hour"
        ]

        df = df.drop(
            columns=[
                c for c in drop_cols
                if c in df.columns
            ],
            errors="ignore"
        )

        # =========================
        # SORT COLUMNS
        # =========================
        first_cols = [
            "time",
            "ID",
            "station_name",
            "lat",
            "lon",
            "zone"
        ]

        remaining_cols = [
            c for c in df.columns
            if c not in first_cols
        ]

        df = df[
            first_cols + remaining_cols
        ]

        # =========================
        # APPEND
        # =========================
        df_master.append(df)

        print(
            f"OK {station}: "
            f"{len(df):,} filas"
        )

    # =========================
    # VALIDATION
    # =========================
    if len(df_master) == 0:

        print("ERROR No se ha cargado ninguna estación")

        return

    # =========================
    # CONCAT
    # =========================
    print("\nConcatenando datasets...")

    df_master = pd.concat(
        df_master,
        ignore_index=True
    )

    # =========================
    # SORT
    # =========================
    df_master = df_master.sort_values(
        ["time", "ID"]
    ).reset_index(drop=True)

    # =========================
    # SANITIZE INVALID VALUES
    # (antes de interpolar meteo)
    # =========================
    print("\nSanitizando valores fisicamente invalidos...")
    df_master, sanitize_report = sanitize_physical_values(df_master)

    if sanitize_report.empty:
        print("Sin valores invalidos a sustituir.")
    else:
        print(sanitize_report.to_string(index=False))
        report_dir = Path(QUALITY_REPORT_DIR) / name
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "sanitize_summary.csv"
        sanitize_report.to_csv(report_path, index=False)
        print(f"Informe sanitize: {report_path}")

    # =========================
    # FINAL TEMPORAL IMPUTATION
    # =========================

    for var in meteo_vars:

        df_master[var] = (
            df_master
            .groupby("station_name")[var]
            .transform(
                lambda x:
                x.interpolate(
                    limit_direction="both"
                )
            )
        )
    # =========================
    # SAHARIAN FLAG
    # =========================
    time_dates = pd.to_datetime(
        df_master["time"],
        errors="coerce"
    ).dt.normalize()

    df_master["Sah"] = (
        (
            time_dates.dt.year
            >= SAHARIAN_START_YEAR
        )
        &
        (
            time_dates.isin(SAHARIAN_DAYS)
        )
    ).astype(int)

    # Continuous net African load at Valderejo (µg/m³); 0 if not in table
    try:
        from src.data.saharan_intensity import load_sah_intensity_by_day

        _sah_int = load_sah_intensity_by_day()
        df_master["Sah_int"] = (
            time_dates.map(_sah_int).fillna(0.0).astype(float)
        )
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING Sah_int not attached: {exc}")
        df_master["Sah_int"] = 0.0

    # =========================
    # OUTPUT
    # =========================
    output_file = (
        f"{OUTPUT_PATH}/dataset_{name}.csv"
    )

    df_master.to_csv(
        output_file,
        index=False
    )

    print("\n=========================")
    print("DATASET FINAL GENERADO")
    print("=========================")

    print(f"Archivo: {output_file}")

    print("\nShape:")
    print(df_master.shape)

    print("\nColumnas:")
    print(df_master.columns.tolist())

    print("\nIDs estaciones:")
    print(df_master["ID"].unique())

    print("\nRango temporal:")
    print(
        f"{df_master['time'].min()} -> {df_master['time'].max()}"
    )

    print("\nValores nulos (%):")
    print(
        (
            df_master.isna().mean() * 100
        ).round(2)
    )

# =========================
# MISSING METEO BY STATION
# =========================

    print("\n=========================")
    print("NULOS METEO POR ESTACIÓN")
    print("=========================")

    for var in meteo_vars:

        print(f"\n{var}")

        tmp = (
            df_master
            .groupby("station_name")[var]
            .apply(
                lambda x:
                round(
                    x.isna().mean() * 100,
                    2
                )
            )
            .sort_values(
                ascending=False
            )
        )

        print(tmp.head(15))
# =========================
# MAIN
# =========================
def parse_zones(text: str) -> list[int]:
    """
    Acepta listas y rangos: "1,3-8" -> [1,3,4,5,6,7,8]
    """
    zones: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            zones.extend(range(start, end + 1))
        else:
            zones.append(int(part))
    # únicos, ordenados
    return sorted(set(zones))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Construye dataset_zone_*.csv usando "
            "stations_meteo_graph_check.csv"
        )
    )
    parser.add_argument(
        "--zones",
        type=str,
        default="1-8",
        help=(
            "Zonas a generar. Ejemplos: '1,3-8', '2', '1-8'. "
            "Default: 1-8."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Also write dataset_all.csv",
    )
    parser.add_argument(
        "--test-all",
        action="store_true",
        help="Also write dataset_test_all.csv",
    )
    args = parser.parse_args()

    zones = parse_zones(args.zones)
    print(f"\nZonas a generar: {zones}")

    if args.test_all:
        build_dataset(None, "test_all")

    for zone in zones:
        build_dataset([zone], f"zone_{zone}")

    if args.all:
        build_dataset(None, "all")