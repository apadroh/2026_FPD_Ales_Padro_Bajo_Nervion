import os
import pandas as pd
import numpy as np
import unicodedata

# =========================
# PATHS
# =========================

METADATA_PATH = "data/metadata/stations_metadata.csv"
STATIONS_PATH = "data/intermediate/stations_merged"
OUTPUT_PATH = "data/graphs"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# =========================
# TARGET ZONE
# =========================

TARGET_ZONES = [
    1, 2, 3, 4, 5, 6, 7, 8
]

# =========================
# VARIABLES METEO
# =========================

METEO_VARS = [
    "VV",
    "DV",
    "T",
    "H",
    "PR"
]

# Default weight when meteo correlation is missing (e.g. station without data).
# With only 2 stations in the zone, the edge is always created.
FALLBACK_WEIGHT = 0.5

# =========================
# NORMALIZE
# =========================

def normalize(text):

    text = str(text)

    text = (
        unicodedata
        .normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode()
    )

    text = text.lower().strip()

    return text


# =========================
# LOAD METADATA
# =========================

meta = pd.read_csv(
    METADATA_PATH,
    sep=";"
)

meta.columns = (
    meta.columns
    .str.strip()
    .str.lower()
)

meta["zone"] = pd.to_numeric(
    meta["zone"],
    errors="coerce"
)

meta["name_clean"] = (
    meta["name"]
    .apply(normalize)
)

# =========================
# FILTER TARGET ZONE
# =========================
meta_all = meta.copy()

def station_similarity(df1, df2):

    scores = []

    for var in METEO_VARS:

        if (
            var not in df1.columns
            or var not in df2.columns
        ):
            continue

        temp = pd.merge(
            df1[["time", var]],
            df2[["time", var]],
            on="time",
            suffixes=("_1", "_2")
        )

        temp = temp.dropna()

        # mínimo de observaciones
        if len(temp) < 100:
            continue

        corr = temp[
            f"{var}_1"
        ].corr(
            temp[f"{var}_2"]
        )

        if pd.isna(corr):
            continue

        scores.append(
            abs(corr)
        )

    if len(scores) == 0:
        return np.nan

    return np.mean(scores)


for TARGET_ZONE in TARGET_ZONES:
    meta = meta_all[
        meta_all["zone"] == TARGET_ZONE
    ].copy()

    print()
    print("===================================")
    print(f"ZONA SELECCIONADA: {TARGET_ZONE}")
    print("===================================")

    print()
    print(f"Estaciones encontradas: {len(meta)}")

    print()
    print(meta[
        ["id", "name", "zone"]
    ])

    # =========================
    # LOAD STATIONS
    # =========================

    station_data = {}

    valid_stations = set(
        meta["name_clean"]
    )

    for file in os.listdir(STATIONS_PATH):

        if not file.endswith(".csv"):
            continue

        station_name = normalize(
            file.replace(".csv", "")
        )

        # solo estaciones de la zona
        if station_name not in valid_stations:
            continue

        path = os.path.join(
            STATIONS_PATH,
            file
        )

        try:

            df = pd.read_csv(path)

            if "time" not in df.columns:
                continue

            df["time"] = pd.to_datetime(
                df["time"],
                errors="coerce"
            )

            station_data[station_name] = df

        except Exception as e:

            print(
                f"ERROR leyendo {file}: {e}"
            )

    print()
    print(
        f"{len(station_data)} estaciones cargadas para el grafo"
    )

    print()
    print("Estaciones cargadas:")
    print(sorted(station_data.keys()))

    # =========================
    # BUILD GRAPH
    # =========================

    edges = []

    stations = list(
        station_data.keys()
    )

    print()
    print(
        f"Calculando correlaciones para "
        f"{len(stations)} estaciones..."
    )

    for i in range(len(stations)):

        for j in range(i + 1, len(stations)):

            s1 = stations[i]
            s2 = stations[j]

            score = station_similarity(
                station_data[s1],
                station_data[s2]
            )

            if pd.isna(score):
                # Con 2 estaciones deben quedar unidas aunque falte meteo
                if len(stations) == 2:
                    score = FALLBACK_WEIGHT
                    print(
                        f"WARNING sin correlacion {s1}-{s2}; "
                        f"arista fallback weight={score}"
                    )
                else:
                    continue

            row1 = meta[
                meta["name_clean"] == s1
            ]

            row2 = meta[
                meta["name_clean"] == s2
            ]

            if (
                row1.empty
                or row2.empty
            ):
                continue

            id1 = row1.iloc[0]["id"]
            id2 = row2.iloc[0]["id"]

            edges.append(
                [
                    id1,
                    id2,
                    score
                ]
            )

    # =========================
    # SAVE
    # =========================

    graph = pd.DataFrame(
        edges,
        columns=[
            "source",
            "target",
            "weight"
        ]
    )

    output_file = (
        f"{OUTPUT_PATH}/meteo_graph_zone_{TARGET_ZONE}.csv"
    )

    graph.to_csv(
        output_file,
        index=False
    )

    # =========================
    # STATS
    # =========================

    print()
    print("===================================")
    print("GRAFO GENERADO")
    print("===================================")

    print()
    print(f"Archivo: {output_file}")

    print()
    print("Primeras aristas:")
    print(graph.head())

    print()
    print("Número de aristas:")
    print(len(graph))

    if len(graph) > 0:

        print()
        print("Estadísticas pesos:")
        print(
            graph["weight"]
            .describe()
        )

        print()
        print("Top 20 correlaciones:")
        print(
            graph
            .sort_values(
                "weight",
                ascending=False
            )
            .head(20)
        )
