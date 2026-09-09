"""
Export long-format data → wide CSV per station for Informer2020.

Format: date, feature_1, ..., target (target last).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.feature_configs import (
    ABLATION_LEVELS,
    default_dataset_path,
    get_feature_config,
    resolve_features_in_dataset,
)
from src.data.saharan_intensity import add_sah_int_context_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export wide CSV per station for Informer2020"
    )
    parser.add_argument("--contaminant", default="PM10")
    parser.add_argument("--zone", type=int, default=2)
    parser.add_argument("--station", required=True)
    parser.add_argument(
        "--feature-config",
        default="F3",
        choices=ABLATION_LEVELS,
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--start-date", default="2019-01-01")
    parser.add_argument(
        "--output-root",
        default="data/training",
        help="Root directory of training datasets (wide CSV per station)",
    )
    parser.add_argument(
        "--min-availability",
        type=float,
        default=80.0,
    )
    return parser.parse_args()


def preprocess_station(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("time").reset_index(drop=True)

    numeric_cols = [
        c
        for c in df.columns
        if c not in {"time", "ID", "station_name", "lat", "lon", "zone"}
    ]
    df[numeric_cols] = df[numeric_cols].interpolate(limit=3)

    if "DV" in df.columns:
        df["wind_sin"] = np.sin(np.deg2rad(df["DV"]))
        df["wind_cos"] = np.cos(np.deg2rad(df["DV"]))

    # Cyclical calendar features (needed by F2–F5; not always in the long CSV)
    t = pd.to_datetime(df["time"])
    df["hour_sin"] = np.sin(2 * np.pi * t.dt.hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * t.dt.hour / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * t.dt.dayofweek / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * t.dt.dayofweek / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * t.dt.month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * t.dt.month / 12.0)

    if "Sah_int" in df.columns:
        df = add_sah_int_context_features(df)

    return df


def main() -> None:
    args = parse_args()
    config = get_feature_config(args.contaminant, args.feature_config)
    dataset_path = Path(
        args.dataset or default_dataset_path(args.zone, args.contaminant)
    )

    df = pd.read_csv(dataset_path, parse_dates=["time"], low_memory=False)
    df = df[df["time"] >= pd.Timestamp(args.start_date)]
    station_df = df[df["station_name"] == args.station].copy()
    if station_df.empty:
        raise ValueError(f"Estacion no encontrada: {args.station}")

    station_df = preprocess_station(station_df)
    feature_cols, excluded, availability = resolve_features_in_dataset(
        station_df,
        args.contaminant,
        args.feature_config,
        min_availability_pct=args.min_availability,
    )

    if config.target not in feature_cols:
        feature_cols = [c for c in feature_cols if c != config.target]
    input_cols = [c for c in feature_cols if c != config.target]
    export_cols = input_cols + [config.target]

    wide = station_df[["time"] + export_cols].copy()
    wide = wide.rename(columns={"time": "date"})
    wide = wide.dropna(subset=export_cols)

    station_key = args.station.replace(" ", "_").replace("(", "").replace(")", "")
    out_dir = (
        Path(args.output_root)
        / f"zone_{args.zone}"
        / "informer2020"
        / station_key
        / args.feature_config
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "data.csv"
    wide.to_csv(csv_path, index=False)

    metadata = {
        "station": args.station,
        "zone": args.zone,
        "contaminant": args.contaminant,
        "feature_config": args.feature_config,
        "target": config.target,
        "input_cols": input_cols,
        "export_cols": export_cols,
        "features_excluded": excluded,
        "feature_availability_pct": availability,
        "n_rows": int(len(wide)),
        "period_start": str(wide["date"].min()),
        "period_end": str(wide["date"].max()),
        "split_ratios": {"train": 0.7, "val": 0.1, "test": 0.2},
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"Exportado: {csv_path}")
    print(f"  Filas: {len(wide):,}")
    print(f"  Columnas: {export_cols}")


if __name__ == "__main__":
    main()
