"""
Export Informer2020 wide CSVs for all stations / one or more feature configs.

Usage:
  python src/data/export_informer2020_all_stations.py --feature-configs F2 F4
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

from src.data.export_informer2020_csv import preprocess_station
from src.data.feature_configs import (
    ALL_FEATURE_LEVELS,
    default_dataset_path,
    get_feature_config,
    resolve_features_in_dataset,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--contaminant", default="PM10")
    p.add_argument("--zone", type=int, default=2)
    p.add_argument(
        "--feature-configs",
        nargs="+",
        default=["F2", "F4"],
        choices=ALL_FEATURE_LEVELS,
    )
    p.add_argument("--dataset", default=None)
    p.add_argument("--start-date", default="2019-01-01")
    p.add_argument("--output-root", default="data/training")
    p.add_argument("--min-availability", type=float, default=80.0)
    p.add_argument(
        "--stations",
        nargs="*",
        default=None,
        help="Subset of station_name values; default = all in dataset",
    )
    return p.parse_args()


def export_one(
    station_df: pd.DataFrame,
    station: str,
    *,
    zone: int,
    contaminant: str,
    feature_config: str,
    output_root: Path,
    min_availability: float,
) -> Path:
    config = get_feature_config(contaminant, feature_config)
    station_df = preprocess_station(station_df.copy())
    feature_cols, excluded, availability = resolve_features_in_dataset(
        station_df,
        contaminant,
        feature_config,
        min_availability_pct=min_availability,
    )
    input_cols = [c for c in feature_cols if c != config.target]
    export_cols = input_cols + [config.target]

    wide = station_df[["time"] + export_cols].copy()
    wide = wide.rename(columns={"time": "date"})
    wide = wide.dropna(subset=export_cols)

    # Informer windows need seq_len + pred_len contiguous rows after dropna.
    min_rows = 48 + 24
    if len(wide) < min_rows:
        raise ValueError(
            f"too few rows after dropna ({len(wide)} < {min_rows}); "
            f"used={export_cols}; excluded={excluded}. "
            f"Resolve features per station (never zone-wide) for Informer packs."
        )

    station_key = station.replace(" ", "_").replace("(", "").replace(")", "")
    out_dir = (
        output_root
        / f"zone_{zone}"
        / "informer2020"
        / station_key
        / feature_config
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "data.csv"
    wide.to_csv(csv_path, index=False)

    metadata = {
        "station": station,
        "zone": zone,
        "contaminant": contaminant,
        "feature_config": feature_config,
        "n_rows": int(len(wide)),
        "columns": ["date"] + export_cols,
        # Keys expected by experiment_informer2020.py
        "features": export_cols,
        "export_cols": export_cols,
        "input_cols": input_cols,
        "features_excluded": excluded,
        "excluded": excluded,
        "availability_pct": availability,
        "description": config.description,
        "model": "informer2020",
        "seq_len": 48,
        "label_len": 24,
        "pred_len": 24,
        "path": str(csv_path.as_posix()),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return csv_path


def main() -> None:
    args = parse_args()
    dataset_path = Path(
        args.dataset or default_dataset_path(args.zone, args.contaminant)
    )
    print(f"Loading {dataset_path} ...", flush=True)
    df = pd.read_csv(dataset_path, parse_dates=["time"], low_memory=False)
    df = df[df["time"] >= pd.Timestamp(args.start_date)]
    stations = args.stations or sorted(df["station_name"].dropna().unique())
    print(
        f"Stations={len(stations)} configs={args.feature_configs}",
        flush=True,
    )

    ok = fail = 0
    for cfg in args.feature_configs:
        print(f"=== {cfg} ===", flush=True)
        for station in stations:
            try:
                station_df = df[df["station_name"] == station]
                if station_df.empty:
                    raise ValueError(f"empty station slice: {station!r}")
                path = export_one(
                    station_df,
                    station,
                    zone=args.zone,
                    contaminant=args.contaminant,
                    feature_config=cfg,
                    output_root=Path(args.output_root),
                    min_availability=args.min_availability,
                )
                ok += 1
                print(f"  OK {station} -> {path}", flush=True)
            except Exception as exc:  # noqa: BLE001
                fail += 1
                print(f"  FAIL {station}: {exc}", flush=True)

    print(f"Done. ok={ok} fail={fail}", flush=True)


if __name__ == "__main__":
    main()
