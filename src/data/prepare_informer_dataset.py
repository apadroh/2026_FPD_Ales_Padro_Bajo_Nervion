"""
Build Informer training tensors from quasi-definitive datasets
per contaminant and feature configuration (F1–F5).

Generates sliding windows per station:
  X: [n_samples, seq_len, n_features]
  y: [n_samples, horizon, n_targets]
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
    default_output_dir,
    get_feature_config,
    resolve_features_in_dataset,
)
from src.data.preprocessing import preprocess_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Informer X/y tensors (F1–F5 ablation)"
    )
    parser.add_argument("--contaminant", default="PM10")
    parser.add_argument("--zone", type=int, default=2)
    parser.add_argument(
        "--feature-config",
        default="F3",
        choices=ABLATION_LEVELS,
        help="Feature configuration F1–F5",
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--start-date", default="2019-01-01")
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--auto-seq-len", action="store_true")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument(
        "--min-availability",
        type=float,
        default=80.0,
        help="Minimum availability (%%) to include a feature",
    )
    return parser.parse_args()


def load_dataset(path: Path, start_date: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontro {path}")

    df = pd.read_csv(path, parse_dates=["time"], low_memory=False)
    start = pd.Timestamp(start_date)
    df = df[df["time"] >= start].copy()
    if df.empty:
        raise ValueError(
            f"No hay filas tras filtrar desde {start_date} en {path}"
        )
    return df.sort_values(["time", "ID"]).reset_index(drop=True)


def build_station_sequences(
    df_station: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    seq_len: int,
    horizon: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[pd.Timestamp]]:
    df_station = df_station.sort_values("time").reset_index(drop=True)
    features = df_station[feature_cols].to_numpy(dtype=np.float32)
    target = df_station[target_col].to_numpy(dtype=np.float32)
    times = df_station["time"].tolist()

    x_windows: list[np.ndarray] = []
    y_windows: list[np.ndarray] = []
    anchor_times: list[pd.Timestamp] = []

    max_start = len(df_station) - seq_len - horizon + 1
    for start_idx in range(max_start):
        end_idx = start_idx + seq_len
        target_end = end_idx + horizon

        x_window = features[start_idx:end_idx]
        y_window = target[end_idx:target_end]

        if np.isnan(x_window).any() or np.isnan(y_window).any():
            continue

        x_windows.append(x_window)
        y_windows.append(y_window.reshape(-1, 1))
        anchor_times.append(times[end_idx - 1])

    return x_windows, y_windows, anchor_times


def temporal_split(
    x_list: list[np.ndarray],
    y_list: list[np.ndarray],
    anchor_times: list[pd.Timestamp],
    train_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(anchor_times)
    split_idx = int(len(order) * train_ratio)

    train_idx = order[:split_idx]
    val_idx = order[split_idx:]

    x_train = np.stack([x_list[i] for i in train_idx])
    y_train = np.stack([y_list[i] for i in train_idx])
    x_val = np.stack([x_list[i] for i in val_idx])
    y_val = np.stack([y_list[i] for i in val_idx])

    return x_train, y_train, x_val, y_val


def save_tensors(
    output_dir: Path,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    metadata: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "X_train.npy", x_train)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "X_val.npy", x_val)
    np.save(output_dir / "y_val.npy", y_val)

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)


def resolve_seq_len(
    args: argparse.Namespace,
    dataset_path: Path,
    target_col: str,
) -> tuple[int, dict | None]:
    if args.seq_len is not None and args.auto_seq_len:
        raise ValueError("Usa --seq-len o --auto-seq-len, no ambos")

    if args.auto_seq_len:
        rec_path = Path(
            f"analysis/acf/zone_{args.zone}/{target_col}/seq_len_recommendation.json"
        )
        if not rec_path.exists():
            raise FileNotFoundError(
                f"No existe {rec_path}. "
                "Run src/data/feature_selection.ipynb "
                f"(or src/data/acf_seq_len.ipynb) first "
                f"(ZONE={args.zone}, CONTAMINANT={target_col})."
            )
        with open(rec_path, encoding="utf-8") as f:
            recommendation = json.load(f)
        seq_len = int(recommendation["recommended_seq_len"])
        print(
            f"  seq_len = {seq_len} h "
            f"(desde {rec_path.as_posix()}; "
            f"memoria util {recommendation.get('memory_lag_hours')} h)"
        )
        return seq_len, recommendation

    return args.seq_len if args.seq_len is not None else 48, None


def main() -> None:
    args = parse_args()
    config = get_feature_config(args.contaminant, args.feature_config)
    dataset_path = Path(
        args.dataset or default_dataset_path(args.zone, args.contaminant)
    )
    output_dir = Path(
        args.output_dir
        or default_output_dir(args.zone, args.contaminant, args.feature_config)
    )
    seq_len, seq_recommendation = resolve_seq_len(
        args, dataset_path, config.target
    )

    print(f"Contaminante : {args.contaminant}")
    print(f"Feature cfg  : {args.feature_config} — {config.description}")
    print(f"Cargando     : {dataset_path}")

    df = load_dataset(dataset_path, args.start_date)
    print(f"  Filas: {len(df):,} | Estaciones: {df['station_name'].nunique()}")
    print(f"  Periodo: {df['time'].min()} -> {df['time'].max()}")

    print("\nPreprocesando...")
    df, _ = preprocess_dataset(df)

    feature_cols, excluded, availability = resolve_features_in_dataset(
        df,
        args.contaminant,
        args.feature_config,
        min_availability_pct=args.min_availability,
    )
    print(f"  Features usadas ({len(feature_cols)}): {feature_cols}")
    if excluded:
        print(f"  Excluidas (<{args.min_availability}%): {excluded}")

    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_times: list[pd.Timestamp] = []

    print("\nGenerando ventanas por estacion...")
    for station_name, group in df.groupby("station_name", sort=True):
        x_windows, y_windows, anchor_times = build_station_sequences(
            group,
            feature_cols,
            config.target,
            seq_len,
            args.horizon,
        )
        if not x_windows:
            print(f"  [SKIP] {station_name}: sin ventanas validas")
            continue

        all_x.extend(x_windows)
        all_y.extend(y_windows)
        all_times.extend(anchor_times)
        print(f"  {station_name}: {len(x_windows):,} ventanas")

    if not all_x:
        raise RuntimeError("No se generaron ventanas validas para entrenar")

    x_train, y_train, x_val, y_val = temporal_split(
        all_x, all_y, all_times, args.train_ratio
    )

    metadata = {
        "dataset": str(dataset_path.as_posix()),
        "output_dir": str(output_dir.as_posix()),
        "zone": args.zone,
        "contaminant": args.contaminant,
        "feature_config": args.feature_config,
        "feature_config_description": config.description,
        "start_date": args.start_date,
        "seq_len": seq_len,
        "horizon": args.horizon,
        "seq_len_recommendation": seq_recommendation,
        "train_ratio": args.train_ratio,
        "min_availability_pct": args.min_availability,
        "target_col": config.target,
        "feature_cols": feature_cols,
        "features_excluded": excluded,
        "feature_availability_pct": availability,
        "n_features": len(feature_cols),
        "n_targets": 1,
        "n_train": int(len(x_train)),
        "n_val": int(len(x_val)),
        "x_train_shape": list(x_train.shape),
        "y_train_shape": list(y_train.shape),
        "period_start": str(df["time"].min()),
        "period_end": str(df["time"].max()),
        "n_stations": int(df["station_name"].nunique()),
    }

    save_tensors(output_dir, x_train, y_train, x_val, y_val, metadata)

    print("\nTensores guardados en:", output_dir)
    print(f"  X_train: {tuple(x_train.shape)}")
    print(f"  y_train: {tuple(y_train.shape)}")
    print(f"  X_val  : {tuple(x_val.shape)}")
    print(f"  y_val  : {tuple(y_val.shape)}")


if __name__ == "__main__":
    main()
