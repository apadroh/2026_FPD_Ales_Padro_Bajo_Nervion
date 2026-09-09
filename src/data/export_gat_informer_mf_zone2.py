"""
Export GAT-Informer multi-feature (GAT-MF) zone packs.

Unlike the original GAT export (PM10-only), windows include all F* channels:
  X [W, N, L, F], Y [W, N, H]  (Y = target only).

Does not overwrite data/training/.../gat_informer/ (v1 packs stay intact).

Usage:
  python src/data/export_gat_informer_mf_zone2.py --feature-configs F1 F2 F3 F4 F5
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

from src.data.export_airformer_zone2 import (
    add_derived_features,
    build_adjacency,
    interpolate_numeric,
    pivot_feature_cube,
    resolve_seq_len,
    temporal_split_indices,
)
from src.data.feature_configs import (
    ABLATION_LEVELS,
    default_dataset_path,
    resolve_features_in_dataset,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export GAT-Informer-MF ZONE2 packs")
    p.add_argument("--zone", type=int, default=2)
    p.add_argument("--contaminant", default="PM10")
    p.add_argument(
        "--feature-configs",
        nargs="+",
        default=["F1", "F2", "F3", "F4", "F5"],
        choices=ABLATION_LEVELS,
    )
    p.add_argument("--dataset", default=None)
    p.add_argument("--start-date", default="2019-01-01")
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.10)
    p.add_argument("--min-availability", type=float, default=80.0)
    p.add_argument("--interpolate-limit", type=int, default=3)
    p.add_argument("--graph-csv", default=None)
    p.add_argument("--station-map-csv", default=None)
    p.add_argument("--output-root", default="data/training")
    return p.parse_args()


def make_node_windows_mf(
    cube_tnf: np.ndarray, seq_len: int, horizon: int, target_idx: int = 0
):
    """cube [T,N,F] -> X [W,N,L,F], Y [W,N,H] (target only)."""
    t, n, f = cube_tnf.shape
    xs, ys = [], []
    for i in range(t - seq_len - horizon + 1):
        x = np.transpose(cube_tnf[i : i + seq_len], (1, 0, 2))  # N,L,F
        y = cube_tnf[i + seq_len : i + seq_len + horizon, :, target_idx].T  # N,H
        if np.isnan(x[:, :, target_idx]).any() or np.isnan(y).any():
            continue
        x = np.nan_to_num(x, nan=0.0)
        xs.append(x.astype(np.float32))
        ys.append(y.astype(np.float32))
    if not xs:
        return (
            np.empty((0, n, seq_len, f), dtype=np.float32),
            np.empty((0, n, horizon), dtype=np.float32),
        )
    return np.stack(xs), np.stack(ys)


def export_level(
    df: pd.DataFrame,
    *,
    level: str,
    zone: int,
    contaminant: str,
    spatial_stations: list[str],
    adj: np.ndarray,
    seq_len: int,
    horizon: int,
    train_ratio: float,
    val_ratio: float,
    min_availability: float,
    out_root: Path,
    seq_len_subdir: bool = False,
    horizon_subdir: bool = False,
) -> dict | None:
    feature_cols, excluded, _ = resolve_features_in_dataset(
        df, contaminant, level, min_availability_pct=min_availability
    )
    # Target must be channel 0 for train_mf.
    if feature_cols[0] != contaminant:
        feature_cols = [contaminant] + [c for c in feature_cols if c != contaminant]

    _times, cube = pivot_feature_cube(df, spatial_stations, feature_cols)
    print(f"GAT-MF {level}: cube {cube.shape} features={feature_cols}")

    X, Y = make_node_windows_mf(cube, seq_len, horizon, target_idx=0)
    print(f"  windows: X={X.shape} Y={Y.shape}")
    if len(X) == 0:
        print(f"  SKIP {level}: no complete windows")
        return None

    n_train, n_val, n_test = temporal_split_indices(len(X), train_ratio, val_ratio)
    train_x = X[:n_train]
    # Per-feature min-max on train windows
    vmax = np.nanmax(train_x, axis=(0, 1, 2)).astype(np.float64)  # [F]
    vmin = np.nanmin(train_x, axis=(0, 1, 2)).astype(np.float64)
    span = vmax - vmin
    span[span == 0.0] = 1.0

    def norm_x(a: np.ndarray) -> np.ndarray:
        return ((a - vmin) / span).astype(np.float32)

    def norm_y(a: np.ndarray) -> np.ndarray:
        return ((a - vmin[0]) / span[0]).astype(np.float32)

    out = {
        "train_x_raw": norm_x(X[:n_train]),
        "train_y": norm_y(Y[:n_train]),
        "vail_x_raw": norm_x(X[n_train : n_train + n_val]),
        "vail_y": norm_y(Y[n_train : n_train + n_val]),
        "test_x_raw": norm_x(X[n_train + n_val :]),
        "test_y": norm_y(Y[n_train + n_val :]),
        # [F, 2]: col0=vmax, col1=vmin (target row 0 used for inverse Y/pred)
        "max_min": np.stack([vmax, vmin], axis=1).astype(np.float32),
        "graph": adj.astype(np.float32),
    }

    out_dir = out_root / f"zone_{zone}" / "gat_informer_mf"
    if seq_len_subdir:
        out_dir = out_dir / f"seq_len_{seq_len}"
    if horizon_subdir:
        out_dir = out_dir / f"horizon_{horizon}"
    out_dir = out_dir / level
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"data{seq_len}.npz"
    np.savez_compressed(npz_path, **out)

    meta = {
        "model": "gat_informer_mf",
        "zone": zone,
        "contaminant": contaminant,
        "feature_config": level,
        "note_features": (
            "Multi-channel X [W,N,L,F]; Y is target only. "
            "GAT uses flatten(L*F); Informer sees target channel 0."
        ),
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "features_excluded": excluded,
        "stations": spatial_stations,
        "num_nodes": len(spatial_stations),
        "seq_len": seq_len,
        "horizon": horizon,
        "n_train": int(n_train),
        "n_val": int(n_val),
        "n_test": int(n_test),
        "x_shape": list(out["train_x_raw"].shape),
        "y_shape": list(out["train_y"].shape),
        "path": str(npz_path.relative_to(ROOT)).replace("\\", "/"),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return meta


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset) if args.dataset else ROOT / default_dataset_path(
        args.zone, args.contaminant
    )
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    graph_csv = (
        Path(args.graph_csv)
        if args.graph_csv
        else ROOT / f"data/graphs/meteo_graph_zone_{args.zone}_top5.csv"
    )
    station_map_csv = (
        Path(args.station_map_csv)
        if args.station_map_csv
        else ROOT / "data/metadata/stations_meteo_graph_check.csv"
    )
    out_root = ROOT / args.output_root
    seq_len = resolve_seq_len(args.zone, args.contaminant, args.seq_len)
    seq_len_subdir = int(seq_len) != 48
    horizon_subdir = int(args.horizon) != 24

    print(f"Loading {dataset_path} …")
    raw = pd.read_csv(dataset_path, parse_dates=["time"], low_memory=False)
    raw = raw[raw["time"] >= pd.Timestamp(args.start_date)].copy()
    df = add_derived_features(raw)
    df = interpolate_numeric(df, limit=args.interpolate_limit)
    df = df.sort_values(["time", "ID"]).reset_index(drop=True)

    stations = sorted(df["station_name"].dropna().unique().tolist())
    map_df = pd.read_csv(station_map_csv)
    mapped_names = set(map_df["station"].astype(str))
    spatial_stations = [s for s in stations if s in mapped_names]
    if len(spatial_stations) < 2:
        spatial_stations = stations

    adj, n_edges = build_adjacency(spatial_stations, graph_csv, station_map_csv)
    print(
        f"stations={len(spatial_stations)} edges={n_edges} "
        f"seq_len={seq_len} horizon={args.horizon}"
    )

    manifest = []
    for level in args.feature_configs:
        meta = export_level(
            df,
            level=level,
            zone=args.zone,
            contaminant=args.contaminant,
            spatial_stations=spatial_stations,
            adj=adj,
            seq_len=seq_len,
            horizon=args.horizon,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            min_availability=args.min_availability,
            out_root=out_root,
            seq_len_subdir=seq_len_subdir,
            horizon_subdir=horizon_subdir,
        )
        if meta:
            manifest.append(meta)
            print(f"  wrote {meta['path']}")

    print(f"Done. GAT-MF exports: {len(manifest)}")


if __name__ == "__main__":
    main()
