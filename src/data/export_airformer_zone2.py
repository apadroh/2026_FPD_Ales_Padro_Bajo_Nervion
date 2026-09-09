"""
Export AirFormer zone packs: train/val/test.npz + adj_mx.pkl + metadata.json.

Mirrors the AirFormer cells in prepare_training_datasets.ipynb as a CLI so
HPC ablation (F1–F5) does not depend on the notebook.

Usage:
  python src/data/export_airformer_zone2.py --feature-configs F2 F4
  python src/data/export_airformer_zone2.py --feature-configs F1 F2 F3 F4 F5
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.feature_configs import (
    ALL_FEATURE_LEVELS,
    default_dataset_path,
    resolve_features_in_dataset,
)
from src.data.saharan_intensity import add_sah_int_context_features


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export AirFormer ZONE2 packs")
    p.add_argument("--zone", type=int, default=2)
    p.add_argument("--contaminant", default="PM10")
    p.add_argument(
        "--feature-configs",
        nargs="+",
        default=["F2", "F4"],
        choices=ALL_FEATURE_LEVELS,
    )
    p.add_argument("--dataset", default=None, help="Override zone CSV path")
    p.add_argument("--start-date", default="2019-01-01")
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.10)
    p.add_argument("--min-availability", type=float, default=80.0)
    p.add_argument("--interpolate-limit", type=int, default=3)
    p.add_argument(
        "--graph-csv",
        default=None,
        help="Default: data/graphs/meteo_graph_zone_{ZONE}_top5.csv",
    )
    p.add_argument(
        "--station-map-csv",
        default=None,
        help="Default: data/metadata/stations_meteo_graph_check.csv",
    )
    p.add_argument("--output-root", default="data/training")
    return p.parse_args()


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"])
    out["hour"] = out["time"].dt.hour
    out["dow"] = out["time"].dt.dayofweek
    out["month"] = out["time"].dt.month
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    if "DV" in out.columns:
        out["wind_sin"] = np.sin(np.deg2rad(out["DV"]))
        out["wind_cos"] = np.cos(np.deg2rad(out["DV"]))
    if "Sah_int" in out.columns:
        out = add_sah_int_context_features(out)
    return out


def interpolate_numeric(df: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
    out = df.copy()
    skip = {"time", "ID", "station_name", "lat", "lon", "zone", "hour", "dow", "month"}
    cols = [
        c
        for c in out.columns
        if c not in skip and pd.api.types.is_numeric_dtype(out[c])
    ]
    out[cols] = out.groupby("ID")[cols].transform(lambda s: s.interpolate(limit=limit))
    return out


def station_key(name: str) -> str:
    return name.replace(" ", "_").replace("(", "").replace(")", "")


def temporal_split_indices(
    n_samples: int, train_ratio: float, val_ratio: float
) -> tuple[int, int, int]:
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    n_test = n_samples - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError(f"Split too small for n_samples={n_samples}")
    return n_train, n_val, n_test


def build_adjacency(
    stations_ordered: list[str], graph_csv: Path, map_csv: Path
) -> tuple[np.ndarray, int]:
    if not graph_csv.exists():
        raise FileNotFoundError(graph_csv)
    if not map_csv.exists():
        raise FileNotFoundError(map_csv)

    id_map = pd.read_csv(map_csv)
    sid_to_name = dict(zip(id_map["id"].astype(str), id_map["station"].astype(str)))
    name_to_idx = {n: i for i, n in enumerate(stations_ordered)}

    edges = pd.read_csv(graph_csv)
    n = len(stations_ordered)
    adj = np.zeros((n, n), dtype=np.float64)
    used_edges = 0
    for _, row in edges.iterrows():
        s_name = sid_to_name.get(str(row["source"]))
        t_name = sid_to_name.get(str(row["target"]))
        if s_name not in name_to_idx or t_name not in name_to_idx:
            continue
        i, j = name_to_idx[s_name], name_to_idx[t_name]
        w = float(row["weight"])
        adj[i, j] = max(adj[i, j], w)
        adj[j, i] = max(adj[j, i], w)
        used_edges += 1

    np.fill_diagonal(adj, 1.0)
    return adj, used_edges


def pivot_feature_cube(
    df_in: pd.DataFrame, stations_ordered: list[str], feature_cols: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Return times and array (T, N, F) aligned on common timestamps."""
    pieces = []
    for col in feature_cols:
        wide = (
            df_in.pivot_table(
                index="time", columns="station_name", values=col, aggfunc="mean"
            )
            .reindex(columns=stations_ordered)
            .sort_index()
        )
        pieces.append(wide)
    times = pieces[0].index
    for p in pieces[1:]:
        times = times.intersection(p.index)
    mats = [p.reindex(index=times).to_numpy(dtype=np.float32) for p in pieces]
    cube = np.stack(mats, axis=-1)
    return times.to_numpy(), cube


def fill_cube_nans(cube: np.ndarray, target_idx: int = 0) -> np.ndarray:
    out = cube.copy()
    _t, n, f = out.shape
    for j in range(n):
        for k in range(f):
            s = pd.Series(out[:, j, k])
            s = s.ffill().bfill()
            out[:, j, k] = s.to_numpy(dtype=np.float32)
            if k != target_idx:
                out[:, j, k] = np.nan_to_num(out[:, j, k], nan=0.0)
    return out


def make_st_windows(
    cube: np.ndarray, seq_len: int, horizon: int, target_idx: int
) -> tuple[np.ndarray, np.ndarray]:
    """cube (T,N,F) -> X (S,seq,N,F), Y (S,horizon,N,1)"""
    cube = fill_cube_nans(cube, target_idx=target_idx)
    t, n, f = cube.shape
    xs, ys = [], []
    max_start = t - seq_len - horizon + 1
    for i in range(max_start):
        x = cube[i : i + seq_len]
        y = cube[i + seq_len : i + seq_len + horizon, :, target_idx : target_idx + 1]
        if np.isnan(x[:, :, target_idx]).any() or np.isnan(y).any():
            continue
        x = np.nan_to_num(x, nan=0.0)
        xs.append(x)
        ys.append(y)
    if not xs:
        return (
            np.empty((0, seq_len, n, f), dtype=np.float32),
            np.empty((0, horizon, n, 1), dtype=np.float32),
        )
    return np.stack(xs), np.stack(ys)


def resolve_seq_len(zone: int, contaminant: str, explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    acf = (
        ROOT
        / "analysis"
        / "acf"
        / f"zone_{zone}"
        / contaminant
        / "seq_len_recommendation.json"
    )
    if acf.exists():
        return int(json.loads(acf.read_text(encoding="utf-8"))["recommended_seq_len"])
    return 48


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
    if contaminant not in feature_cols:
        raise ValueError(f"AirFormer {level}: missing target {contaminant}")

    ordered = [contaminant] + [c for c in feature_cols if c != contaminant]
    _times, cube = pivot_feature_cube(df, spatial_stations, ordered)
    print(f"AirFormer {level}: cube {cube.shape} (T,N,F) features={ordered}")

    X, Y = make_st_windows(cube, seq_len, horizon, target_idx=0)
    print(f"  windows: X={X.shape} Y={Y.shape}")
    if len(X) == 0:
        print(f"  SKIP {level}: no complete windows")
        return None

    n_train, n_val, n_test = temporal_split_indices(len(X), train_ratio, val_ratio)
    splits = {
        "train": (X[:n_train], Y[:n_train]),
        "val": (X[n_train : n_train + n_val], Y[n_train : n_train + n_val]),
        "test": (X[n_train + n_val :], Y[n_train + n_val :]),
    }

    ds_name = f"ZONE{zone}_{contaminant}"
    base = out_root / f"zone_{zone}" / "airformer" / ds_name
    if seq_len_subdir:
        base = base / f"seq_len_{seq_len}"
    if horizon_subdir:
        base = base / f"horizon_{horizon}"
    out_dir = base / level
    out_dir.mkdir(parents=True, exist_ok=True)

    for cat, (x_cat, y_cat) in splits.items():
        np.savez_compressed(out_dir / f"{cat}.npz", x=x_cat, y=y_cat)

    sensor_ids = [station_key(s) for s in spatial_stations]
    sensor_id_to_ind = {sid: i for i, sid in enumerate(sensor_ids)}
    pkl_path = out_dir / "adj_mx.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(
            (sensor_ids, sensor_id_to_ind, adj.astype(np.float32)), f, protocol=4
        )

    meta = {
        "model": "airformer",
        "dataset_name": ds_name,
        "zone": zone,
        "contaminant": contaminant,
        "feature_config": level,
        "features": ordered,
        "features_excluded": excluded,
        "stations": spatial_stations,
        "num_nodes": len(spatial_stations),
        "seq_len": seq_len,
        "horizon": horizon,
        "n_train": int(n_train),
        "n_val": int(n_val),
        "n_test": int(n_test),
        "x_shape": list(X.shape),
        "y_shape": list(Y.shape),
        "path": str(out_dir.relative_to(ROOT)).replace("\\", "/"),
        "adj_path": str(pkl_path.relative_to(ROOT)).replace("\\", "/"),
        "note": "Register dataset in AirFormer get_num_nodes before training.",
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
        print("WARNING: few stations matched graph map; using all dataset stations.")

    adj, n_edges = build_adjacency(spatial_stations, graph_csv, station_map_csv)
    print(
        f"stations={len(spatial_stations)} edges={n_edges} "
        f"seq_len={seq_len} horizon={args.horizon} "
        f"subdir={('seq_len_'+str(seq_len) if seq_len_subdir else 'baseline')}"
        f"{(' horizon_'+str(args.horizon) if horizon_subdir else '')}"
    )
    print(f"configs={args.feature_configs}")

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
            print(
                f"  wrote {meta['path']}  "
                f"input_dim={len(meta['features'])}  "
                f"n_train={meta['n_train']}"
            )

    print(f"Done. AirFormer exports: {len(manifest)}")
    for m in manifest:
        print(f"  {m['feature_config']}: dim={len(m['features'])} -> {m['path']}")


if __name__ == "__main__":
    main()
