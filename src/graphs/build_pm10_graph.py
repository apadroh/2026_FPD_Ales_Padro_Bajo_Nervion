"""
Build a PM10 similarity graph (Pearson) for zone stations, then top-k sparsify.

Methodological note: correlations use only the **train** temporal slice
(default first 70% of timestamps), matching the GAT export split, so the
graph does not peek at val/test.

Outputs (zone 2 default):
  data/graphs/pm10_graph_zone_{z}.csv
  data/graphs/pm10_graph_zone_{z}_top5.csv

IDs are the same S* codes as the meteo graph (stations_meteo_graph_check.csv),
so build_adjacency() works unchanged.

Usage:
  python src/graphs/build_pm10_graph.py --zone 2 --k 5
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

from src.data.feature_configs import default_dataset_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PM10 Pearson graph + top-k")
    p.add_argument("--zone", type=int, default=2)
    p.add_argument("--contaminant", default="PM10")
    p.add_argument("--dataset", default=None)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--min-obs", type=int, default=200, help="Min paired hours for corr")
    p.add_argument(
        "--stations-from-npz-meta",
        default=None,
        help="Optional metadata.json from a GAT pack to fix station order/set",
    )
    p.add_argument(
        "--station-map-csv",
        default=str(ROOT / "data/metadata/stations_meteo_graph_check.csv"),
    )
    p.add_argument("--out-dir", default=str(ROOT / "data/graphs"))
    return p.parse_args()


def _load_station_ids(map_csv: Path) -> tuple[dict[str, str], dict[str, str]]:
    m = pd.read_csv(map_csv)
    name_to_id = {
        str(r["station"]): str(r["id"]) for _, r in m.iterrows()
    }
    id_to_name = {v: k for k, v in name_to_id.items()}
    return name_to_id, id_to_name


def _resolve_stations(
    wide: pd.DataFrame,
    name_to_id: dict[str, str],
    meta_path: Path | None,
) -> list[str]:
    if meta_path and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        stations = list(meta.get("stations") or [])
        missing = [s for s in stations if s not in wide.columns]
        if missing:
            raise SystemExit(
                f"Stations in NPZ meta missing from dataset wide PM10: {missing[:5]}…"
            )
        return stations
    # Prefer stations that have an S* id (aligned with GAT exports)
    cols = [c for c in wide.columns if c in name_to_id]
    if len(cols) < 2:
        cols = list(wide.columns)
    return sorted(cols)


def build_pm10_corr_edges(
    wide_train: pd.DataFrame,
    stations: list[str],
    name_to_id: dict[str, str],
    min_obs: int,
) -> pd.DataFrame:
    rows = []
    for i, a in enumerate(stations):
        for b in stations[i + 1 :]:
            pair = wide_train[[a, b]].dropna()
            if len(pair) < min_obs:
                continue
            corr = float(pair[a].corr(pair[b]))
            if not np.isfinite(corr):
                continue
            # Use |corr| as weight (anti-correlated still "related")
            w = abs(corr)
            id_a = name_to_id.get(a)
            id_b = name_to_id.get(b)
            if id_a is None or id_b is None:
                continue
            rows.append({"source": id_a, "target": id_b, "weight": w})
    return pd.DataFrame(rows, columns=["source", "target", "weight"])


def topk_undirected(edges: pd.DataFrame, k: int) -> pd.DataFrame:
    if edges.empty:
        return edges.copy()
    both = pd.concat(
        [
            edges,
            edges.rename(columns={"source": "target", "target": "source"}),
        ],
        ignore_index=True,
    )
    parts = []
    for node, g in both.groupby("source"):
        parts.append(g.nlargest(k, "weight"))
    out = pd.concat(parts, ignore_index=True)
    # Dedup undirected for a cleaner file: keep one direction lexically
    a = out["source"].astype(str)
    b = out["target"].astype(str)
    swap = a > b
    out.loc[swap, ["source", "target"]] = out.loc[swap, ["target", "source"]].values
    out = out.sort_values("weight", ascending=False).drop_duplicates(
        subset=["source", "target"], keep="first"
    )
    return out.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    dataset = Path(args.dataset) if args.dataset else ROOT / default_dataset_path(
        args.zone, args.contaminant
    )
    if not dataset.exists():
        raise SystemExit(f"Dataset not found: {dataset}")

    map_csv = Path(args.station_map_csv)
    name_to_id, _ = _load_station_ids(map_csv)

    print(f"Loading {dataset} …")
    df = pd.read_csv(dataset, parse_dates=["time"], low_memory=False)
    if args.contaminant not in df.columns:
        raise SystemExit(f"Missing column {args.contaminant} in {dataset}")

    wide = (
        df.pivot_table(
            index="time",
            columns="station_name",
            values=args.contaminant,
            aggfunc="mean",
        )
        .sort_index()
    )

    meta_path = Path(args.stations_from_npz_meta) if args.stations_from_npz_meta else (
        ROOT
        / f"data/training/zone_{args.zone}/gat_informer_mf/F1/metadata.json"
    )
    stations = _resolve_stations(
        wide, name_to_id, meta_path if meta_path.exists() else None
    )
    wide = wide.reindex(columns=stations)

    times = wide.index
    n_train = max(1, int(len(times) * args.train_ratio))
    wide_train = wide.iloc[:n_train]
    print(
        f"stations={len(stations)} times={len(times)} "
        f"train_times={n_train} ({args.train_ratio:.0%}) min_obs={args.min_obs}"
    )

    edges = build_pm10_corr_edges(wide_train, stations, name_to_id, args.min_obs)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    full_path = out_dir / f"pm10_graph_zone_{args.zone}.csv"
    top_path = out_dir / f"pm10_graph_zone_{args.zone}_top{args.k}.csv"
    edges.to_csv(full_path, index=False)
    top = topk_undirected(edges, args.k)
    # Keep bidirectional listing like meteo top5 (build_adjacency maxes both dirs)
    top_bi = pd.concat(
        [top, top.rename(columns={"source": "target", "target": "source"})],
        ignore_index=True,
    ).drop_duplicates(subset=["source", "target"])
    top_bi.to_csv(top_path, index=False)

    print(f"wrote {full_path} ({len(edges)} undirected pairs)")
    print(f"wrote {top_path} ({len(top_bi)} directed rows, k={args.k})")
    if not edges.empty:
        print(
            f"weight mean={edges['weight'].mean():.3f} "
            f"max={edges['weight'].max():.3f}"
        )


if __name__ == "__main__":
    main()
