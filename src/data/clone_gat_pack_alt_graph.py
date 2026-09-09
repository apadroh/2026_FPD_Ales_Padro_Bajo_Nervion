"""
Clone a GAT-MF NPZ pack and replace only the adjacency (`graph`) with another
CSV (e.g. PM10 top-5). X/Y/scales stay identical → fair Adj ablation.

Usage:
  python src/data/clone_gat_pack_alt_graph.py \\
    --src-config F1 --graph-csv data/graphs/pm10_graph_zone_2_top5.csv \\
    --tag pm10graph
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.export_airformer_zone2 import build_adjacency
from src.utils.paths import TRAINING_ROOT, apply_layout_tags


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clone GAT-MF pack with alternate graph")
    p.add_argument("--zone", type=int, default=2)
    p.add_argument("--src-config", default="F1")
    p.add_argument(
        "--src-model",
        default="gat_informer_mf",
        help="Source pack folder under data/training/zone_{z}/",
    )
    p.add_argument(
        "--tag",
        default="pm10graph",
        help="Dest folder suffix: {src_model}_{tag}",
    )
    p.add_argument("--graph-csv", required=True)
    p.add_argument(
        "--station-map-csv",
        default=str(ROOT / "data/metadata/stations_meteo_graph_check.csv"),
    )
    p.add_argument("--seq-len", type=int, default=48)
    p.add_argument("--horizon", type=int, default=24)
    return p.parse_args()


def _pack_dir(zone: int, model: str, config: str, *, seq_len: int, horizon: int) -> Path:
    base = apply_layout_tags(
        TRAINING_ROOT / f"zone_{zone}" / model,
        seq_len=seq_len,
        horizon=horizon,
    )
    return base / config


def main() -> None:
    args = parse_args()
    src_dir = _pack_dir(
        args.zone, args.src_model, args.src_config,
        seq_len=args.seq_len, horizon=args.horizon,
    )
    npz_src = src_dir / f"data{args.seq_len}.npz"
    meta_src = src_dir / "metadata.json"
    if not npz_src.exists():
        raise SystemExit(f"Missing source pack: {npz_src}")
    if not meta_src.exists():
        raise SystemExit(f"Missing metadata: {meta_src}")

    meta = json.loads(meta_src.read_text(encoding="utf-8"))
    stations = list(meta["stations"])
    graph_csv = Path(args.graph_csv)
    if not graph_csv.exists():
        raise SystemExit(f"Missing graph CSV: {graph_csv}")

    adj, n_edges = build_adjacency(
        stations, graph_csv, Path(args.station_map_csv)
    )
    print(f"stations={len(stations)} edges_used={n_edges} adj_nonzero_offdiag="
          f"{int(((adj > 0) & ~np.eye(len(stations), dtype=bool)).sum())}")

    dest_model = f"{args.src_model}_{args.tag}"
    dest_dir = _pack_dir(
        args.zone, dest_model, args.src_config,
        seq_len=args.seq_len, horizon=args.horizon,
    )
    dest_dir.mkdir(parents=True, exist_ok=True)

    raw = dict(np.load(npz_src, allow_pickle=True))
    old = np.asarray(raw["graph"])
    raw["graph"] = adj.astype(np.float32)
    npz_dst = dest_dir / f"data{args.seq_len}.npz"
    np.savez_compressed(npz_dst, **raw)

    meta_out = dict(meta)
    meta_out["graph_csv"] = str(graph_csv.as_posix())
    meta_out["graph_tag"] = args.tag
    meta_out["graph_note"] = (
        "Adjacency replaced; X/Y identical to source pack for Adj ablation."
    )
    meta_out["graph_n_edges_build_adjacency"] = int(n_edges)
    meta_out["source_pack"] = str(src_dir.as_posix())
    (dest_dir / "metadata.json").write_text(
        json.dumps(meta_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # optional: copy any other small files
    for name in ("feature_scaler.json",):
        s = src_dir / name
        if s.exists():
            shutil.copy2(s, dest_dir / name)

    delta = float(np.abs(adj - old).sum())
    print(f"wrote {npz_dst}")
    print(f"wrote {dest_dir / 'metadata.json'}")
    print(f"|Adj_new - Adj_old|_1 = {delta:.4f} (0 means identical)")


if __name__ == "__main__":
    main()
