"""
Train / evaluate AirFormer on zone-2 PM10 packs (F1–F5).

One multi-node model per feature config (18 stations jointly).
Writes hourly + MA24 metrics under results/airformer/...
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.build_airformer_zone2_dartboard import build_zone2_dartboard
from src.data.feature_configs import ALL_FEATURE_LEVELS
from src.models.informer2020.metrics import (
    metric_hourly_and_ma24,
    moving_average_first_24h,
    moving_average_second_24h,
)
from src.utils.paths import (
    airformer_data_dir,
    airformer_results_dir,
    airformer_upstream,
)

# Approximate dims when metadata.json is missing (PM10 zone-2 after availability filter)
INPUT_DIM_FALLBACK = {
    "F1": 1,
    "F2": 7,
    "F3": 13,
    "F3S": 14,
    "F3I": 14,
    "F3IL": 19,
    "F4": 15,
    "F5": 16,
    "F5I": 16,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AirFormer zone-2 experiment")
    p.add_argument("--zone", type=int, default=2)
    p.add_argument("--dataset", default="ZONE2_PM10")
    p.add_argument(
        "--feature-config",
        default="F1",
        choices=list(ALL_FEATURE_LEVELS),
    )
    p.add_argument(
        "--results-tag",
        default="",
        help="Append to results folder (e.g. _pw) without changing data pack",
    )
    p.add_argument("--seq-len", type=int, default=48)
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--n-hidden", type=int, default=32)
    p.add_argument("--num-heads", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--base-lr", type=float, default=5e-4)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n-exp", type=int, default=0)
    p.add_argument(
        "--stochastic-flag",
        default="False",
        help="False recommended for ZONE2 (F1 has input_dim=1)",
    )
    p.add_argument("--spatial-flag", default="True")
    p.add_argument(
        "--dartboard",
        type=int,
        default=4,
        help="4 = zone2_18 trivial partition (N=18)",
    )
    p.add_argument("--mode", default="train", choices=["train", "test"])
    p.add_argument(
        "--peak-weight-alpha",
        type=float,
        default=0.0,
        help="Train-only peak emphasis (0=off). Typical: 1.0",
    )
    p.add_argument("--peak-weight-thr", type=float, default=40.0)
    p.add_argument("--peak-weight-thr2", type=float, default=80.0)
    p.add_argument(
        "--peak-asym-beta",
        type=float,
        default=0.0,
        help="Train-only under-prediction penalty. Typical: 0.5",
    )
    p.add_argument(
        "--train-loss",
        default="mae",
        choices=["mae", "mse"],
        help="Upstream masked loss (default mae; use experiment_airformer_mse.py for MSE ablation)",
    )
    return p.parse_args()


def resolve_input_dim(data_dir: Path, feature_config: str) -> int:
    meta_path = data_dir / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        feats = meta.get("features") or []
        if feats:
            return len(feats)
        shape = meta.get("x_shape")
        if shape and len(shape) == 4:
            return int(shape[-1])
    return INPUT_DIM_FALLBACK[feature_config]


def per_station_metrics(
    preds: np.ndarray,
    trues: np.ndarray,
    stations: list[str],
) -> list[dict]:
    """preds/trues: (samples, horizon, nodes, 1)."""
    rows = []
    n_nodes = preds.shape[2]
    for i in range(n_nodes):
        name = stations[i] if i < len(stations) else f"node_{i}"
        m = metric_hourly_and_ma24(preds[:, :, i : i + 1, :], trues[:, :, i : i + 1, :])
        row = {
            "station": name,
            "hourly_mae": m["hourly"]["mae"],
            "hourly_rmse": m["hourly"]["rmse"],
            "ma24_mae": m["ma24"]["mae"],
            "ma24_rmse": m["ma24"]["rmse"],
        }
        if "ma24_d1" in m:
            row["ma24_d1_rmse"] = m["ma24_d1"]["rmse"]
            row["ma24_d2_rmse"] = m["ma24_d2"]["rmse"]
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    upstream = airformer_upstream()
    data_dir = airformer_data_dir(
        args.zone,
        args.feature_config,
        args.dataset,
        seq_len=args.seq_len,
        horizon=args.horizon,
    )
    if not (data_dir / "train.npz").exists():
        raise SystemExit(f"Missing AirFormer pack: {data_dir}")

    adj_path = data_dir / "adj_mx.pkl"
    build_zone2_dartboard(adj_path=adj_path)

    out_dir = airformer_results_dir(
        args.zone,
        args.dataset,
        args.feature_config,
        seq_len=args.seq_len,
        horizon=args.horizon,
    )
    tag = (args.results_tag or "").strip()
    if tag:
        if not tag.startswith("_"):
            tag = "_" + tag
        out_dir = out_dir.parent / f"{out_dir.name}{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    input_dim = resolve_input_dim(data_dir, args.feature_config)
    meta = {}
    meta_path = data_dir / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    stations = meta.get("stations") or [f"node_{i}" for i in range(18)]

    # Upstream resolves relative paths from its own cwd
    cmd = [
        sys.executable,
        "-u",
        str(upstream / "experiments" / "airformer" / "main.py"),
        "--mode",
        args.mode,
        "--dataset",
        args.dataset,
        "--datapath",
        str(data_dir.resolve()),
        "--graph_pkl",
        str(adj_path.resolve()),
        "--log_dir",
        str(log_dir.resolve()) + os.sep,
        "--seq_len",
        str(args.seq_len),
        "--horizon",
        str(args.horizon),
        "--input_dim",
        str(input_dim),
        "--output_dim",
        "1",
        "--batch_size",
        str(args.batch_size),
        "--max_epochs",
        str(args.max_epochs),
        "--patience",
        str(args.patience),
        "--n_hidden",
        str(args.n_hidden),
        "--num_heads",
        str(args.num_heads),
        "--dropout",
        str(args.dropout),
        "--base_lr",
        str(args.base_lr),
        "--gpu",
        str(args.gpu),
        "--n_exp",
        str(args.n_exp),
        "--stochastic_flag",
        str(args.stochastic_flag),
        "--spatial_flag",
        str(args.spatial_flag),
        "--dartboard",
        str(args.dartboard),
        "--peak_weight_alpha",
        str(args.peak_weight_alpha),
        "--peak_weight_thr",
        str(args.peak_weight_thr),
        "--peak_weight_thr2",
        str(args.peak_weight_thr2),
        "--peak_asym_beta",
        str(args.peak_asym_beta),
        "--train_loss",
        str(args.train_loss),
    ]

    env = os.environ.copy()
    # Upstream `src` must win over MASTER `src`
    env["PYTHONPATH"] = str(upstream.resolve()) + os.pathsep + env.get("PYTHONPATH", "")

    print("AirFormer |", args.dataset, args.feature_config, f"input_dim={input_dim}")
    print("datapath :", data_dir)
    print("results  :", out_dir)
    print(" ".join(cmd))

    subprocess.run(cmd, check=True, cwd=str(upstream), env=env)

    preds_path = log_dir / f"test_preds.npy"
    labels_path = log_dir / f"test_labels.npy"
    # Trainer saves without n_exp suffix for arrays
    if not preds_path.exists():
        # fallback search
        cand = list(log_dir.glob("**/test_preds.npy"))
        if cand:
            preds_path = cand[0]
            labels_path = preds_path.with_name("test_labels.npy")

    if not preds_path.exists():
        raise SystemExit(
            f"Training finished but test_preds.npy not found under {log_dir}"
        )

    preds = np.load(preds_path)
    trues = np.load(labels_path)
    # Ensure (B, H, N, C)
    if preds.ndim == 3:
        preds = preds[..., None]
        trues = trues[..., None]

    overall = metric_hourly_and_ma24(preds, trues)
    by_station = per_station_metrics(preds, trues, stations)

    np.save(out_dir / "test_preds_hourly.npy", preds)
    np.save(out_dir / "test_trues_hourly.npy", trues)
    if preds.shape[1] >= 48:
        np.save(out_dir / "test_preds_ma24.npy", moving_average_first_24h(preds))
        np.save(out_dir / "test_trues_ma24.npy", moving_average_first_24h(trues))
        np.save(out_dir / "test_preds_ma24_d2.npy", moving_average_second_24h(preds))
        np.save(out_dir / "test_trues_ma24_d2.npy", moving_average_second_24h(trues))
    else:
        np.save(out_dir / "test_preds_ma24.npy", preds.mean(axis=1, keepdims=True))
        np.save(out_dir / "test_trues_ma24.npy", trues.mean(axis=1, keepdims=True))

    # Copy upstream metrics csv if present
    metrics_csv = log_dir / f"metrics_{args.n_exp}.csv"
    if metrics_csv.exists():
        shutil.copy2(metrics_csv, out_dir / "upstream_metrics.csv")

    results = {
        "model": "airformer",
        "dataset": args.dataset,
        "zone": args.zone,
        "feature_config": args.feature_config,
        "input_dim": input_dim,
        "seq_len": args.seq_len,
        "horizon": args.horizon,
        "num_nodes": int(preds.shape[2]),
        "stations": stations,
        "forecast_design": {
            "at_hour_h": "model outputs hours h+1 ... h+horizon for all nodes",
            "ma24": (
                "mean(pred[h+1], ..., pred[h+24]); at horizon>=48 also ma24_d2 "
                "(h+25..h+48) in test_metrics_raw"
            ),
        },
        "stochastic_flag": args.stochastic_flag,
        "spatial_flag": args.spatial_flag,
        "dartboard": args.dartboard,
        "results_tag": args.results_tag or "",
        "train_loss": args.train_loss,
        "peak_weight": {
            "alpha": args.peak_weight_alpha,
            "thr": args.peak_weight_thr,
            "thr2": args.peak_weight_thr2,
            "asym_beta": args.peak_asym_beta,
        },
        "test_metrics_raw": {
            "hourly": overall["hourly"],
            "ma24": overall["ma24"],
            **(
                {"ma24_d1": overall["ma24_d1"], "ma24_d2": overall["ma24_d2"]}
                if "ma24_d1" in overall
                else {}
            ),
        },
        "per_station": by_station,
        "artifacts": {
            "test_preds_hourly": str((out_dir / "test_preds_hourly.npy").as_posix()),
            "test_trues_hourly": str((out_dir / "test_trues_hourly.npy").as_posix()),
            "test_preds_ma24": str((out_dir / "test_preds_ma24.npy").as_posix()),
            "test_trues_ma24": str((out_dir / "test_trues_ma24.npy").as_posix()),
            "checkpoint_dir": str(log_dir.as_posix()),
        },
    }
    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Compact CSV for display / summaries
    import csv

    summary_csv = out_dir / "per_station_summary.csv"
    fieldnames = [
        "station",
        "hourly_mae",
        "hourly_rmse",
        "ma24_mae",
        "ma24_rmse",
    ]
    if by_station and "ma24_d1_rmse" in by_station[0]:
        fieldnames.extend(["ma24_d1_rmse", "ma24_d2_rmse"])
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(by_station)

    print(
        f"\nOverall hourly MAE={overall['hourly']['mae']:.4f} "
        f"RMSE={overall['hourly']['rmse']:.4f}"
    )
    print(
        f"Overall MA24   MAE={overall['ma24']['mae']:.4f} "
        f"RMSE={overall['ma24']['rmse']:.4f}"
    )
    if "ma24_d2" in overall:
        print(
            f"Overall MA24_D2 MAE={overall['ma24_d2']['mae']:.4f} "
            f"RMSE={overall['ma24_d2']['rmse']:.4f}"
        )
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
