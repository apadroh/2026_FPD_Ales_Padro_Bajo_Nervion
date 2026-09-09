"""
Train Informer2020 on all stations for a feature config (e.g. F1).

Designed for local or HPC GPU runs. Skips stations that already
have results.json unless --force is set.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.feature_configs import ABLATION_LEVELS
from src.utils.paths import (
    TRAINING_ROOT,
    apply_layout_tags,
    informer2020_results_dir,
)
from src.utils.stations import dedupe_station_names


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch Informer2020 over stations")
    p.add_argument("--zone", type=int, default=2)
    p.add_argument("--contaminant", default="PM10")
    p.add_argument("--feature-config", default="F1", choices=ABLATION_LEVELS)
    p.add_argument(
        "--stations",
        nargs="*",
        default=None,
        help="Subset of station folder names; default = all with data.csv",
    )
    p.add_argument("--seq-len", type=int, default=48)
    p.add_argument("--label-len", type=int, default=24)
    p.add_argument("--pred-len", type=int, default=24)
    p.add_argument(
        "--features",
        default=None,
        choices=["S", "M", "MS"],
        help="Override Informer features mode (default: S for F1, else MS)",
    )
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--d-ff", type=int, default=2048)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--e-layers", type=int, default=2)
    p.add_argument("--d-layers", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--train-epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--force", action="store_true", help="Retrain even if results exist")
    p.add_argument(
        "--preset",
        choices=["paper", "fast"],
        default="paper",
        help="paper=Informer defaults; fast=lighter (d_model=128) for CPU smoke tests",
    )
    return p.parse_args()


def discover_stations(zone: int, feature_config: str) -> list[str]:
    base = TRAINING_ROOT / f"zone_{zone}" / "informer2020"
    if not base.exists():
        return []
    stations = []
    for d in sorted(base.iterdir()):
        if (d / feature_config / "data.csv").exists():
            stations.append(d.name)
    return dedupe_station_names(stations)


def default_features_mode(feature_config: str, override: str | None) -> str:
    if override:
        return override
    # F1 packs only the target column → univariate Informer mode
    return "S" if feature_config == "F1" else "MS"


def apply_preset(args: argparse.Namespace) -> None:
    if args.preset == "fast":
        args.d_model = 128
        args.d_ff = 256
        args.n_heads = 4


def main() -> None:
    args = parse_args()
    apply_preset(args)
    features_mode = default_features_mode(args.feature_config, args.features)

    stations = args.stations or discover_stations(args.zone, args.feature_config)
    if not stations:
        raise SystemExit(
            f"No stations found for zone={args.zone} feature_config={args.feature_config}"
        )

    summary_rows: list[dict] = []
    zone_base = apply_layout_tags(
        ROOT / "results" / "informer2020" / f"zone_{args.zone}",
        seq_len=args.seq_len,
        horizon=args.pred_len,
    )
    summary_path = zone_base / f"batch_summary_{args.feature_config}.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Batch Informer2020 | zone={args.zone} | {args.feature_config} | "
        f"seq_len={args.seq_len} | features={features_mode} | "
        f"preset={args.preset} | n={len(stations)}"
    )
    print(f"Stations: {', '.join(stations)}")

    for i, station in enumerate(stations, 1):
        out_dir = informer2020_results_dir(
            args.zone,
            station,
            args.feature_config,
            seq_len=args.seq_len,
            horizon=args.pred_len,
        )
        results_json = out_dir / "results.json"
        if results_json.exists() and not args.force:
            print(f"[{i}/{len(stations)}] SKIP {station} (results exist)")
            with open(results_json, encoding="utf-8") as f:
                prev = json.load(f)
            raw = prev.get("test_metrics_raw", {})
            summary_rows.append(
                {
                    "station": station,
                    "status": "skipped",
                    "hourly_mae": raw.get("hourly", {}).get("mae"),
                    "hourly_rmse": raw.get("hourly", {}).get("rmse"),
                    "ma24_mae": raw.get("ma24", {}).get("mae"),
                    "ma24_rmse": raw.get("ma24", {}).get("rmse"),
                    "elapsed_s": None,
                }
            )
            continue

        cmd = [
            sys.executable,
            "-u",
            str(ROOT / "src/experiments/experiment_informer2020.py"),
            "--station",
            station,
            "--zone",
            str(args.zone),
            "--contaminant",
            args.contaminant,
            "--feature-config",
            args.feature_config,
            "--features",
            features_mode,
            "--seq-len",
            str(args.seq_len),
            "--label-len",
            str(args.label_len),
            "--pred-len",
            str(args.pred_len),
            "--d-model",
            str(args.d_model),
            "--d-ff",
            str(args.d_ff),
            "--n-heads",
            str(args.n_heads),
            "--e-layers",
            str(args.e_layers),
            "--d-layers",
            str(args.d_layers),
            "--batch-size",
            str(args.batch_size),
            "--train-epochs",
            str(args.train_epochs),
            "--patience",
            str(args.patience),
            "--learning-rate",
            str(args.learning_rate),
        ]
        print(f"\n[{i}/{len(stations)}] TRAIN {station}")
        t0 = time.time()
        try:
            subprocess.run(cmd, check=True, cwd=ROOT)
            status = "ok"
            err = None
        except subprocess.CalledProcessError as e:
            status = "failed"
            err = str(e)
            print(f"FAILED {station}: {e}")

        elapsed = round(time.time() - t0, 1)
        hourly_mae = hourly_rmse = ma24_mae = ma24_rmse = None
        if (out_dir / "results.json").exists():
            with open(out_dir / "results.json", encoding="utf-8") as f:
                res = json.load(f)
            raw = res.get("test_metrics_raw", {})
            hourly_mae = raw.get("hourly", {}).get("mae")
            hourly_rmse = raw.get("hourly", {}).get("rmse")
            ma24_mae = raw.get("ma24", {}).get("mae")
            ma24_rmse = raw.get("ma24", {}).get("rmse")

        summary_rows.append(
            {
                "station": station,
                "status": status,
                "hourly_mae": hourly_mae,
                "hourly_rmse": hourly_rmse,
                "ma24_mae": ma24_mae,
                "ma24_rmse": ma24_rmse,
                "elapsed_s": elapsed,
                "error": err,
            }
        )

        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "station",
                    "status",
                    "hourly_mae",
                    "hourly_rmse",
                    "ma24_mae",
                    "ma24_rmse",
                    "elapsed_s",
                    "error",
                ],
            )
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"\nSummary written to {summary_path}")
    ok = sum(1 for r in summary_rows if r["status"] in {"ok", "skipped"})
    failed = sum(1 for r in summary_rows if r["status"] == "failed")
    print(f"Done: {ok} ok/skipped, {failed} failed / {len(stations)} total")


if __name__ == "__main__":
    main()
