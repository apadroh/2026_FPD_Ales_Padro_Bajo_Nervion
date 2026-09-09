"""
Regenerate Informer2020 plots from existing results folders.

Usage:
  python src/experiments/plot_informer2020_results.py --zone 2 --feature-config F5
  python src/experiments/plot_informer2020_results.py --zone 2 --all-configs
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

from src.data.feature_configs import ABLATION_LEVELS
from src.models.informer2020.plots import save_run_plots
from src.utils.paths import informer2020_station_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--zone", type=int, default=2)
    p.add_argument("--feature-config", default="F5", choices=ABLATION_LEVELS)
    p.add_argument(
        "--all-configs",
        action="store_true",
        help="Plot every F* folder that has results.json",
    )
    p.add_argument(
        "--stations",
        nargs="*",
        default=None,
        help="Subset of station folder names",
    )
    p.add_argument(
        "--results-root",
        default="results/informer2020",
    )
    p.add_argument(
        "--with-split",
        action="store_true",
        help="Also plot target series with train/val/test spans (needs data.csv)",
    )
    return p.parse_args()


def plot_one(out_dir: Path, *, with_split: bool) -> bool:
    results_path = out_dir / "results.json"
    if not results_path.exists():
        return False
    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    preds_path = out_dir / "test_preds_ma24.npy"
    trues_path = out_dir / "test_trues_ma24.npy"
    preds = np.load(preds_path) if preds_path.exists() else None
    trues = np.load(trues_path) if trues_path.exists() else None

    target_series = None
    borders = results.get("borders")
    if with_split and borders:
        station = results["station"]
        zone = int(results["zone"])
        cfg = results["feature_config"]
        # Prefer folder name key under informer2020/
        data_csv = informer2020_station_dir(zone, out_dir.parent.name, cfg) / "data.csv"
        if not data_csv.exists():
            # Fallback: station string with spaces
            key = station.replace(" ", "_").replace("(", "").replace(")", "")
            data_csv = informer2020_station_dir(zone, key, cfg) / "data.csv"
        if data_csv.exists():
            df = pd.read_csv(data_csv, usecols=[results.get("contaminant", "PM10")])
            target_series = df.iloc[:, 0].to_numpy()

    arts = save_run_plots(
        out_dir,
        history=results.get("history") or [],
        preds_ma24=preds,
        trues_ma24=trues,
        target_series=target_series,
        borders=borders,
        station=results.get("station", out_dir.parent.name),
        feature_config=results.get("feature_config", ""),
        contaminant=results.get("contaminant", "PM10"),
    )
    results.setdefault("artifacts", {}).update(arts)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"OK {out_dir} -> {', '.join(Path(p).name for p in arts.values())}")
    return True


def main() -> None:
    args = parse_args()
    root = Path(args.results_root) / f"zone_{args.zone}"
    if not root.exists():
        raise SystemExit(f"Missing {root}")

    configs = ABLATION_LEVELS if args.all_configs else [args.feature_config]
    n = 0
    for station_dir in sorted(root.iterdir()):
        if not station_dir.is_dir():
            continue
        if args.stations and station_dir.name not in args.stations:
            continue
        for cfg in configs:
            out_dir = station_dir / cfg
            if plot_one(out_dir, with_split=args.with_split):
                n += 1
    print(f"Done. plotted={n}")


if __name__ == "__main__":
    main()
