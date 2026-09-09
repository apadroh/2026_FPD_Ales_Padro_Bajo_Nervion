"""
Error distribution diagnostics (baseline seq_len=48).

1) By hour-of-day (0–23): absolute error pooled over stations / leads.
2) By PM10 level bins of the true value: MAE / RMSE / counts.

Uses best F* per model (MA24 RMSE) from compare_all_models, same hourly
arrays as lead_time_metrics.

CLI:
  python src/experiments/error_distribution_analysis.py --zone 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.compare_all_models import (  # noqa: E402
    MODEL_LABELS,
    PRIMARY,
    best_config_per_model,
    fair_global_table,
    load_all,
    station_key,
)
from src.experiments.lead_time_metrics import (  # noqa: E402
    LEARNED,
    load_hourly_station,
    station_list_for_model,
)
from src.utils.paths import definitive_csv, informer2020_station_dir

PM10_BINS = [0, 10, 20, 40, 80, np.inf]
PM10_LABELS = ["0-10", "10-20", "20-40", "40-80", "80+"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Error distribution by hour / PM10 level")
    p.add_argument("--zone", type=int, default=2)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: results/comparison/zone_Z/all_models",
    )
    return p.parse_args()


def _test_origin_hours_airformer(zone: int, seq_len: int = 48) -> np.ndarray | None:
    """Hour-of-day for each test window origin (start of prediction horizon)."""
    meta_path = (
        ROOT
        / f"data/training/zone_{zone}/airformer/ZONE2_PM10/F1/metadata.json"
    )
    csv_path = definitive_csv(zone, "PM10")
    if not meta_path.exists() or not csv_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    n_train = int(meta["n_train"])
    n_val = int(meta["n_val"])
    n_test = int(meta["n_test"])
    raw = pd.read_csv(csv_path, usecols=["time"], parse_dates=["time"])
    times = pd.to_datetime(raw["time"]).drop_duplicates().sort_values().reset_index(drop=True)
    # Window i → horizon starts at times[i + seq_len]
    start = n_train + n_val
    idxs = np.arange(start, start + n_test) + seq_len
    if idxs.max() >= len(times):
        return None
    return times.iloc[idxs].dt.hour.to_numpy()


def _test_origin_hours_informer(zone: int, station: str, config: str, seq_len: int = 48) -> np.ndarray | None:
    data_dir = informer2020_station_dir(zone, station, config)
    csv_path = data_dir / "data.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    date_col = "date" if "date" in df.columns else "time"
    dates = pd.to_datetime(df[date_col])
    n = len(dates)
    train_ratio, val_ratio = 0.7, 0.1
    num_train = int(n * train_ratio)
    num_test = int(n * (1 - train_ratio - val_ratio))
    # Test set starts at n - num_test - seq_len (same as AirQualityDataset)
    border1 = n - num_test - seq_len
    border2 = n
    # Sample i in test uses data[border1+i : border1+i+seq_len]; pred starts at border1+i+seq_len
    n_windows = border2 - border1 - seq_len - 24 + 1
    if n_windows <= 0:
        return None
    origins = border1 + np.arange(n_windows) + seq_len
    return dates.iloc[origins].dt.hour.to_numpy()


def expand_hours(origin_hours: np.ndarray, horizon: int = 24) -> np.ndarray:
    """(n,) origin hours → (n, H) clock hour for each lead."""
    leads = np.arange(horizon)
    return (origin_hours[:, None] + leads[None, :]) % 24


def collect_errors(
    df: pd.DataFrame,
    best: pd.DataFrame,
    zone: int,
) -> pd.DataFrame:
    rows = []
    af_hours = _test_origin_hours_airformer(zone, seq_len=48)

    for _, r in best.iterrows():
        model = str(r["model"])
        if model not in LEARNED:
            continue
        cfg = str(r["feature_config"])
        stations = station_list_for_model(df, model, cfg)
        print(f"Errors: {MODEL_LABELS.get(model, model)} {cfg} ({len(stations)} st)…")
        for st in stations:
            loaded = load_hourly_station(model, zone, cfg, st)
            if loaded is None:
                continue
            pred, true = loaded
            n, h = pred.shape
            abs_err = np.abs(pred - true)

            if model in {"airformer", "gat_informer"} and af_hours is not None and len(af_hours) == n:
                hours = expand_hours(af_hours, h)
            else:
                oh = _test_origin_hours_informer(zone, st, cfg, seq_len=48)
                if oh is not None and len(oh) == n:
                    hours = expand_hours(oh, h)
                else:
                    hours = np.full((n, h), -1, dtype=int)

            flat = abs_err.ravel()
            rows.append(
                pd.DataFrame(
                    {
                        "model": model,
                        "feature_config": cfg,
                        "station": st,
                        "station_key": station_key(st),
                        "abs_err": flat,
                        "true_pm10": true.ravel(),
                        "hour": hours.ravel(),
                        "lead": np.tile(np.arange(1, h + 1), n),
                    }
                )
            )
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def summarize_by_hour(err: pd.DataFrame) -> pd.DataFrame:
    sub = err[err["hour"] >= 0]
    if sub.empty:
        return pd.DataFrame()
    g = (
        sub.groupby(["model", "feature_config", "hour"], as_index=False)
        .agg(mae=("abs_err", "mean"), rmse=("abs_err", lambda x: float(np.sqrt(np.mean(np.square(x))))), n=("abs_err", "size"))
    )
    return g


def summarize_by_pm10(err: pd.DataFrame) -> pd.DataFrame:
    e = err.copy()
    e["pm10_bin"] = pd.cut(
        e["true_pm10"], bins=PM10_BINS, labels=PM10_LABELS, right=False, include_lowest=True
    )
    g = (
        e.groupby(["model", "feature_config", "pm10_bin"], as_index=False, observed=True)
        .agg(mae=("abs_err", "mean"), rmse=("abs_err", lambda x: float(np.sqrt(np.mean(np.square(x))))), n=("abs_err", "size"))
    )
    return g


def plot_hour(zone_hour: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for model, sub in zone_hour.groupby("model"):
        sub = sub.sort_values("hour")
        ax.plot(
            sub["hour"],
            sub["mae"],
            marker="o",
            markersize=3,
            label=MODEL_LABELS.get(model, model),
        )
    ax.set_xlabel("Hour of day (local)")
    ax.set_ylabel("MAE (µg/m³)")
    ax.set_title("Absolute error by hour of day (best F* per model)")
    ax.set_xticks(range(0, 24))
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_pm10(zone_pm: pd.DataFrame, out: Path) -> None:
    models = list(zone_pm["model"].unique())
    x = np.arange(len(PM10_LABELS))
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, model in enumerate(models):
        sub = zone_pm[zone_pm["model"] == model].set_index("pm10_bin").reindex(PM10_LABELS)
        ax.bar(
            x + i * width - 0.4 + width / 2,
            sub["mae"].to_numpy(dtype=float),
            width=width,
            label=MODEL_LABELS.get(model, model),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(PM10_LABELS)
    ax.set_xlabel("True PM10 bin (µg/m³)")
    ax.set_ylabel("MAE (µg/m³)")
    ax.set_title("Absolute error by PM10 level (best F* per model)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or ROOT / f"results/comparison/zone_{args.zone}/all_models"
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    df = load_all(args.zone)
    best = best_config_per_model(fair_global_table(df), PRIMARY)
    err = collect_errors(df, best, args.zone)
    if err.empty:
        raise SystemExit("No hourly prediction arrays found.")

    by_hour = summarize_by_hour(err)
    by_pm = summarize_by_pm10(err)
    by_hour.to_csv(tab_dir / "error_by_hour_of_day.csv", index=False)
    by_pm.to_csv(tab_dir / "error_by_pm10_level.csv", index=False)

    if not by_hour.empty:
        plot_hour(by_hour, fig_dir / "fig_error_by_hour_of_day.png")
        print("Wrote", fig_dir / "fig_error_by_hour_of_day.png")
    plot_pm10(by_pm, fig_dir / "fig_error_by_pm10_level.png")
    print("Wrote", fig_dir / "fig_error_by_pm10_level.png")
    print("Tables:", tab_dir / "error_by_hour_of_day.csv", tab_dir / "error_by_pm10_level.csv")


if __name__ == "__main__":
    main()
