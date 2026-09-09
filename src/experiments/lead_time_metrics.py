"""
Lead-time (horizon) metrics: MAE / RMSE at H+1 … H+24.

Uses the same hourly prediction arrays as training evaluation
(`test_preds_hourly.npy` / `test_trues_hourly.npy`).

Protocol (fair with zone ranking):
  - For each station: MAE/RMSE over test windows at each lead k.
  - Zone score = equal mean over stations (1/N).
  - Default: best F* per model by MA24 RMSE (from fair global table).

CLI:
  python src/experiments/lead_time_metrics.py --zone 2
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
    GAT_RESULTS_DIR,
    MODEL_LABELS,
    PRIMARY,
    ROOT as CMP_ROOT,
    _find_informer_station_dir,
    _stations_from_multinode,
    best_config_per_model,
    fair_global_table,
    load_all,
    station_key,
)

assert CMP_ROOT == ROOT

SNAPSHOT_LEADS = (1, 6, 12, 24)
LEARNED = ("informer", "airformer", "gat_informer", "xgboost")


def _squeeze_hourly(arr: np.ndarray) -> np.ndarray:
    """Return (n, pred_len) float array."""
    a = np.asarray(arr, dtype=np.float64)
    while a.ndim > 2 and a.shape[-1] == 1:
        a = a[..., 0]
    if a.ndim == 1:
        raise ValueError(f"Expected (n, H) or (n, H, S…), got {arr.shape}")
    return a


def _mae_rmse_by_lead(pred: np.ndarray, true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """pred/true: (n, H) → mae[H], rmse[H]."""
    err = pred - true
    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(err**2, axis=0))
    return mae, rmse


def _xgboost_station_dir(zone: int, station: str, config: str) -> Path | None:
    base = ROOT / f"results/xgboost/zone_{zone}"
    if not base.exists():
        return None
    sk = station_key(station)
    for st_dir in base.iterdir():
        if st_dir.is_dir() and station_key(st_dir.name) == sk:
            hit = st_dir / config
            if (hit / "test_preds_hourly.npy").exists():
                return hit
    return None


def load_hourly_station(
    model: str, zone: int, config: str, station: str
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (n, 24) pred/true for one station."""
    if model == "informer":
        d = _find_informer_station_dir(zone, station, config)
        if d is None or not (d / "test_preds_hourly.npy").exists():
            return None
        pred = _squeeze_hourly(np.load(d / "test_preds_hourly.npy"))
        true = _squeeze_hourly(np.load(d / "test_trues_hourly.npy"))
        return pred, true

    if model == "xgboost":
        d = _xgboost_station_dir(zone, station, config)
        if d is None:
            return None
        pred = _squeeze_hourly(np.load(d / "test_preds_hourly.npy"))
        true = _squeeze_hourly(np.load(d / "test_trues_hourly.npy"))
        return pred, true

    if model == "airformer":
        d = ROOT / f"results/airformer/zone_{zone}/ZONE2_PM10" / config
    elif model == "gat_informer":
        d = ROOT / f"results/{GAT_RESULTS_DIR}/zone_{zone}" / config
    else:
        return None

    pred_p, true_p, meta = (
        d / "test_preds_hourly.npy",
        d / "test_trues_hourly.npy",
        d / "results.json",
    )
    if not pred_p.exists() or not true_p.exists():
        return None
    stations = _stations_from_multinode(meta)
    if not stations and meta.exists():
        data = json.loads(meta.read_text(encoding="utf-8"))
        stations = [ps["station"] for ps in data.get("per_station") or []]
    sk = station_key(station)
    idx = next((i for i, s in enumerate(stations) if station_key(s) == sk), None)
    if idx is None:
        return None
    pred = np.asarray(np.load(pred_p), dtype=np.float64)
    true = np.asarray(np.load(true_p), dtype=np.float64)
    # (n, 24, S, 1) or (n, 24, S)
    while pred.ndim > 3 and pred.shape[-1] == 1:
        pred = pred[..., 0]
        true = true[..., 0]
    if pred.ndim != 3:
        raise ValueError(f"{model} hourly shape unexpected: {pred.shape}")
    return pred[:, :, idx], true[:, :, idx]


def station_list_for_model(df: pd.DataFrame, model: str, config: str) -> list[str]:
    sub = df[(df["model"] == model) & (df["feature_config"] == config)]
    # Prefer display names from metrics table
    rows = sub.drop_duplicates("station_key")
    return list(rows["station"].astype(str))


def lead_metrics_for_model(
    df: pd.DataFrame, model: str, zone: int, config: str
) -> pd.DataFrame:
    stations = station_list_for_model(df, model, config)
    per_st = []
    for st in stations:
        loaded = load_hourly_station(model, zone, config, st)
        if loaded is None:
            continue
        pred, true = loaded
        if pred.shape != true.shape or pred.ndim != 2:
            continue
        mae, rmse = _mae_rmse_by_lead(pred, true)
        for k in range(pred.shape[1]):
            per_st.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS.get(model, model),
                    "feature_config": config,
                    "station": st,
                    "station_key": station_key(st),
                    "lead": k + 1,
                    "mae": float(mae[k]),
                    "rmse": float(rmse[k]),
                    "n_windows": int(pred.shape[0]),
                }
            )
    return pd.DataFrame(per_st)


def aggregate_zone(per_station: pd.DataFrame) -> pd.DataFrame:
    """Equal mean over stations at each lead."""
    if per_station.empty:
        return per_station
    g = (
        per_station.groupby(
            ["model", "model_label", "feature_config", "lead"], as_index=False
        )
        .agg(
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            n_stations=("station_key", "nunique"),
        )
        .sort_values(["model", "lead"])
        .reset_index(drop=True)
    )
    return g


def snapshot_table(zone_df: pd.DataFrame, leads: tuple[int, ...] = SNAPSHOT_LEADS) -> pd.DataFrame:
    rows = []
    for (model, cfg), sub in zone_df.groupby(["model", "feature_config"]):
        row = {
            "model": model,
            "model_label": MODEL_LABELS.get(model, model),
            "feature_config": cfg,
            "n_stations": int(sub["n_stations"].iloc[0]) if len(sub) else 0,
        }
        for L in leads:
            hit = sub[sub["lead"] == L]
            if hit.empty:
                row[f"mae_h{L}"] = np.nan
                row[f"rmse_h{L}"] = np.nan
            else:
                row[f"mae_h{L}"] = float(hit["mae"].iloc[0])
                row[f"rmse_h{L}"] = float(hit["rmse"].iloc[0])
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty and "mae_h24" in out.columns:
        out = out.sort_values("mae_h24").reset_index(drop=True)
    return out


def plot_lead_curves(
    zone_df: pd.DataFrame,
    metric: str,
    out_path: Path,
    title: str,
    *,
    overall: dict[str, float] | None = None,
    overall_label: str = "hourly mean",
) -> Path:
    """
    Plot lead curves. If ``overall`` is given (model → scalar), draw a dashed
    horizontal line per model (e.g. reported Hourly MAE / mean of the curve).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    colors = {
        "informer": "#457b9d",
        "airformer": "#e63946",
        "gat_informer": "#2a9d8f",
        "xgboost": "#e9c46a",
    }
    for model in LEARNED:
        sub = zone_df[zone_df["model"] == model].sort_values("lead")
        if sub.empty:
            continue
        color = colors.get(model, None)
        label = MODEL_LABELS.get(model, model)
        ax.plot(
            sub["lead"],
            sub[metric],
            marker="o",
            markersize=3,
            linewidth=2,
            label=label,
            color=color,
        )
        # Overall mean (reported hourly metric, or mean of the 24 leads)
        if overall is not None and model in overall and np.isfinite(overall[model]):
            y = float(overall[model])
        else:
            y = float(sub[metric].mean())
        ax.axhline(
            y,
            color=color,
            linestyle="--",
            linewidth=1.3,
            alpha=0.85,
            zorder=1,
            label=f"{label} {overall_label}={y:.2f}",
        )
    for L in SNAPSHOT_LEADS:
        ax.axvline(L, color="#bbbbbb", linestyle=":", linewidth=0.9, zorder=0)
    ax.set_xlabel("Lead time (hour ahead)")
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.set_xticks(list(range(1, 25)))
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def run_lead_time_analysis(
    zone: int = 2,
    out_dir: Path | None = None,
    *,
    df: pd.DataFrame | None = None,
    best: pd.DataFrame | None = None,
) -> dict:
    out_dir = out_dir or ROOT / f"results/comparison/zone_{zone}/all_models"
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    if df is None:
        df = load_all(zone)
    if best is None:
        global_df = fair_global_table(df)
        best = best_config_per_model(global_df, PRIMARY)

    per_parts = []
    for _, r in best.iterrows():
        model = r["model"]
        if model not in LEARNED:
            continue
        cfg = r["feature_config"]
        print(f"Lead-time: {MODEL_LABELS.get(model, model)} {cfg} …")
        part = lead_metrics_for_model(df, model, zone, cfg)
        if part.empty:
            print(f"  WARNING: no hourly arrays for {model}/{cfg}")
            continue
        print(f"  stations={part['station_key'].nunique()} leads=24")
        per_parts.append(part)

    if not per_parts:
        return {"per_station": pd.DataFrame(), "zone": pd.DataFrame(), "snapshot": pd.DataFrame()}

    per_station = pd.concat(per_parts, ignore_index=True)
    zone_df = aggregate_zone(per_station)
    snap = snapshot_table(zone_df)

    per_station.to_csv(tab_dir / "lead_time_mae_rmse_by_station.csv", index=False)
    zone_df.to_csv(tab_dir / "lead_time_mae_rmse_zone.csv", index=False)
    snap.to_csv(tab_dir / "lead_time_snapshot_h1_h6_h12_h24.csv", index=False)

    # Reported zone-level hourly metrics (same as table1 / best F*)
    overall_mae = {
        str(r["model"]): float(r["hourly_mae"])
        for _, r in best.iterrows()
        if r["model"] in LEARNED and "hourly_mae" in r and np.isfinite(r["hourly_mae"])
    }
    overall_rmse = {
        str(r["model"]): float(r["hourly_rmse"])
        for _, r in best.iterrows()
        if r["model"] in LEARNED and "hourly_rmse" in r and np.isfinite(r["hourly_rmse"])
    }

    plot_lead_curves(
        zone_df,
        "mae",
        fig_dir / "fig_lead_time_mae.png",
        "MAE by lead time + Hourly MAE (dashed; best F* per model)",
        overall=overall_mae,
        overall_label="Hourly MAE",
    )
    plot_lead_curves(
        zone_df,
        "rmse",
        fig_dir / "fig_lead_time_rmse.png",
        "RMSE by lead time + Hourly RMSE (dashed; best F* per model)",
        overall=overall_rmse,
        overall_label="Hourly RMSE",
    )

    print("Snapshot (MAE @ H+1 / H+6 / H+12 / H+24):")
    cols = [
        "model_label",
        "feature_config",
        "mae_h1",
        "mae_h6",
        "mae_h12",
        "mae_h24",
    ]
    print(snap[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    return {
        "per_station": per_station,
        "zone": zone_df,
        "snapshot": snap,
        "fig_mae": fig_dir / "fig_lead_time_mae.png",
        "fig_rmse": fig_dir / "fig_lead_time_rmse.png",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zone", type=int, default=2)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: results/comparison/zone_Z/all_models",
    )
    args = p.parse_args()
    run_lead_time_analysis(zone=args.zone, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
