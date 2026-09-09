"""
Publication figure: lead-time RMSE + hourly exceedance hit rate (POD).
Headline models only: XGBoost F5 and AirFormer F5 (hourly arrays available).
"""
from __future__ import annotations

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

from src.experiments.compare_all_models import load_all  # noqa: E402
from src.experiments.lead_time_metrics import (  # noqa: E402
    _mae_rmse_by_lead,
    lead_metrics_for_model,
    load_hourly_station,
    station_list_for_model,
)

MODELS = (
    ("xgboost", "F5", "XGBoost F5", "#e9c46a"),
    ("airformer", "F5", "AirFormer F5", "#e63946"),
)
THR = 45.0
ZONE = 2


def pod_by_lead(model: str, cfg: str, df: pd.DataFrame, threshold: float) -> np.ndarray:
    """POD at each lead: fraction of obs>=thr correctly predicted >=thr."""
    pods = []
    for lead_idx in range(24):
        tp = fn = 0
        for st in station_list_for_model(df, model, cfg):
            loaded = load_hourly_station(model, ZONE, cfg, st)
            if loaded is None:
                continue
            pred, true = loaded
            obs = true[:, lead_idx] >= threshold
            prd = pred[:, lead_idx] >= threshold
            tp += int((obs & prd).sum())
            fn += int((obs & ~prd).sum())
        pods.append(100.0 * tp / (tp + fn) if (tp + fn) else np.nan)
    return np.array(pods)


def main() -> None:
    df = load_all(ZONE)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    # --- Panel A: RMSE by lead ---
    ax = axes[0]
    for model, cfg, label, color in MODELS:
        part = lead_metrics_for_model(df, model, ZONE, cfg)
        zone = (
            part.groupby("lead", as_index=False)
            .agg(rmse=("rmse", "mean"))
            .sort_values("lead")
        )
        ax.plot(
            zone["lead"],
            zone["rmse"],
            marker="o",
            markersize=3,
            linewidth=2,
            color=color,
            label=label,
        )
        # MA24 RMSE reference (from zone aggregate - approximate from results)
        ma24_ref = 5.45 if model == "xgboost" else 5.65
        ax.axhline(
            ma24_ref,
            color=color,
            linestyle="--",
            linewidth=1.1,
            alpha=0.7,
            label=f"{label} MA24 RMSE={ma24_ref:.2f}",
        )
    for L in (1, 6, 12, 24):
        ax.axvline(L, color="#cccccc", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Lead time (h ahead)")
    ax.set_ylabel("Hourly RMSE (µg m$^{-3}$)")
    ax.set_title("(a) Error grows with lead time")
    ax.set_xticks([1, 6, 12, 18, 24])
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=7, loc="upper left")

    # --- Panel B: POD hourly >= 45 by lead ---
    ax = axes[1]
    for model, cfg, label, color in MODELS:
        pod = pod_by_lead(model, cfg, df, THR)
        ax.plot(
            np.arange(1, 25),
            pod,
            marker="s",
            markersize=3,
            linewidth=2,
            color=color,
            label=label,
        )
    ax.axvspan(1, 6, color="#f0f0f0", alpha=0.5, zorder=0)
    ax.set_xlabel("Lead time (h ahead)")
    ax.set_ylabel(f"Hit rate / POD (%) — hourly $\\geq${THR:.0f}")
    ax.set_title(f"(b) Exceedance detection by lead")
    ax.set_xticks([1, 6, 12, 18, 24])
    ax.set_ylim(0, max(55, ax.get_ylim()[1]))
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=7, loc="upper right")

    fig.tight_layout()
    out_comp = (
        ROOT
        / "results/comparison/zone_2/all_models/figures/fig_lead_rmse_hit_headline.png"
    )
    out_print = ROOT / "docs/figures_print/fig_lead_rmse_hit_headline.png"
    out_comp.parent.mkdir(parents=True, exist_ok=True)
    out_print.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_comp, dpi=180)
    fig.savefig(out_print, dpi=180)
    plt.close(fig)
    print(f"Saved {out_comp}")
    print(f"Saved {out_print}")

    # Export snapshot table
    rows = []
    for model, cfg, label, _ in MODELS:
        part = lead_metrics_for_model(df, model, ZONE, cfg)
        zone = part.groupby("lead", as_index=False).agg(rmse=("rmse", "mean"))
        pod = pod_by_lead(model, cfg, df, THR)
        for lead in range(1, 25):
            rm = float(zone.loc[zone["lead"] == lead, "rmse"].iloc[0])
            rows.append(
                {
                    "model": label,
                    "lead": lead,
                    "rmse": rm,
                    "pod_pct": float(pod[lead - 1]),
                }
            )
    tab = pd.DataFrame(rows)
    tab_path = (
        ROOT
        / "results/comparison/zone_2/all_models/tables/lead_rmse_pod_thr45_headline.csv"
    )
    tab.to_csv(tab_path, index=False)
    print(f"Saved {tab_path}")
    snap = tab[tab["lead"].isin([1, 3, 6, 12, 24])]
    print(snap.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
