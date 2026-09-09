"""Error histograms + PM10 bin / episode counts (zone 2 baseline)."""

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

from src.experiments.compare_all_models import (  # noqa: E402
    MODEL_LABELS,
    PRIMARY,
    best_config_per_model,
    fair_global_table,
    load_all,
)
from src.experiments.lead_time_metrics import (  # noqa: E402
    LEARNED,
    load_hourly_station,
    station_list_for_model,
)
from src.utils.paths import definitive_csv


def count_episodes(mask: np.ndarray) -> tuple[int, np.ndarray]:
    if mask.size == 0:
        return 0, np.array([], dtype=int)
    d = np.diff(mask.astype(int), prepend=0, append=0)
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return len(starts), (ends - starts)


def main() -> None:
    out_dir = ROOT / "results/comparison/zone_2/all_models"
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    # --- Full series: point counts + contiguous episodes on zone-mean ---
    raw = pd.read_csv(
        definitive_csv(2, "PM10"),
        usecols=["time", "PM10", "station_name"],
        parse_dates=["time"],
    ).dropna(subset=["PM10"])
    pm = raw["PM10"].to_numpy(dtype=float)
    n = len(pm)
    print("=== Puntos horarios (todas estaciones, serie completa) ===")
    bins = [(0, 10, "0-10"), (10, 20, "10-20"), (20, 40, "20-40"), (40, 80, "40-80"), (80, None, "80+")]
    rows_full = []
    for lo, hi, lab in bins:
        m = (pm >= 80) if hi is None else ((pm >= lo) & (pm < hi))
        rows_full.append({"scope": "all_stations_points", "pm10_bin": lab, "n": int(m.sum()), "pct": 100 * float(m.mean())})
        print(f"  {lab}: {m.sum()} ({100 * m.mean():.2f}%)")
    print(f"  total: {n}")

    z = raw.groupby("time")["PM10"].mean().sort_index()
    zm = z.to_numpy(dtype=float)
    print("\n=== Media zonal horaria ===")
    for lo, hi, lab in [(40, 80, "40-80"), (80, None, "80+")]:
        m = (zm >= 80) if hi is None else ((zm >= lo) & (zm < hi))
        print(f"  horas media zona en {lab}: {m.sum()} / {len(zm)} ({100 * m.mean():.2f}%)")
        rows_full.append(
            {
                "scope": "zone_mean_hours",
                "pm10_bin": lab,
                "n": int(m.sum()),
                "pct": 100 * float(m.mean()),
            }
        )

    n80, L80 = count_episodes(zm >= 80)
    n40, L40 = count_episodes((zm >= 40) & (zm < 80))
    print(
        f"\n  episodios contig. media>=80: {n80}"
        + (f"  duracion_h min/med/max={L80.min()}/{np.median(L80):.0f}/{L80.max()}" if n80 else "")
    )
    print(
        f"  episodios contig. media 40-80: {n40}"
        + (f"  duracion_h min/med/max={L40.min()}/{np.median(L40):.0f}/{L40.max()}" if n40 else "")
    )
    pd.DataFrame(
        [
            {
                "kind": "zone_mean_episodes_ge80",
                "n_episodes": n80,
                "dur_min": int(L80.min()) if n80 else None,
                "dur_median": float(np.median(L80)) if n80 else None,
                "dur_max": int(L80.max()) if n80 else None,
            },
            {
                "kind": "zone_mean_episodes_40_80",
                "n_episodes": n40,
                "dur_min": int(L40.min()) if n40 else None,
                "dur_median": float(np.median(L40)) if n40 else None,
                "dur_max": int(L40.max()) if n40 else None,
            },
        ]
    ).to_csv(tab_dir / "pm10_episode_counts.csv", index=False)
    pd.DataFrame(rows_full).to_csv(tab_dir / "pm10_bin_counts_full_series.csv", index=False)

    # --- Test errors (best F*) ---
    cmp = load_all(2)
    best = best_config_per_model(fair_global_table(cmp), PRIMARY)
    parts = []
    for _, r in best.iterrows():
        model = str(r["model"])
        if model not in LEARNED:
            continue
        cfg = str(r["feature_config"])
        for st in station_list_for_model(cmp, model, cfg):
            loaded = load_hourly_station(model, 2, cfg, st)
            if loaded is None:
                continue
            pred, true = loaded
            parts.append(
                pd.DataFrame(
                    {
                        "model": model,
                        "err": (pred - true).ravel(),
                        "abs_err": np.abs(pred - true).ravel(),
                        "true": true.ravel(),
                    }
                )
            )
    err_df = pd.concat(parts, ignore_index=True)

    print("\n=== Test (AirFormer points, true PM10) ===")
    af = err_df[err_df["model"] == "airformer"]
    counts = []
    for model in err_df["model"].unique():
        sub = err_df[err_df["model"] == model]
        for lo, hi, lab in bins:
            m = (sub["true"] >= 80) if hi is None else ((sub["true"] >= lo) & (sub["true"] < hi))
            counts.append(
                {
                    "model": model,
                    "pm10_bin": lab,
                    "n": int(m.sum()),
                    "pct": 100 * float(m.mean()),
                    "mae": float(sub.loc[m, "abs_err"].mean()) if m.any() else np.nan,
                }
            )
            if model == "airformer":
                print(f"  {lab}: {m.sum()} ({100 * m.mean():.2f}%)")
    pd.DataFrame(counts).to_csv(tab_dir / "error_hist_bin_counts.csv", index=False)

    models = list(err_df["model"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, model in zip(axes.ravel(), models):
        sub = err_df[err_df["model"] == model]
        ax.hist(sub["err"], bins=80, range=(-50, 50), color="#4c78a8", alpha=0.85, density=True)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"{MODEL_LABELS.get(model, model)} (all)")
        ax.set_xlabel("pred − true (µg/m³)")
    fig.suptitle("Error histogram (test, best F*)")
    fig.tight_layout()
    p1 = fig_dir / "fig_error_hist_all.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, model in zip(axes.ravel(), models):
        sub = err_df[(err_df["model"] == model) & (err_df["true"] >= 40)]
        ax.hist(sub["err"], bins=60, range=(-100, 50), color="#e45756", alpha=0.85, density=True)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"{MODEL_LABELS.get(model, model)} (true ≥ 40)")
        ax.set_xlabel("pred − true (µg/m³)")
    fig.suptitle("Error histogram — high PM10 (true ≥ 40)")
    fig.tight_layout()
    p2 = fig_dir / "fig_error_hist_high_pm10.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)

    print(f"\nWrote {p1}")
    print(f"Wrote {p2}")
    print(f"Wrote {tab_dir / 'pm10_episode_counts.csv'}")
    print(f"Wrote {tab_dir / 'pm10_bin_counts_full_series.csv'}")
    print(f"Wrote {tab_dir / 'error_hist_bin_counts.csv'}")


if __name__ == "__main__":
    main()
