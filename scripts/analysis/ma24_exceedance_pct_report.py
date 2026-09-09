"""Print MA24 exceedance rates as percentages."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=40.0)
    p.add_argument("--zone", type=int, default=2)
    args = p.parse_args()

    base = ROOT / f"results/comparison/zone_{args.zone}/all_models/tables"
    thr = int(args.threshold)

    rows = []
    for model in ("xgboost", "airformer"):
        path = base / f"exceedance_ma24_thr{thr}_per_station_{model}.csv"
        if not path.exists():
            continue
        t = pd.read_csv(path)
        t["obs_pct"] = 100.0 * t["obs_exceed"] / t["n_windows"]
        t["pred_pct"] = 100.0 * t["pred_exceed"] / t["n_windows"]
        label = t["model"].iloc[0]
        rows.append(
            {
                "model": label,
                "obs_pct_pooled": 100.0 * t["obs_exceed"].sum() / t["n_windows"].sum(),
                "pred_pct_pooled": 100.0 * t["pred_exceed"].sum() / t["n_windows"].sum(),
                "obs_pct_mean_station": t["obs_pct"].mean(),
                "pred_pct_mean_station": t["pred_pct"].mean(),
            }
        )
        out = t[
            [
                "station",
                "n_windows",
                "obs_exceed",
                "pred_exceed",
                "obs_pct",
                "pred_pct",
            ]
        ].sort_values("obs_pct", ascending=False)
        out_path = base / f"exceedance_ma24_thr{thr}_pct_per_station_{model}.csv"
        out.to_csv(out_path, index=False, float_format="%.3f")

    zpath = base / f"exceedance_ma24_thr{thr}_zone_trigger.csv"
    if zpath.exists():
        z = pd.read_csv(zpath)
        z["obs_pct_windows"] = 100.0 * z["n_obs_events"] / z["n_windows"]
        z["pred_pct_windows"] = 100.0 * z["n_pred_events"] / z["n_windows"]
        z["pod_pct"] = 100.0 * z["pod"]
        z["far_pct"] = 100.0 * z["far"]
        z["csi_pct"] = 100.0 * z["csi"]
        z.to_csv(base / f"exceedance_ma24_thr{thr}_zone_trigger_pct.csv", index=False)

    print(f"MA24 >= {args.threshold} µg/m³ — rates as % of forecast windows\n")
    print("Per-station (pooled over all station×window pairs):")
    for r in rows:
        print(
            f"  {r['model']}: observed {r['obs_pct_pooled']:.2f}% | "
            f"predicted {r['pred_pct_pooled']:.2f}%"
        )
    print("\nPer-station (equal mean over 18 stations):")
    for r in rows:
        print(
            f"  {r['model']}: observed {r['obs_pct_mean_station']:.2f}% | "
            f"predicted {r['pred_pct_mean_station']:.2f}%"
        )

    if zpath.exists():
        print("\nZone trigger (>=2 stations, same window):")
        for _, r in z.iterrows():
            print(
                f"  {r['model']}: obs {r['obs_pct_windows']:.2f}% of windows | "
                f"pred {r['pred_pct_windows']:.2f}% | "
                f"POD {r['pod_pct']:.1f}% | FAR {r['far_pct']:.1f}% | CSI {r['csi_pct']:.1f}%"
            )


if __name__ == "__main__":
    main()
