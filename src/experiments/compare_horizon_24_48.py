"""
Compare H=24 vs H=48 campaign results (zone 2, L=48).

H=24: MA24 RMSE (equal station mean).
H=48: MA24_D1 (h+1..24) and MA24_D2 (h+25..48), same ranking.

Usage:
  python src/experiments/compare_horizon_24_48.py --zone 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.feature_configs import ALL_FEATURE_LEVELS

MODELS = (
    ("informer", "Informer2020"),
    ("airformer", "AirFormer"),
    ("gat", "GAT-Informer"),
    ("xgboost", "XGBoost"),
)


def _mean_rmse(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _zone_json(zone: int, model: str, cfg: str, horizon: int) -> Path | None:
    if model == "airformer":
        base = ROOT / f"results/airformer/zone_{zone}/ZONE2_PM10"
        return (base / "horizon_48" / cfg if horizon == 48 else base / cfg) / "results.json"
    if model == "gat":
        base = ROOT / f"results/gat_informer_mf_pm10graph/zone_{zone}"
        return (base / "horizon_48" / cfg if horizon == 48 else base / cfg) / "results.json"
    return None


def _per_station_jsons(zone: int, model: str, cfg: str, horizon: int) -> list[Path]:
    if model == "informer":
        base = ROOT / f"results/informer2020/zone_{zone}"
    elif model == "xgboost":
        base = ROOT / f"results/xgboost/zone_{zone}"
    else:
        return []
    if horizon == 48:
        base = base / "horizon_48"
    paths = []
    for st_dir in sorted(base.iterdir()):
        if not st_dir.is_dir() or st_dir.name.startswith("batch"):
            continue
        p = st_dir / cfg / "results.json"
        if p.exists():
            paths.append(p)
    return paths


def _metrics_from_json(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, float] = {}
    raw = data.get("test_metrics_raw") or {}
    if "ma24_d1" in raw:
        out["ma24_d1"] = float(raw["ma24_d1"]["rmse"])
    if "ma24_d2" in raw:
        out["ma24_d2"] = float(raw["ma24_d2"]["rmse"])
    if "ma24" in raw:
        out["ma24"] = float(raw["ma24"]["rmse"])
    if "hourly" in raw:
        out["hourly"] = float(raw["hourly"]["rmse"])

    per = data.get("per_station")
    if per:
        for key, short in (
            ("ma24_rmse", "ma24"),
            ("ma24_d1_rmse", "ma24_d1"),
            ("ma24_d2_rmse", "ma24_d2"),
            ("hourly_rmse", "hourly"),
        ):
            vals = [float(r[key]) for r in per if key in r and r.get(key) is not None]
            if vals:
                out[short] = _mean_rmse(vals)

    # H=48 zone JSON may alias ma24→d1 in raw but lack d2; per-station mean is safer for GAT.
    if "ma24_d1" not in out and "ma24" in out and not per:
        out["ma24_d1"] = out["ma24"]
    elif "ma24_d1" not in out and "ma24" in out and per:
        # Prefer per-station aggregate over identical broken global ma24 (seen on GAT H=48).
        pass

    return out


def _batch_summary(zone: int, model: str, cfg: str, horizon: int) -> dict[str, float]:
    if model == "informer":
        base = ROOT / f"results/informer2020/zone_{zone}"
    elif model == "xgboost":
        base = ROOT / f"results/xgboost/zone_{zone}"
    else:
        return {}
    if horizon == 48:
        base = base / "horizon_48"
    path = base / f"batch_summary_{cfg}.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    df = df[df["status"] == "ok"]
    out: dict[str, float] = {}
    if "ma24_rmse" in df.columns:
        out["ma24"] = float(df["ma24_rmse"].mean())
    if "ma24_d1_rmse" in df.columns:
        out["ma24_d1"] = float(df["ma24_d1_rmse"].mean())
    if "ma24_d2_rmse" in df.columns:
        out["ma24_d2"] = float(df["ma24_d2_rmse"].mean())
    if "hourly_rmse" in df.columns:
        out["hourly"] = float(df["hourly_rmse"].mean())
    return out


def load_horizon_metrics(zone: int, model: str, cfg: str, horizon: int) -> dict[str, float]:
    if model in ("airformer", "gat"):
        p = _zone_json(zone, model, cfg, horizon)
        if p and p.exists():
            return _metrics_from_json(p)
    # per-station: prefer aggregating JSON (has D1/D2); fallback batch summary
    paths = _per_station_jsons(zone, model, cfg, horizon)
    if paths:
        keys = ("ma24", "ma24_d1", "ma24_d2", "hourly")
        acc: dict[str, list[float]] = {k: [] for k in keys}
        for p in paths:
            m = _metrics_from_json(p)
            for k in keys:
                if k in m:
                    acc[k].append(m[k])
        out = {k: _mean_rmse(v) for k, v in acc.items() if v}
        if out:
            return out
    return _batch_summary(zone, model, cfg, horizon)


def build_table(zone: int) -> pd.DataFrame:
    rows = []
    for model_key, model_label in MODELS:
        for cfg in ALL_FEATURE_LEVELS:
            h24 = load_horizon_metrics(zone, model_key, cfg, 24)
            h48 = load_horizon_metrics(zone, model_key, cfg, 48)
            if not h24 and not h48:
                continue
            ma24_24 = h24.get("ma24")
            d1_48 = h48.get("ma24_d1", h48.get("ma24"))
            d2_48 = h48.get("ma24_d2")
            rows.append(
                {
                    "model": model_label,
                    "config": cfg,
                    "h24_ma24_rmse": ma24_24,
                    "h48_ma24_d1_rmse": d1_48,
                    "h48_ma24_d2_rmse": d2_48,
                    "h24_hourly_rmse": h24.get("hourly"),
                    "h48_hourly_rmse": h48.get("hourly"),
                    "d1_vs_h24_delta": (d1_48 - ma24_24) if d1_48 and ma24_24 else None,
                    "d2_vs_d1_delta": (d2_48 - d1_48) if d2_48 and d1_48 else None,
                }
            )
    return pd.DataFrame(rows)


def best_per_model(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        if sub["h24_ma24_rmse"].notna().any():
            i24 = sub["h24_ma24_rmse"].idxmin()
            rows.append({"model": model, "horizon": "H=24 MA24", "config": sub.loc[i24, "config"], "rmse": sub.loc[i24, "h24_ma24_rmse"]})
        if sub["h48_ma24_d1_rmse"].notna().any():
            i1 = sub["h48_ma24_d1_rmse"].idxmin()
            rows.append({"model": model, "horizon": "H=48 MA24_D1", "config": sub.loc[i1, "config"], "rmse": sub.loc[i1, "h48_ma24_d1_rmse"]})
        if sub["h48_ma24_d2_rmse"].notna().any():
            i2 = sub["h48_ma24_d2_rmse"].idxmin()
            rows.append({"model": model, "horizon": "H=48 MA24_D2", "config": sub.loc[i2, "config"], "rmse": sub.loc[i2, "h48_ma24_d2_rmse"]})
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zone", type=int, default=2)
    args = p.parse_args()

    df = build_table(args.zone)
    if df.empty:
        print("No metrics found.")
        return 1

    out_dir = ROOT / f"results/comparison/zone_{args.zone}/horizon_48"
    out_dir.mkdir(parents=True, exist_ok=True)
    full_path = out_dir / "compare_h24_vs_h48_full.csv"
    best_path = out_dir / "compare_h24_vs_h48_best.csv"
    df.round(4).to_csv(full_path, index=False)

    best = best_per_model(df)
    best.round(4).to_csv(best_path, index=False)

    print(f"Wrote {full_path}")
    print(f"Wrote {best_path}\n")
    print("=== Best per model ===")
    print(best.to_string(index=False))
    print("\n=== Headline (official F1-F5 ladder, best config each) ===")
    headline = best[best["horizon"].isin(["H=24 MA24", "H=48 MA24_D1", "H=48 MA24_D2"])]
    print(headline.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
