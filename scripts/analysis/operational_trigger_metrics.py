"""Operational trigger metrics: >=N stations exceed threshold for 2 consecutive leads."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.compare_all_models import (  # noqa: E402
    best_config_per_model,
    fair_global_table,
    load_all,
)
from src.experiments.lead_time_metrics import (  # noqa: E402
    LEARNED,
    load_hourly_station,
    station_list_for_model,
)

MODEL_LABELS = {
    "xgboost": "XGBoost",
    "airformer": "AirFormer",
    "informer": "Informer",
    "gat_informer": "GAT-Informer",
}


def contingency(pred_event: np.ndarray, obs_event: np.ndarray) -> dict:
    tp = int(np.sum(pred_event & obs_event))
    fp = int(np.sum(pred_event & ~obs_event))
    fn = int(np.sum(~pred_event & obs_event))
    tn = int(np.sum(~pred_event & ~obs_event))
    pod = tp / (tp + fn) if (tp + fn) else float("nan")
    far = fp / (tp + fp) if (tp + fp) else float("nan")
    csi = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "pod": pod,
        "far": far,
        "csi": csi,
        "n_windows": len(obs_event),
        "n_obs_events": int(obs_event.sum()),
        "n_pred_events": int(pred_event.sum()),
    }


def load_zone_hourly(
    model: str, zone: int, config: str, df: pd.DataFrame, n_leads: int = 2
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return pred/true arrays (n_windows, n_stations, n_leads)."""
    stations = station_list_for_model(df, model, config)
    preds, trues = [], []
    for st in stations:
        loaded = load_hourly_station(model, zone, config, st)
        if loaded is None:
            continue
        p, t = loaded
        preds.append(p[:, :n_leads])
        trues.append(t[:, :n_leads])
    if not preds:
        return None
    n = min(p.shape[0] for p in preds)
    preds = [p[:n] for p in preds]
    trues = [t[:n] for t in trues]
    return np.stack(preds, axis=1), np.stack(trues, axis=1)


def trigger_events(
    arr: np.ndarray, threshold: float, min_stations: int, consecutive_leads: int
) -> np.ndarray:
    """arr: (n, S, L). True if >=min_stations exceed threshold on each of first L leads."""
    counts = (arr >= threshold).sum(axis=1)  # (n, L)
    ok = counts >= min_stations
    if consecutive_leads == 1:
        return ok[:, 0]
    return np.all(ok[:, :consecutive_leads], axis=1)


def run_per_station_report(threshold: float = 40.0, zone: int = 2) -> None:
    """Print observed vs predicted exceedance counts per station."""
    df = load_all(zone)
    best = best_config_per_model(fair_global_table(df))

    def counts(model: str, cfg: str, n_leads: int | None) -> pd.DataFrame:
        rows = []
        for st in station_list_for_model(df, model, cfg):
            loaded = load_hourly_station(model, zone, cfg, st)
            if loaded is None:
                continue
            pred, true = loaded
            if n_leads is not None:
                pred, true = pred[:, :n_leads], true[:, :n_leads]
            n = true.size
            rows.append(
                {
                    "station": st,
                    "n_points": n,
                    "obs_exceed": int((true >= threshold).sum()),
                    "pred_exceed": int((pred >= threshold).sum()),
                    "obs_pct": 100.0 * (true >= threshold).mean(),
                    "pred_pct": 100.0 * (pred >= threshold).mean(),
                }
            )
        return pd.DataFrame(rows)

    out_dir = ROOT / f"results/comparison/zone_{zone}/all_models/tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    for model in ("xgboost", "airformer"):
        cfg = str(best[best["model"] == model].iloc[0]["feature_config"])
        label = MODEL_LABELS.get(model, model)
        for leads, tag in ((None, "all24"), (2, "h1h2")):
            t = counts(model, cfg, leads)
            t = t.sort_values("obs_exceed", ascending=False)
            path = out_dir / f"exceedance_thr{int(threshold)}_per_station_{model}_{tag}.csv"
            t.to_csv(path, index=False)
            print(f"\n{'='*72}")
            print(f"{label} {cfg} | thr={threshold} | {tag}")
            print(t.to_string(index=False))
            print(
                f"TOTAL: obs={t['obs_exceed'].sum()} pred={t['pred_exceed'].sum()} "
                f"points={t['n_points'].sum()}"
            )
            print(f"Saved -> {path}")


def run_ma24_exceedance_report(threshold: float = 40.0, zone: int = 2) -> None:
    """Observed vs predicted MA24 exceedances (mean of h+1..h+24 per window)."""
    df = load_all(zone)
    best = best_config_per_model(fair_global_table(df))
    out_dir = ROOT / f"results/comparison/zone_{zone}/all_models/tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    zone_rows = []

    for model in ("xgboost", "airformer", "informer", "gat_informer"):
        sub = best[best["model"] == model]
        if sub.empty:
            continue
        cfg = str(sub.iloc[0]["feature_config"])
        label = MODEL_LABELS.get(model, model)
        st_rows = []
        obs_ma_all, pred_ma_all = [], []

        for st in station_list_for_model(df, model, cfg):
            loaded = load_hourly_station(model, zone, cfg, st)
            if loaded is None:
                continue
            pred, true = loaded
            obs_ma = true.mean(axis=1)
            pred_ma = pred.mean(axis=1)
            obs_ma_all.append(obs_ma)
            pred_ma_all.append(pred_ma)
            st_rows.append(
                {
                    "model": label,
                    "config": cfg,
                    "station": st,
                    "n_windows": len(obs_ma),
                    "obs_exceed": int((obs_ma >= threshold).sum()),
                    "pred_exceed": int((pred_ma >= threshold).sum()),
                    "obs_pct": 100.0 * (obs_ma >= threshold).mean(),
                    "pred_pct": 100.0 * (pred_ma >= threshold).mean(),
                    "obs_ma_mean": float(obs_ma.mean()),
                    "pred_ma_mean": float(pred_ma.mean()),
                }
            )

        if not st_rows:
            continue

        t = pd.DataFrame(st_rows).sort_values("obs_exceed", ascending=False)
        path = out_dir / f"exceedance_ma24_thr{int(threshold)}_per_station_{model}.csv"
        t.to_csv(path, index=False)

        n = min(len(x) for x in obs_ma_all)
        obs_stack = np.stack([x[:n] for x in obs_ma_all], axis=1)
        pred_stack = np.stack([x[:n] for x in pred_ma_all], axis=1)
        obs_zone = (obs_stack >= threshold).sum(axis=1) >= 2
        pred_zone = (pred_stack >= threshold).sum(axis=1) >= 2
        c = contingency(pred_zone, obs_zone)
        zone_rows.append({"model": label, "config": cfg, **c})

        print(f"\n{'='*72}")
        print(f"{label} {cfg} | MA24 >= {threshold} ug/m3 (media h+1..h+24 por ventana)")
        print(t.to_string(index=False))
        print(
            f"TOTAL estaciones: obs={t['obs_exceed'].sum()} pred={t['pred_exceed'].sum()} "
            f"ventanas={t['n_windows'].sum()}"
        )
        print(
            f"Evento zona (>=2 estaciones MA24>={threshold} en misma ventana): "
            f"obs={c['n_obs_events']} pred={c['n_pred_events']} | "
            f"POD={c['pod']:.3f} FAR={c['far']:.3f} CSI={c['csi']:.3f}"
        )
        print(f"Saved -> {path}")

    if zone_rows:
        zdf = pd.DataFrame(zone_rows)
        zpath = out_dir / f"exceedance_ma24_thr{int(threshold)}_zone_trigger.csv"
        zdf.to_csv(zpath, index=False)
        print(f"\nZone MA24 trigger summary -> {zpath}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zone", type=int, default=2)
    p.add_argument("--threshold", type=float, default=40.0)
    p.add_argument("--min-stations", type=int, default=2)
    p.add_argument("--consecutive-hours", type=int, default=2)
    p.add_argument(
        "--ma24",
        action="store_true",
        help="Per-station MA24 exceedance counts (obs vs pred)",
    )
    args = p.parse_args()

    if args.ma24:
        run_ma24_exceedance_report(args.threshold, zone=args.zone)
        return

    if args.per_station:
        run_per_station_report(args.threshold, zone=args.zone)
        return

    df = load_all(args.zone)
    fair = fair_global_table(df)
    best = best_config_per_model(fair)

    rows = []
    print(
        f"Rule: >={args.min_stations} stations with PM10 >= {args.threshold} "
        f"for {args.consecutive_hours} consecutive forecast lead(s)\n"
    )
    for model in LEARNED:
        sub = best[best["model"] == model]
        if sub.empty:
            continue
        cfg = str(sub.iloc[0]["feature_config"])
        loaded = load_zone_hourly(model, args.zone, cfg, df, n_leads=args.consecutive_hours)
        if loaded is None:
            continue
        pred, true = loaded
        pred_ev = trigger_events(
            pred, args.threshold, args.min_stations, args.consecutive_hours
        )
        obs_ev = trigger_events(
            true, args.threshold, args.min_stations, args.consecutive_hours
        )
        c = contingency(pred_ev, obs_ev)
        rows.append(
            {
                "model": MODEL_LABELS.get(model, model),
                "config": cfg,
                **c,
            }
        )

    out = pd.DataFrame(rows)
    print(
        out[
            [
                "model",
                "config",
                "n_windows",
                "n_obs_events",
                "n_pred_events",
                "tp",
                "fp",
                "fn",
                "pod",
                "far",
                "csi",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )

    tab_dir = ROOT / f"results/comparison/zone_{args.zone}/all_models/tables"
    tab_dir.mkdir(parents=True, exist_ok=True)
    path = tab_dir / f"operational_trigger_thr{int(args.threshold)}.csv"
    out.to_csv(path, index=False)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
