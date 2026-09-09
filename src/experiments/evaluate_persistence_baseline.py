"""
Evaluate persistence baselines on the same test windows as Informer2020.

Baselines (raw units):
  - persistence_last:  yhat_{h+k} = y_h  for k=1..24
                       → MA24_pred = y_h
  - persistence_ma24:  yhat_{h+k} = mean(y_{h-23}..y_h)  for k=1..24
                       → MA24_pred = mean of last 24 observed hours

Then compares against Informer batch_summary_F*.csv (same stations).

Usage:
  python src/experiments/evaluate_persistence_baseline.py --zone 2
  python src/experiments/evaluate_persistence_baseline.py --zone 2 --feature-config F1
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

import re
import unicodedata

from src.data.air_quality_dataset import AirQualityDataset
from src.data.feature_configs import ABLATION_LEVELS
from src.models.informer2020.metrics import metric_hourly_and_ma24
from src.utils.paths import informer2020_station_dir


def station_key(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.upper()
    s = re.sub(r"[^A-Z0-9]+", "_", s).strip("_")
    if "DIAZ" in s and "HARO" in s:
        return "M_DIAZ_HARO"
    return s


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Persistence baseline vs Informer")
    p.add_argument("--zone", type=int, default=2)
    p.add_argument("--contaminant", default="PM10")
    p.add_argument("--target", default="PM10")
    p.add_argument("--seq-len", type=int, default=48)
    p.add_argument("--label-len", type=int, default=24)
    p.add_argument("--pred-len", type=int, default=24)
    p.add_argument(
        "--feature-config",
        default="F1",
        choices=ABLATION_LEVELS,
        help="Which exported CSV layout to read windows from (F1 recommended)",
    )
    p.add_argument(
        "--stations",
        nargs="*",
        default=None,
        help="Subset of station folder names; default = all with data.csv",
    )
    p.add_argument(
        "--results-root",
        default="results/informer2020",
    )
    return p.parse_args()


def discover_stations(zone: int, feature_config: str) -> list[str]:
    base = ROOT / "data" / "training" / f"zone_{zone}" / "informer2020"
    if not base.exists():
        return []
    out = []
    for d in sorted(base.iterdir()):
        if (d / feature_config / "data.csv").exists():
            out.append(d.name)
    return out


def evaluate_station(
    station: str,
    *,
    zone: int,
    feature_config: str,
    target: str,
    seq_len: int,
    label_len: int,
    pred_len: int,
) -> dict:
    data_dir = informer2020_station_dir(zone, station, feature_config)
    if not (data_dir / "data.csv").exists():
        raise FileNotFoundError(data_dir / "data.csv")

    # Univariate windows match F1 protocol (features=S)
    ds = AirQualityDataset(
        root_path=data_dir,
        flag="test",
        size=(seq_len, label_len, pred_len),
        features="S",
        target=target,
        scale=True,
        timeenc=1,
        freq="h",
    )

    preds_last, preds_ma, trues = [], [], []
    for i in range(len(ds)):
        seq_x, seq_y, _, _ = ds[i]
        x = seq_x.numpy() if hasattr(seq_x, "numpy") else np.asarray(seq_x)
        y = seq_y.numpy() if hasattr(seq_y, "numpy") else np.asarray(seq_y)
        # target channel (S → 1 col)
        hist = x[:, -1]
        future = y[-pred_len:, -1]

        last = float(hist[-1])
        ma_hist = float(np.mean(hist[-min(24, len(hist)) :]))

        preds_last.append(np.full(pred_len, last, dtype=np.float64))
        preds_ma.append(np.full(pred_len, ma_hist, dtype=np.float64))
        trues.append(future.astype(np.float64))

    preds_last = np.stack(preds_last)[..., None]  # (n, pred_len, 1)
    preds_ma = np.stack(preds_ma)[..., None]
    trues_arr = np.stack(trues)[..., None]

    # Inverse to raw units (same scaler as Informer)
    preds_last_raw = ds.inverse_transform(preds_last)
    preds_ma_raw = ds.inverse_transform(preds_ma)
    trues_raw = ds.inverse_transform(trues_arr)

    m_last = metric_hourly_and_ma24(preds_last_raw, trues_raw)
    m_ma = metric_hourly_and_ma24(preds_ma_raw, trues_raw)

    return {
        "station": station,
        "n_test_windows": len(ds),
        "persistence_last_hourly_mae": m_last["hourly"]["mae"],
        "persistence_last_hourly_rmse": m_last["hourly"]["rmse"],
        "persistence_last_ma24_mae": m_last["ma24"]["mae"],
        "persistence_last_ma24_rmse": m_last["ma24"]["rmse"],
        "persistence_ma24_hourly_mae": m_ma["hourly"]["mae"],
        "persistence_ma24_hourly_rmse": m_ma["hourly"]["rmse"],
        "persistence_ma24_ma24_mae": m_ma["ma24"]["mae"],
        "persistence_ma24_ma24_rmse": m_ma["ma24"]["rmse"],
    }


def load_informer_summaries(zone_dir: Path) -> pd.DataFrame:
    frames = []
    for cfg in ABLATION_LEVELS:
        path = zone_dir / f"batch_summary_{cfg}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df[df["status"] == "ok"].copy()
        if df.empty:
            continue
        df["feature_config"] = cfg
        for c in ("hourly_mae", "hourly_rmse", "ma24_mae", "ma24_rmse"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_comparison(base: pd.DataFrame, informer: pd.DataFrame) -> pd.DataFrame:
    """One row per station: baselines + each F* ma24_rmse + skill vs persistence_ma24."""
    inf = informer.copy()
    inf["station_key"] = inf["station"].map(station_key)
    rows = []
    for _, b in base.iterrows():
        st = b["station"]
        sk = station_key(st)
        row = b.to_dict()
        row["station"] = sk  # canonical name for fair join with Informer summaries
        for cfg in ABLATION_LEVELS:
            sub = inf[(inf["station_key"] == sk) & (inf["feature_config"] == cfg)]
            if len(sub) == 1:
                row[f"{cfg}_ma24_rmse"] = float(sub.iloc[0]["ma24_rmse"])
                row[f"{cfg}_hourly_rmse"] = float(sub.iloc[0]["hourly_rmse"])
            else:
                row[f"{cfg}_ma24_rmse"] = np.nan
                row[f"{cfg}_hourly_rmse"] = np.nan

        p = row["persistence_ma24_ma24_rmse"]
        for cfg in ABLATION_LEVELS:
            key = f"{cfg}_ma24_rmse"
            if pd.notna(row.get(key)) and p and p > 0:
                # skill score: 1 - model/baseline (positive = better than baseline)
                row[f"{cfg}_skill_vs_pers_ma24"] = 1.0 - float(row[key]) / float(p)
            else:
                row[f"{cfg}_skill_vs_pers_ma24"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def global_means(comp: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "persistence_last_ma24_rmse",
        "persistence_ma24_ma24_rmse",
        *[f"{c}_ma24_rmse" for c in ABLATION_LEVELS],
        *[f"{c}_skill_vs_pers_ma24" for c in ABLATION_LEVELS],
    ]
    present = [c for c in cols if c in comp.columns]
    # only stations with all Informer configs present
    f_cols = [f"{c}_ma24_rmse" for c in ABLATION_LEVELS if f"{c}_ma24_rmse" in comp.columns]
    sub = comp.dropna(subset=f_cols) if f_cols else comp
    means = {c: float(sub[c].mean()) for c in present}
    means["n_stations"] = int(len(sub))
    return pd.DataFrame([means])


def write_md(out_dir: Path, by_station: pd.DataFrame, glob: pd.DataFrame) -> None:
    def fmt(df: pd.DataFrame) -> str:
        show = df.copy()
        for c in show.columns:
            if pd.api.types.is_float_dtype(show[c]):
                show[c] = show[c].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        headers = [str(c) for c in show.columns]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in show.astype(str).values.tolist():
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    keep = [
        "station",
        "persistence_last_ma24_rmse",
        "persistence_ma24_ma24_rmse",
        "F1_ma24_rmse",
        "F2_ma24_rmse",
        "F3_ma24_rmse",
        "F4_ma24_rmse",
        "F5_ma24_rmse",
        "F1_skill_vs_pers_ma24",
    ]
    keep = [c for c in keep if c in by_station.columns]
    lines = [
        "# Persistence baseline vs Informer2020 (zone 2)",
        "",
        "Baselines on the **same test windows** as Informer (seq_len=48, pred_len=24, features=S layout from F1 CSV).",
        "",
        "- **persistence_last**: forecast = last observed hour `y_h` (MA24_pred = `y_h`).",
        "- **persistence_ma24**: forecast = mean of last 24 observed hours "
        "(natural baseline for the MA24 target).",
        "- **skill** = `1 - model_rmse / persistence_ma24_rmse` "
        "(>0 means better than persistence).",
        "",
        "## Global mean (common stations)",
        "",
        fmt(glob),
        "",
        "## By station (MA24 RMSE)",
        "",
        fmt(by_station[keep]),
        "",
    ]
    (out_dir / "PERSISTENCE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    stations = args.stations or discover_stations(args.zone, args.feature_config)
    if not stations:
        raise SystemExit("No stations found")

    zone_dir = Path(args.results_root) / f"zone_{args.zone}"
    out_dir = zone_dir / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for st in stations:
        print(f"Persistence | {st}", flush=True)
        try:
            rows.append(
                evaluate_station(
                    st,
                    zone=args.zone,
                    feature_config=args.feature_config,
                    target=args.target,
                    seq_len=args.seq_len,
                    label_len=args.label_len,
                    pred_len=args.pred_len,
                )
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {st}: {exc}", flush=True)

    base = pd.DataFrame(rows)
    if not base.empty:
        base["station"] = base["station"].map(station_key)
    base.to_csv(out_dir / "persistence_by_station.csv", index=False)

    informer = load_informer_summaries(zone_dir)
    if informer.empty:
        print("No Informer summaries found; wrote persistence only.")
        return

    comp = build_comparison(base, informer)
    comp.to_csv(out_dir / "persistence_vs_informer_by_station.csv", index=False)
    glob = global_means(comp)
    glob.to_csv(out_dir / "persistence_vs_informer_global.csv", index=False)
    write_md(out_dir, comp, glob)

    meta = {
        "zone": args.zone,
        "feature_config_windows": args.feature_config,
        "baselines": ["persistence_last", "persistence_ma24"],
        "n_stations": int(len(comp)),
        "global": glob.to_dict(orient="records")[0],
    }
    (out_dir / "persistence_snapshot.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    print("\n=== GLOBAL (MA24 RMSE) ===")
    print(glob.to_string(index=False))
    print(f"\nWrote {out_dir / 'PERSISTENCE_REPORT.md'}")


if __name__ == "__main__":
    main()
