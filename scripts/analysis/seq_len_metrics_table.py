"""Build model × seq_len metrics table (zone-2 PM10).

Usage:
  python scripts/analysis/seq_len_metrics_table.py
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "comparison" / "zone_2" / "seq_len_sensitivity"


def station_key(name: str) -> str:
    k = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").upper()
    if "DIAZ" in k and "HARO" in k:
        return "M_DIAZ_HARO"
    return k


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs: list[float]) -> float:
  """Sample std (ddof=1) across stations."""
  if len(xs) < 2:
      return float("nan")
  m = mean(xs)
  var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
  return var**0.5


def _station_metrics_from_per_station(ps: list[dict]) -> tuple[list[float], list[float]]:
    maes = [float(x["ma24_mae"]) for x in ps]
    rmses = [float(x["ma24_rmse"]) for x in ps]
    return maes, rmses


def load_airformer() -> list[dict]:
    rows = []
    af_root = ROOT / "results" / "airformer" / "zone_2" / "ZONE2_PM10"
    for seq in (24, 48, 72, 96):
        base = af_root if seq == 48 else af_root / f"seq_len_{seq}"
        for f in ("F1", "F2", "F3", "F4", "F5"):
            p = base / f / "results.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text(encoding="utf-8"))
            # Equal mean over stations (same as TFM ranking), not pooled overall.
            ps = d.get("per_station") or []
            if not ps:
                continue
            maes, rmses = _station_metrics_from_per_station(ps)
            rows.append(
                {
                    "model": "airformer",
                    "seq_len": seq,
                    "F": f,
                    "hourly_mae": mean([float(x["hourly_mae"]) for x in ps]),
                    "hourly_rmse": mean([float(x["hourly_rmse"]) for x in ps]),
                    "ma24_mae": mean(maes),
                    "ma24_rmse": mean(rmses),
                    "ma24_rmse_std": std(rmses),
                    "n_stations": len(ps),
                }
            )
    return rows


def load_batch_model(model: str, root: Path, seqs: tuple[int, ...]) -> list[dict]:
    rows = []
    for seq in seqs:
        base = root if seq == 48 else root / f"seq_len_{seq}"
        for f in ("F1", "F2", "F3", "F4", "F5"):
            p = base / f"batch_summary_{f}.csv"
            if not p.exists():
                continue
            seen: set[str] = set()
            maes: list[float] = []
            rmses: list[float] = []
            hmaes: list[float] = []
            hrmses: list[float] = []
            with p.open(encoding="utf-8", errors="replace") as fh:
                for r in csv.DictReader(fh):
                    status = (r.get("status") or "ok").lower()
                    if status not in ("ok", ""):
                        continue
                    st = r.get("station") or r.get("Station")
                    if not st:
                        continue
                    k = station_key(st)
                    if k in seen:
                        continue
                    seen.add(k)
                    try:
                        maes.append(float(r["ma24_mae"]))
                        rmses.append(float(r["ma24_rmse"]))
                        hmaes.append(float(r["hourly_mae"]))
                        hrmses.append(float(r["hourly_rmse"]))
                    except (KeyError, TypeError, ValueError):
                        continue
            if not rmses:
                continue
            rows.append(
                {
                    "model": model,
                    "seq_len": seq,
                    "F": f,
                    "hourly_mae": mean(hmaes),
                    "hourly_rmse": mean(hrmses),
                    "ma24_mae": mean(maes),
                    "ma24_rmse": mean(rmses),
                    "ma24_rmse_std": std(rmses),
                    "n_stations": len(rmses),
                }
            )
    return rows


def load_gat() -> list[dict]:
    rows = []
    g = ROOT / "results" / "gat_informer_mf_pm10graph" / "zone_2"
    for seq in (24, 48, 72, 96):
        for f in ("F1", "F2", "F3", "F4", "F5"):
            if seq == 48:
                p = g / f / "results.json"
            else:
                p = g / f"seq_len_{seq}" / f / "results.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text(encoding="utf-8"))
            ps = d.get("per_station") or []
            if ps:
                maes, rmses = _station_metrics_from_per_station(ps)
                rows.append(
                    {
                        "model": "gat_informer",
                        "seq_len": int(d.get("seq_len", seq)),
                        "F": f,
                        "hourly_mae": mean([float(x["hourly_mae"]) for x in ps]),
                        "hourly_rmse": mean([float(x["hourly_rmse"]) for x in ps]),
                        "ma24_mae": mean(maes),
                        "ma24_rmse": mean(rmses),
                        "ma24_rmse_std": std(rmses),
                        "n_stations": len(ps),
                    }
                )
                continue
            tm = d.get("test_metrics_raw") or {}
            if "ma24" not in tm:
                continue
            rows.append(
                {
                    "model": "gat_informer",
                    "seq_len": int(d.get("seq_len", seq)),
                    "F": f,
                    "hourly_mae": float(tm["hourly"]["mae"]),
                    "hourly_rmse": float(tm["hourly"]["rmse"]),
                    "ma24_mae": float(tm["ma24"]["mae"]),
                    "ma24_rmse": float(tm["ma24"]["rmse"]),
                    "ma24_rmse_std": float("nan"),
                    "n_stations": len(d.get("stations") or []),
                }
            )
    return rows


def main() -> None:
    rows: list[dict] = []
    rows.extend(load_airformer())
    rows.extend(
        load_batch_model("xgboost", ROOT / "results" / "xgboost" / "zone_2", (24, 48, 72, 96))
    )
    rows.extend(
        load_batch_model(
            "informer", ROOT / "results" / "informer2020" / "zone_2", (24, 48, 72, 96)
        )
    )
    rows.extend(load_gat())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    long_path = OUT_DIR / "metrics_by_model_seqlen.csv"
    keys = [
        "model",
        "seq_len",
        "F",
        "ma24_rmse",
        "ma24_rmse_std",
        "ma24_mae",
        "hourly_rmse",
        "hourly_mae",
        "n_stations",
    ]
    with long_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["model"], x["seq_len"], x["F"])):
            w.writerow({k: r.get(k, "") for k in keys})

    best: dict[tuple, dict] = {}
    for r in rows:
        k = (r["model"], r["seq_len"])
        if k not in best or r["ma24_rmse"] < best[k]["ma24_rmse"]:
            best[k] = r

    best_path = OUT_DIR / "best_F_by_model_seqlen.csv"
    with best_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for k in sorted(best):
            w.writerow({kk: best[k].get(kk, "") for kk in keys})

    print(f"wrote {long_path} ({len(rows)} rows)")
    print(f"wrote {best_path} ({len(best)} rows)")
    print("--- best F by model/seq (MA24 RMSE ± std across stations) ---")
    for k in sorted(best):
        r = best[k]
        print(
            f"{r['model']:12} L={r['seq_len']:2} {r['F']}  "
            f"MA24_RMSE={r['ma24_rmse']:.4f}±{r['ma24_rmse_std']:.2f}  "
            f"MA24_MAE={r['ma24_mae']:.4f}  "
            f"H_RMSE={r['hourly_rmse']:.4f}"
        )

    labels = {
        "airformer": "AirFormer",
        "xgboost": "XGBoost",
        "informer": "Informer2020",
        "gat_informer": "GAT-Informer",
    }
    print("\n--- LaTeX rows (mean ± std, best F) ---")
    for model in ("airformer", "xgboost", "informer", "gat_informer"):
        cells = []
        for seq in (24, 48, 72, 96):
            r = best.get((model, seq))
            if not r:
                cells.append("--")
                continue
            mean_v = r["ma24_rmse"]
            std_v = r["ma24_rmse_std"]
            cell = f"${mean_v:.2f} \\pm {std_v:.2f}$ ({r['F']})"
            if model == "xgboost" and seq == 48:
                cell = f"\\textbf{{{mean_v:.2f} $\\pm$ {std_v:.2f}}} ({r['F']})"
            cells.append(cell)
        print(f"{labels[model]} & " + " & ".join(cells) + " \\\\")


if __name__ == "__main__":
    main()
