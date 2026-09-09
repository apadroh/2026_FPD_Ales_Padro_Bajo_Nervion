"""
Load & compare Informer2020 / AirFormer / GAT-Informer / XGBoost (+ persistence).

Default headline comparison uses learned families present on disk:
  - Informer2020
  - AirFormer
  - GAT-Informer = multi-feature + PM10 similarity graph
    (results/gat_informer_mf_pm10graph/)
  - XGBoost (optional classical baseline; results/xgboost/)

Legacy GAT-v1 / GAT-MF-meteo / GNN→Informer are not included in the
default export (they were internal variants).

Used by the paper-style comparison notebook and as a CLI:

  python src/experiments/compare_all_models.py --zone 2
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.feature_configs import (
    ABLATION_LEVELS,
    ALL_FEATURE_LEVELS,
    LEVEL_DESCRIPTIONS,
)

METRICS = ("hourly_mae", "hourly_rmse", "ma24_mae", "ma24_rmse")
PRIMARY = "ma24_rmse"
# Headline models only (order = plot legend order)
MODEL_ORDER = (
    "informer",
    "airformer",
    "gat_informer",
    "xgboost",
    "persistence_last",
    "persistence_ma24",
)
MODEL_LABELS = {
    "informer": "Informer2020",
    "airformer": "AirFormer",
    "gat_informer": "GAT-Informer",
    "xgboost": "XGBoost",
    "persistence_last": "Persistence (last)",
    "persistence_ma24": "Persistence (MA24)",
}

# Adopted GAT packs: multi-feature + PM10 Adj
GAT_RESULTS_DIR = "gat_informer_mf_pm10graph"


def station_key(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.upper()
    s = re.sub(r"[^A-Z0-9]+", "_", s).strip("_")
    if "DIAZ" in s and "HARO" in s:
        return "M_DIAZ_HARO"
    return s


def load_informer(zone: int) -> pd.DataFrame:
    path = ROOT / f"results/informer2020/zone_{zone}/comparison/metrics_long.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df["status"] == "ok"].copy()
    df["station_key"] = df["station"].map(station_key)
    df["model"] = "informer"
    return df


def load_xgboost(zone: int) -> pd.DataFrame:
    """Per-station XGBoost baselines (results/xgboost/zone_Z/batch_summary_F*.csv)."""
    base = ROOT / f"results/xgboost/zone_{zone}"
    if not base.exists():
        return pd.DataFrame()
    rows = []
    for cfg in ALL_FEATURE_LEVELS:
        path = base / f"batch_summary_{cfg}.csv"
        if path.exists():
            df = pd.read_csv(path)
            if "status" in df.columns:
                df = df[df["status"] == "ok"].copy()
            for _, r in df.iterrows():
                rows.append(
                    {
                        "station": r["station"],
                        "station_key": station_key(r["station"]),
                        "status": "ok",
                        "hourly_mae": float(r["hourly_mae"]),
                        "hourly_rmse": float(r["hourly_rmse"]),
                        "ma24_mae": float(r["ma24_mae"]),
                        "ma24_rmse": float(r["ma24_rmse"]),
                        "feature_config": cfg,
                        "model": "xgboost",
                    }
                )
            continue
        # Fallback: read per-station results.json
        for st_dir in sorted(base.iterdir()):
            if not st_dir.is_dir() or st_dir.name == "comparison":
                continue
            meta = st_dir / cfg / "results.json"
            if not meta.exists():
                continue
            data = json.loads(meta.read_text(encoding="utf-8"))
            raw = data.get("test_metrics_raw") or {}
            hourly = raw.get("hourly") or {}
            ma24 = raw.get("ma24") or {}
            rows.append(
                {
                    "station": data.get("station", st_dir.name),
                    "station_key": station_key(data.get("station", st_dir.name)),
                    "status": "ok",
                    "hourly_mae": float(hourly["mae"]),
                    "hourly_rmse": float(hourly["rmse"]),
                    "ma24_mae": float(ma24["mae"]),
                    "ma24_rmse": float(ma24["rmse"]),
                    "feature_config": cfg,
                    "model": "xgboost",
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Encoding duplicates (e.g. Mª vs mojibake) collapse to the same station_key.
    out = (
        out.sort_values("ma24_rmse")
        .drop_duplicates(subset=["station_key", "feature_config"], keep="first")
        .reset_index(drop=True)
    )
    return out

def _load_multinode_results(base: Path, model: str) -> pd.DataFrame:
    rows = []
    if not base.exists():
        return pd.DataFrame()
    for cfg in ALL_FEATURE_LEVELS:
        meta = base / cfg / "results.json"
        if not meta.exists():
            continue
        data = json.loads(meta.read_text(encoding="utf-8"))
        for ps in data.get("per_station") or []:
            rows.append(
                {
                    "station": ps["station"],
                    "station_key": station_key(ps["station"]),
                    "status": "ok",
                    "hourly_mae": float(ps["hourly_mae"]),
                    "hourly_rmse": float(ps["hourly_rmse"]),
                    "ma24_mae": float(ps["ma24_mae"]),
                    "ma24_rmse": float(ps["ma24_rmse"]),
                    "feature_config": cfg,
                    "model": model,
                }
            )
    return pd.DataFrame(rows)


def load_airformer(zone: int) -> pd.DataFrame:
    return _load_multinode_results(
        ROOT / f"results/airformer/zone_{zone}/ZONE2_PM10", "airformer"
    )


def load_gat(zone: int) -> pd.DataFrame:
    """Adopted GAT-Informer: multi-feature + PM10 similarity graph."""
    return _load_multinode_results(
        ROOT / f"results/{GAT_RESULTS_DIR}/zone_{zone}", "gat_informer"
    )


def load_persistence(zone: int) -> pd.DataFrame:
    path = (
        ROOT
        / f"results/informer2020/zone_{zone}/comparison/persistence_by_station.csv"
    )
    if not path.exists():
        return pd.DataFrame()
    raw = pd.read_csv(path)
    rows = []
    for _, r in raw.iterrows():
        sk = station_key(r["station"])
        for model, prefix in (
            ("persistence_last", "persistence_last"),
            ("persistence_ma24", "persistence_ma24"),
        ):
            rows.append(
                {
                    "station": r["station"],
                    "station_key": sk,
                    "status": "ok",
                    "hourly_mae": r.get(f"{prefix}_hourly_mae"),
                    "hourly_rmse": r.get(f"{prefix}_hourly_rmse"),
                    "ma24_mae": r.get(f"{prefix}_ma24_mae"),
                    "ma24_rmse": r.get(f"{prefix}_ma24_rmse"),
                    "feature_config": "baseline",
                    "model": model,
                }
            )
    return pd.DataFrame(rows)


def load_all(zone: int) -> pd.DataFrame:
    parts = [
        load_informer(zone),
        load_airformer(zone),
        load_gat(zone),
        load_xgboost(zone),
        load_persistence(zone),
    ]
    parts = [p for p in parts if not p.empty]
    if not parts:
        raise SystemExit("No model results found.")
    df = pd.concat(parts, ignore_index=True, sort=False)
    df["model_label"] = df["model"].map(lambda m: MODEL_LABELS.get(m, m))
    return df


def available_models(df: pd.DataFrame) -> list[str]:
    present = set(df["model"].unique())
    return [m for m in MODEL_ORDER if m in present]


def fair_global_table(df: pd.DataFrame, models: list[str] | None = None) -> pd.DataFrame:
    """Mean metrics over stations present for all *learned* models in that F-config."""
    learned = [
        m
        for m in (models or available_models(df))
        if not str(m).startswith("persistence")
    ]
    rows = []
    for cfg in ALL_FEATURE_LEVELS:
        sub = df[df["feature_config"] == cfg]
        if sub.empty:
            continue
        keys = None
        for m in learned:
            ks = set(sub.loc[sub["model"] == m, "station_key"])
            if not ks:
                keys = set()
                break
            keys = ks if keys is None else keys & ks
        if not keys:
            continue
        for m in learned:
            block = sub[(sub["model"] == m) & (sub["station_key"].isin(keys))]
            rows.append(
                {
                    "feature_config": cfg,
                    "description": LEVEL_DESCRIPTIONS.get(cfg, ""),
                    "model": m,
                    "model_label": MODEL_LABELS.get(m, m),
                    "n_stations": len(keys),
                    **{met: float(block[met].mean()) for met in METRICS},
                }
            )
    return pd.DataFrame(rows)


def best_config_per_model(global_df: pd.DataFrame, metric: str = PRIMARY) -> pd.DataFrame:
    if global_df.empty:
        return global_df
    idx = global_df.groupby("model")[metric].idxmin()
    return global_df.loc[idx].sort_values(metric).reset_index(drop=True)


def skill_vs_persistence(
    global_df: pd.DataFrame,
    pers_df: pd.DataFrame,
    metric: str = PRIMARY,
    baseline: str = "persistence_ma24",
) -> pd.DataFrame:
    """Skill = 1 - RMSE_model / RMSE_baseline (positive = better than baseline)."""
    if pers_df.empty or global_df.empty:
        return pd.DataFrame()
    base = float(pers_df.loc[pers_df["model"] == baseline, metric].mean())
    if not np.isfinite(base) or base <= 0:
        return pd.DataFrame()
    out = global_df.copy()
    out["baseline"] = baseline
    out["baseline_rmse"] = base
    out["skill"] = 1.0 - out[metric] / base
    out["improvement_pct"] = 100.0 * out["skill"]
    return out


def wins_by_station(df: pd.DataFrame, metric: str = PRIMARY) -> pd.DataFrame:
    """For each F-config, which learned model wins most stations."""
    learned = [m for m in available_models(df) if not m.startswith("persistence")]
    rows = []
    for cfg in ALL_FEATURE_LEVELS:
        sub = df[df["feature_config"] == cfg]
        wide = sub.pivot_table(
            index="station_key", columns="model", values=metric, aggfunc="mean"
        )
        cols = [c for c in learned if c in wide.columns]
        if len(cols) < 2:
            continue
        w = wide[cols].dropna()
        if w.empty:
            continue
        winners = w.idxmin(axis=1)
        counts = winners.value_counts().to_dict()
        row = {
            "feature_config": cfg,
            "n_stations": int(len(w)),
            **{f"{m}_wins": int(counts.get(m, 0)) for m in cols},
            **{f"{m}_mean": float(w[m].mean()) for m in cols},
        }
        rows.append(row)
    return pd.DataFrame(rows)


def per_station_best(df: pd.DataFrame, metric: str = PRIMARY) -> pd.DataFrame:
    """Best (model, config) per station among learned models."""
    learned = df[~df["model"].str.startswith("persistence")].copy()
    if learned.empty:
        return pd.DataFrame()
    idx = learned.groupby("station_key")[metric].idxmin()
    best = learned.loc[idx, ["station", "station_key", "model", "feature_config", *METRICS]]
    return best.sort_values(metric).reset_index(drop=True)


def oracle_routing_analysis(
    df: pd.DataFrame,
    metric: str = PRIMARY,
) -> dict:
    """
    'Oracle' = at each station pick the best learned (model, F*) on the same
    metric (typically test MA24 RMSE). Upper bound on station-wise routing.

    Compares mean metric of that selection vs each single global champion
    (best F* of each model, evaluated on the common station set).
    """
    learned = df[~df["model"].str.startswith("persistence")].copy()
    if learned.empty:
        return {"oracle": pd.DataFrame(), "summary": pd.DataFrame(), "wins": pd.DataFrame()}

    oracle = per_station_best(learned, metric)
    common = set(oracle["station_key"])

    # Restrict to stations present for all learned models (fair mean)
    models = [m for m in available_models(learned) if not str(m).startswith("persistence")]
    for m in models:
        common &= set(learned.loc[learned["model"] == m, "station_key"])
    oracle = oracle[oracle["station_key"].isin(common)].copy()
    if oracle.empty:
        return {"oracle": pd.DataFrame(), "summary": pd.DataFrame(), "wins": pd.DataFrame()}

    oracle_mean = float(oracle[metric].mean())
    rows = [
        {
            "strategy": "oracle_best_per_station",
            "model": "oracle",
            "model_label": "Oracle (best per station)",
            "feature_config": "mixed",
            "n_stations": int(len(oracle)),
            metric: oracle_mean,
            "vs_oracle_delta": 0.0,
            "improvement_vs_oracle_pct": 0.0,
        }
    ]

    # Best global F* per model, mean over the same stations
    for m in models:
        sub = learned[(learned["model"] == m) & (learned["station_key"].isin(common))]
        # pick config that minimizes mean metric over common stations
        cfg_means = sub.groupby("feature_config")[metric].mean()
        best_cfg = cfg_means.idxmin()
        mean_m = float(cfg_means.loc[best_cfg])
        # per-station values under that config
        under = sub[sub["feature_config"] == best_cfg].set_index("station_key")[metric]
        under = under[~under.index.duplicated(keep="first")]
        under = under.reindex(oracle["station_key"]).astype(float)
        rows.append(
            {
                "strategy": "single_model_best_F",
                "model": m,
                "model_label": MODEL_LABELS.get(m, m),
                "feature_config": best_cfg,
                "n_stations": int(under.notna().sum()),
                metric: mean_m,
                "vs_oracle_delta": mean_m - oracle_mean,
                "improvement_vs_oracle_pct": 100.0 * (1.0 - oracle_mean / mean_m)
                if mean_m > 0
                else np.nan,
            }
        )

    summary = pd.DataFrame(rows).sort_values(metric).reset_index(drop=True)

    wins = (
        oracle.groupby(["model", "feature_config"])
        .size()
        .reset_index(name="n_stations_won")
        .sort_values("n_stations_won", ascending=False)
    )
    wins["model_label"] = wins["model"].map(lambda m: MODEL_LABELS.get(m, m))

    # Persistence baseline mean on same stations if available
    pers = df[
        (df["model"] == "persistence_ma24") & (df["station_key"].isin(common))
    ]
    pers_mean = float(pers[metric].mean()) if not pers.empty else np.nan
    if np.isfinite(pers_mean) and pers_mean > 0:
        summary["skill_vs_pers_ma24"] = 1.0 - summary[metric] / pers_mean
        summary["improvement_vs_pers_pct"] = 100.0 * summary["skill_vs_pers_ma24"]
        oracle_skill = 1.0 - oracle_mean / pers_mean
    else:
        summary["skill_vs_pers_ma24"] = np.nan
        summary["improvement_vs_pers_pct"] = np.nan
        oracle_skill = np.nan

    return {
        "oracle": oracle,
        "summary": summary,
        "wins": wins,
        "oracle_mean": oracle_mean,
        "pers_mean": pers_mean,
        "oracle_skill": oracle_skill,
        "n_stations": int(len(oracle)),
    }


def plot_oracle_vs_single(
    summary: pd.DataFrame,
    metric: str,
    out_path: Path,
) -> Path | None:
    if summary.empty:
        return None
    # Match realistic panel size / left margin so stacked figures align in LaTeX.
    fig, ax = plt.subplots(figsize=(8, 4.2))
    order = summary.sort_values(metric)
    labels = [
        "Oracle (per station)"
        if r.model == "oracle"
        else f"{r.model_label} ({r.feature_config})"
        for r in order.itertuples()
    ]
    colors = ["#2a9d8f" if m == "oracle" else "#264653" for m in order["model"]]
    ax.barh(labels, order[metric], color=colors)
    ax.set_xlabel(metric.replace("_", " ").upper())
    ax.set_title("Oracle (best per station) vs single model (best F)")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.30, right=0.98, top=0.88, bottom=0.16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _series_rmse(pred: np.ndarray, true: np.ndarray) -> float:
    p = np.asarray(pred, dtype=float).reshape(-1)
    t = np.asarray(true, dtype=float).reshape(-1)
    mask = np.isfinite(p) & np.isfinite(t)
    if not mask.any():
        return float("nan")
    err = p[mask] - t[mask]
    return float(np.sqrt(np.mean(err**2)))


def _load_all_ma24_halves(
    zone: int,
    learned: pd.DataFrame,
    select_frac: float = 0.5,
) -> pd.DataFrame:
    """
    For each (model, F*, station) compute MA24 RMSE on the first and second
    chronological halves of the *test* series.

    Proxy for val→test selection when per-station val metrics were not saved:
    choose on half A, score on half B (same holdout for all strategies).
    """
    rows: list[dict] = []
    # Cache multinode arrays: (model, config) -> (stations, pred, true)
    cache: dict[tuple[str, str], tuple[list[str], np.ndarray, np.ndarray]] = {}

    combos = (
        learned[["model", "feature_config", "station", "station_key"]]
        .drop_duplicates()
        .itertuples(index=False)
    )
    for model, cfg, station, sk in combos:
        if model in ("airformer", "gat_informer"):
            key = (model, cfg)
            if key not in cache:
                if model == "airformer":
                    d = ROOT / f"results/airformer/zone_{zone}/ZONE2_PM10" / cfg
                else:
                    d = ROOT / f"results/{GAT_RESULTS_DIR}/zone_{zone}" / cfg
                pred_p, true_p, meta = (
                    d / "test_preds_ma24.npy",
                    d / "test_trues_ma24.npy",
                    d / "results.json",
                )
                if not pred_p.exists() or not true_p.exists() or not meta.exists():
                    cache[key] = ([], np.array([]), np.array([]))
                else:
                    cache[key] = (
                        _stations_from_multinode(meta),
                        np.load(pred_p),
                        np.load(true_p),
                    )
            stations, pred_all, true_all = cache[key]
            if not stations:
                continue
            idx = next(
                (i for i, s in enumerate(stations) if station_key(s) == sk), None
            )
            if idx is None:
                continue
            pred = pred_all[:, 0, idx, 0].reshape(-1)
            true = true_all[:, 0, idx, 0].reshape(-1)
        else:
            pair = load_ma24_pred_true(model, zone, cfg, station)
            if pair is None:
                continue
            pred, true = pair

        n = min(len(pred), len(true))
        if n < 20:
            continue
        cut = max(1, int(n * select_frac))
        if cut >= n:
            continue
        rows.append(
            {
                "station": station,
                "station_key": sk,
                "model": model,
                "feature_config": cfg,
                "n_points": n,
                "select_rmse": _series_rmse(pred[:cut], true[:cut]),
                "eval_rmse": _series_rmse(pred[cut:], true[cut:]),
            }
        )
    return pd.DataFrame(rows)


def realistic_routing_analysis(
    zone: int,
    df: pd.DataFrame,
    select_frac: float = 0.5,
) -> dict:
    """
    Deployable-style station routing (proxy):
      - select best (model, F*) per station on first half of test MA24
      - evaluate that fixed choice on second half

    Compared on the same second-half holdout against:
      - single global champion (best F* of each model, chosen on first-half *mean*)
      - oracle on second half (upper bound; peeks at eval)

    Note: true validation metrics were not saved per station; chronological
    half-split of test is an honest proxy of the idea.
    """
    learned = df[~df["model"].str.startswith("persistence")].copy()
    empty = {
        "halves": pd.DataFrame(),
        "selection": pd.DataFrame(),
        "summary": pd.DataFrame(),
        "agreement": {},
    }
    if learned.empty:
        return empty

    halves = _load_all_ma24_halves(zone, learned, select_frac=select_frac)
    if halves.empty:
        return empty

    models = [m for m in available_models(learned) if not str(m).startswith("persistence")]
    common = set(halves["station_key"])
    for m in models:
        common &= set(halves.loc[halves["model"] == m, "station_key"])
    halves = halves[halves["station_key"].isin(common)].copy()
    if halves.empty:
        return empty

    # Per station: realistic pick (min select_rmse) and oracle pick (min eval_rmse)
    idx_sel = halves.groupby("station_key")["select_rmse"].idxmin()
    idx_ora = halves.groupby("station_key")["eval_rmse"].idxmin()
    selection = halves.loc[idx_sel].copy()
    selection = selection.rename(
        columns={
            "model": "sel_model",
            "feature_config": "sel_config",
            "select_rmse": "sel_half_rmse",
            "eval_rmse": "deployed_eval_rmse",
        }
    )
    oracle_holdout = halves.loc[idx_ora][
        ["station_key", "model", "feature_config", "eval_rmse"]
    ].rename(
        columns={
            "model": "ora_model",
            "feature_config": "ora_config",
            "eval_rmse": "oracle_eval_rmse",
        }
    )
    selection = selection.merge(oracle_holdout, on="station_key", how="left")
    selection["agree_with_oracle"] = (
        (selection["sel_model"] == selection["ora_model"])
        & (selection["sel_config"] == selection["ora_config"])
    )
    selection["model_label"] = selection["sel_model"].map(
        lambda m: MODEL_LABELS.get(m, m)
    )

    realistic_mean = float(selection["deployed_eval_rmse"].mean())
    oracle_mean = float(selection["oracle_eval_rmse"].mean())

    rows = [
        {
            "strategy": "oracle_on_eval_half",
            "model": "oracle",
            "model_label": "Oracle (chooses using 2nd half)",
            "feature_config": "mixed",
            "n_stations": int(len(selection)),
            "eval_rmse": oracle_mean,
        },
        {
            "strategy": "select_on_first_half",
            "model": "router_val_proxy",
            "model_label": "Realistic selector (choose on 1st half, eval on 2nd)",
            "feature_config": "mixed",
            "n_stations": int(len(selection)),
            "eval_rmse": realistic_mean,
        },
    ]

    # Single-model baselines: pick best F* by mean select_rmse, score mean eval_rmse
    for m in models:
        sub = halves[halves["model"] == m]
        cfg_sel = sub.groupby("feature_config")["select_rmse"].mean()
        best_cfg = cfg_sel.idxmin()
        under = sub[sub["feature_config"] == best_cfg].set_index("station_key")[
            "eval_rmse"
        ]
        under = under.reindex(selection["station_key"]).astype(float)
        rows.append(
            {
                "strategy": "single_model_best_F",
                "model": m,
                "model_label": MODEL_LABELS.get(m, m),
                "feature_config": best_cfg,
                "n_stations": int(under.notna().sum()),
                "eval_rmse": float(under.mean()),
            }
        )

    summary = pd.DataFrame(rows).sort_values("eval_rmse").reset_index(drop=True)
    best_single = float(
        summary.loc[summary["strategy"] == "single_model_best_F", "eval_rmse"].min()
    )
    summary["vs_best_single_delta"] = summary["eval_rmse"] - best_single
    summary["vs_oracle_delta"] = summary["eval_rmse"] - oracle_mean

    wins = (
        selection.groupby(["sel_model", "sel_config"])
        .size()
        .reset_index(name="n_stations_chosen")
        .sort_values("n_stations_chosen", ascending=False)
    )
    wins["model_label"] = wins["sel_model"].map(lambda m: MODEL_LABELS.get(m, m))

    agree = float(selection["agree_with_oracle"].mean()) if len(selection) else np.nan
    return {
        "halves": halves,
        "selection": selection.sort_values("deployed_eval_rmse").reset_index(drop=True),
        "summary": summary,
        "wins": wins,
        "agreement_pct": 100.0 * agree if np.isfinite(agree) else np.nan,
        "realistic_mean": realistic_mean,
        "oracle_mean": oracle_mean,
        "best_single_mean": best_single,
        "n_stations": int(len(selection)),
        "select_frac": select_frac,
        "note": (
            "Proxy: no per-station validation MA24 saved; "
            "choose on the first temporal half of test, evaluate on the second."
        ),
    }


def plot_realistic_vs_single(
    summary: pd.DataFrame,
    out_path: Path,
) -> Path | None:
    if summary.empty:
        return None
    # Same canvas as oracle panel (long CSV labels caused LaTeX misalignment).
    fig, ax = plt.subplots(figsize=(8, 4.2))
    order = summary.sort_values("eval_rmse")
    labels = []
    colors = []
    for r in order.itertuples():
        if r.model == "oracle":
            labels.append("Oracle (2nd half)")
            colors.append("#2a9d8f")
        elif r.model == "router_val_proxy":
            labels.append("Realistic selector")
            colors.append("#e9c46a")
        else:
            labels.append(f"{r.model_label} ({r.feature_config})")
            colors.append("#264653")
    ax.barh(labels, order["eval_rmse"], color=colors)
    ax.set_xlabel("MA24 RMSE (2nd half of test)")
    ax.set_title("Realistic selector vs oracle vs single model")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.30, right=0.98, top=0.88, bottom=0.16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


# ----- plots -----


def plot_grouped_bars(
    global_df: pd.DataFrame,
    metric: str,
    out_path: Path,
    title: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.2))
    configs = [c for c in ABLATION_LEVELS if c in set(global_df["feature_config"])]
    models = [m for m in available_models(global_df) if not m.startswith("persistence")]
    x = np.arange(len(configs))
    width = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        vals = []
        for c in configs:
            block = global_df[(global_df["feature_config"] == c) & (global_df["model"] == m)]
            vals.append(float(block[metric].iloc[0]) if len(block) else np.nan)
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=MODEL_LABELS.get(m, m))
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylabel(metric.replace("_", " ").upper())
    ax.set_xlabel("Feature config")
    if title:
        ax.set_title(title)
    # Legend above the axes (outside bars) so labels stay readable in email/PDF.
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=min(len(models), 4),
        frameon=True,
        fancybox=False,
        edgecolor="#cccccc",
        fontsize=9,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_skill_bars(skill_df: pd.DataFrame, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    # best config per model
    best = skill_df.loc[skill_df.groupby("model")["skill"].idxmax()]
    labels = [MODEL_LABELS.get(m, m) for m in best["model"]]
    colors = ["#2a6f97" if s >= 0 else "#9b2226" for s in best["skill"]]
    ax.barh(labels, best["improvement_pct"], color=colors)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("Skill vs persistence MA24 (%)")
    ax.set_title("Forecast skill (best F-config per model)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_station_heatmap(
    df: pd.DataFrame,
    metric: str,
    out_path: Path,
    config: str = "best",
) -> Path | None:
    """Heatmap station × model. config='best' uses each model's best F."""
    learned = df[~df["model"].str.startswith("persistence")].copy()
    if learned.empty:
        return None
    if config == "best":
        idx = learned.groupby(["model", "station_key"])[metric].idxmin()
        learned = learned.loc[idx]
        title = f"{metric} by station (best F per model)"
    else:
        learned = learned[learned["feature_config"] == config]
        title = f"{metric} by station ({config})"
    wide = learned.pivot_table(
        index="station_key", columns="model", values=metric, aggfunc="mean"
    )
    cols = [c for c in available_models(df) if c in wide.columns and not c.startswith("persistence")]
    wide = wide[cols].dropna(how="all")
    if wide.empty:
        return None
    wide = wide.rename(columns={c: MODEL_LABELS.get(c, c) for c in wide.columns})
    fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(wide))))
    data = wide.to_numpy(dtype=float)
    im = ax.imshow(data, aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(wide.columns)))
    ax.set_xticklabels(list(wide.columns), rotation=30, ha="right")
    ax.set_yticks(range(len(wide.index)))
    ax.set_yticklabels(list(wide.index))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label=metric)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Station")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


# ----- pred vs reality -----


def _stations_from_multinode(results_json: Path) -> list[str]:
    if not results_json.exists():
        return []
    data = json.loads(results_json.read_text(encoding="utf-8"))
    return list(data.get("stations") or [])


def _find_informer_station_dir(zone: int, station: str, config: str) -> Path | None:
    base = ROOT / f"results/informer2020/zone_{zone}"
    if not base.exists():
        return None
    candidates = [
        base / station / config,
        base / station.replace(" ", "_") / config,
        base / station.replace(" ", "_").replace("(", "").replace(")", "") / config,
    ]
    sk = station_key(station)
    for d in sorted(base.iterdir()):
        if d.is_dir() and station_key(d.name) == sk:
            candidates.insert(0, d / config)
    for c in candidates:
        if (c / "test_preds_ma24.npy").exists():
            return c
    return None


def load_ma24_pred_true(
    model: str,
    zone: int,
    config: str,
    station: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return 1-D MA24 pred/true for one station, or None if missing."""
    if model == "informer":
        d = _find_informer_station_dir(zone, station, config)
        if d is None:
            return None
        return (
            np.load(d / "test_preds_ma24.npy").reshape(-1),
            np.load(d / "test_trues_ma24.npy").reshape(-1),
        )

    if model == "xgboost":
        d = ROOT / f"results/xgboost/zone_{zone}"
        sk = station_key(station)
        hit = None
        if d.exists():
            for st_dir in d.iterdir():
                if st_dir.is_dir() and station_key(st_dir.name) == sk:
                    hit = st_dir / config
                    break
        if hit is None or not (hit / "test_preds_ma24.npy").exists():
            return None
        return (
            np.load(hit / "test_preds_ma24.npy").reshape(-1),
            np.load(hit / "test_trues_ma24.npy").reshape(-1),
        )

    if model == "airformer":
        d = ROOT / f"results/airformer/zone_{zone}/ZONE2_PM10" / config
    elif model == "gat_informer":
        d = ROOT / f"results/{GAT_RESULTS_DIR}/zone_{zone}" / config
    else:
        return None

    pred_p, true_p, meta = d / "test_preds_ma24.npy", d / "test_trues_ma24.npy", d / "results.json"
    if not pred_p.exists() or not true_p.exists():
        return None
    stations = _stations_from_multinode(meta)
    sk = station_key(station)
    idx = next((i for i, s in enumerate(stations) if station_key(s) == sk), None)
    if idx is None:
        return None
    pred, true = np.load(pred_p), np.load(true_p)
    return pred[:, 0, idx, 0].reshape(-1), true[:, 0, idx, 0].reshape(-1)


def plot_pred_vs_reality_interactive(
    pred: np.ndarray,
    true: np.ndarray,
    out_path: Path,
    *,
    title: str = "",
) -> Path | None:
    """Interactive HTML (Plotly): zoom / pan / rangeslider on test MA24 series."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("  skip interactive HTML (plotly not installed; PNG still written)")
        return None

    y_hat = np.asarray(pred, dtype=float).reshape(-1)
    y = np.asarray(true, dtype=float).reshape(-1)
    x = np.arange(len(y))
    mae = float(np.mean(np.abs(y_hat - y)))
    rmse = float(np.sqrt(np.mean((y_hat - y) ** 2)))
    err = y_hat - y

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"{title}  ·  MAE={mae:.2f}  RMSE={rmse:.2f}", "Error (pred − real)"),
    )
    fig.add_trace(
        go.Scatter(x=x, y=y, name="realidad", line=dict(color="#222222", width=1.5)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_hat,
            name="predicción",
            line=dict(color="#d62728", width=1.2),
            opacity=0.9,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=err,
            name="error",
            fill="tozeroy",
            line=dict(color="#9467bd", width=0.8),
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        height=620,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=50, r=20, t=80, b=40),
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Ventana de test", rangeslider=dict(visible=True), row=2, col=1)
    fig.update_yaxes(title_text="PM10 MA24", row=1, col=1)
    fig.update_yaxes(title_text="Error", row=2, col=1)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs=True, full_html=True)
    return out_path


def plot_pred_vs_reality_series(
    pred: np.ndarray,
    true: np.ndarray,
    out_path: Path,
    *,
    title: str = "",
    max_points: int = 600,
) -> Path:
    y_hat = np.asarray(pred, dtype=float).reshape(-1)
    y = np.asarray(true, dtype=float).reshape(-1)
    n = len(y)
    if n == 0:
        raise ValueError("Empty series")
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points, dtype=int)
        y_hat_s, y_s, x = y_hat[idx], y[idx], idx
        subsampled = True
    else:
        y_hat_s, y_s, x = y_hat, y, np.arange(n)
        subsampled = False

    mae = float(np.mean(np.abs(y_hat - y)))
    rmse = float(np.sqrt(np.mean((y_hat - y) ** 2)))

    fig, axes = plt.subplots(
        3, 1, figsize=(10, 8.5), gridspec_kw={"height_ratios": [1.2, 0.7, 1.0]}
    )
    axes[0].plot(x, y_s, color="#222222", lw=1.2, label="realidad")
    axes[0].plot(x, y_hat_s, color="#d62728", lw=1.0, alpha=0.85, label="prediction")
    axes[0].set_ylabel("PM10 MA24")
    axes[0].set_title(f"{title}\nMAE={mae:.2f}  RMSE={rmse:.2f}")
    axes[0].legend(loc="upper right", frameon=False)
    axes[0].grid(True, alpha=0.3)

    err = y_hat_s - y_s
    axes[1].fill_between(x, err, 0, color="#9467bd", alpha=0.35)
    axes[1].plot(x, err, color="#9467bd", lw=0.8)
    axes[1].axhline(0.0, color="k", lw=0.8)
    axes[1].set_ylabel("Error")
    axes[1].set_xlabel("Test window" + (" (subsample)" if subsampled else ""))
    axes[1].grid(True, alpha=0.3)

    lo, hi = float(min(y.min(), y_hat.min())), float(max(y.max(), y_hat.max()))
    axes[2].scatter(y, y_hat, s=8, alpha=0.25, c="#1f77b4", edgecolors="none")
    axes[2].plot([lo, hi], [lo, hi], "k--", lw=1.0, label="y = x")
    axes[2].set_xlabel("Realidad")
    axes[2].set_ylabel("Prediction")
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].legend(frameon=False)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_informer_loss_curves(
    zone: int, station: str, config: str, out_path: Path
) -> Path | None:
    d = _find_informer_station_dir(zone, station, config)
    if d is None:
        return None
    meta = d / "results.json"
    if not meta.exists():
        return None
    history = json.loads(meta.read_text(encoding="utf-8")).get("history") or []
    if not history:
        return None
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(8, 4))
    if "train_loss" in history[0]:
        ax.plot(epochs, [h["train_loss"] for h in history], "o-", label="train")
        ax.plot(epochs, [h["val_loss"] for h in history], "s-", label="val")
        if history[0].get("test_loss") is not None:
            ax.plot(epochs, [h.get("test_loss") for h in history], "^-", label="test")
        ax.set_ylabel("Loss (MSE)")
    else:
        ax.plot(epochs, [h.get("train_mae", np.nan) for h in history], "o-", label="train")
        ax.plot(epochs, [h.get("val_mae", np.nan) for h in history], "s-", label="val")
        ax.set_ylabel("MAE")
    ax.set_xlabel("Epoch")
    ax.set_title(f"Informer loss — {station} / {config}")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_best_model_per_station(
    zone: int,
    station_best: pd.DataFrame,
    out_dir: Path,
    *,
    max_points: int = 500,
) -> list[Path]:
    """One pred-vs-reality figure per station using that station's best (model, F*)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if station_best.empty:
        return paths
    for _, row in station_best.iterrows():
        station = str(row["station"])
        m = str(row["model"])
        cfg = str(row["feature_config"])
        loaded = load_ma24_pred_true(m, zone, cfg, station)
        if loaded is None:
            print(f"  skip BEST pred/true {m}/{cfg}/{station}")
            continue
        pred, true = loaded
        label = MODEL_LABELS.get(m, m)
        sk = station_key(station)
        p = out_dir / f"fig_pred_vs_real_BEST_{sk}_{m}_{cfg}.png"
        plot_pred_vs_reality_series(
            pred,
            true,
            p,
            title=f"{station} — best: {label} {cfg}",
            max_points=max_points,
        )
        paths.append(p)
        print(f"  BEST {sk}: {label} {cfg} -> {p.name}")
    return paths


def plot_pred_vs_reality_all_models(
    zone: int,
    station: str,
    best: pd.DataFrame,
    out_dir: Path,
    *,
    max_points: int = 500,
) -> list[Path]:
    """Pred-vs-reality per learned model (best F*) + overview + Informer loss."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    panels: list[tuple[str, np.ndarray, np.ndarray]] = []

    for _, row in best.iterrows():
        m, cfg = row["model"], row["feature_config"]
        loaded = load_ma24_pred_true(m, zone, cfg, station)
        if loaded is None:
            print(f"  skip pred/true {m}/{cfg}/{station}")
            continue
        pred, true = loaded
        label = MODEL_LABELS.get(m, m)
        p = out_dir / f"fig_pred_vs_real_{m}_{cfg}_{station_key(station)}.png"
        plot_pred_vs_reality_series(
            pred,
            true,
            p,
            title=f"{label} · {cfg} · {station} (test MA24)",
            max_points=max_points,
        )
        paths.append(p)
        html_p = out_dir / f"fig_pred_vs_real_{m}_{cfg}_{station_key(station)}.html"
        got_html = plot_pred_vs_reality_interactive(
            pred,
            true,
            html_p,
            title=f"{label} · {cfg} · {station} (test MA24)",
        )
        if got_html is not None:
            paths.append(got_html)
        panels.append((f"{label}\n{cfg}", pred, true))
        if m == "informer":
            lp = out_dir / f"fig_loss_informer_{cfg}_{station_key(station)}.png"
            got = plot_informer_loss_curves(zone, station, cfg, lp)
            if got is not None:
                paths.append(got)

    if panels:
        n = len(panels)
        fig, axes = plt.subplots(n, 1, figsize=(10, 2.4 * n), sharex=False)
        if n == 1:
            axes = [axes]
        for ax, (lab, pred, true) in zip(axes, panels):
            y_hat, y = pred.reshape(-1), true.reshape(-1)
            if len(y) > max_points:
                idx = np.linspace(0, len(y) - 1, max_points, dtype=int)
                y_hat, y, x = y_hat[idx], y[idx], idx
            else:
                x = np.arange(len(y))
            ax.plot(x, y, color="#222", lw=1.0, label="realidad")
            ax.plot(x, y_hat, color="#d62728", lw=0.9, alpha=0.85, label="prediction")
            rmse = float(np.sqrt(np.mean((pred.reshape(-1) - true.reshape(-1)) ** 2)))
            ax.set_ylabel("PM10 MA24")
            ax.set_title(f"{lab}  RMSE={rmse:.2f}")
            ax.legend(loc="upper right", frameon=False, fontsize=8)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Test window")
        fig.suptitle(f"Prediction vs ground truth — {station}", y=1.01)
        fig.tight_layout()
        overview = out_dir / f"fig_pred_vs_real_overview_{station_key(station)}.png"
        fig.savefig(overview, dpi=140, bbox_inches="tight")
        plt.close(fig)
        paths.append(overview)

        # Interactive overview (all models, zoom/pan)
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig_h = make_subplots(
                rows=n,
                cols=1,
                shared_xaxes=False,
                vertical_spacing=0.06,
                subplot_titles=[lab.replace("\n", " · ") for lab, _, _ in panels],
            )
            for i, (lab, pred, true) in enumerate(panels, start=1):
                y_hat, y = pred.reshape(-1), true.reshape(-1)
                x = np.arange(len(y))
                fig_h.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        name="realidad",
                        line=dict(color="#222", width=1.2),
                        showlegend=(i == 1),
                        legendgroup="real",
                    ),
                    row=i,
                    col=1,
                )
                fig_h.add_trace(
                    go.Scatter(
                        x=x,
                        y=y_hat,
                        name="predicción",
                        line=dict(color="#d62728", width=1.0),
                        showlegend=(i == 1),
                        legendgroup="pred",
                    ),
                    row=i,
                    col=1,
                )
                fig_h.update_yaxes(title_text="PM10 MA24", row=i, col=1)
            fig_h.update_layout(
                height=280 * n,
                template="plotly_white",
                title=f"Predicción vs realidad — {station} (zoom/pan)",
                hovermode="x unified",
                legend=dict(orientation="h", y=1.02),
            )
            fig_h.update_xaxes(rangeslider=dict(visible=True), row=n, col=1)
            overview_html = (
                out_dir / f"fig_pred_vs_real_overview_{station_key(station)}.html"
            )
            fig_h.write_html(
                str(overview_html), include_plotlyjs=True, full_html=True
            )
            paths.append(overview_html)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip interactive overview: {exc}")

    return paths


def plot_improvement_over_baseline(
    global_df: pd.DataFrame,
    pers_mean: float,
    metric: str,
    out_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    best = best_config_per_model(global_df, metric)
    labels = [
        f"{MODEL_LABELS.get(r.model, r.model)}\n({r.feature_config})"
        for r in best.itertuples()
    ]
    vals = 100.0 * (1.0 - best[metric] / pers_mean)
    ax.bar(labels, vals, color="#1d3557")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylabel(f"% improvement vs persistence ({metric})")
    ax.set_title("Relative improvement over persistence MA24 baseline")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def write_markdown_report(
    out_dir: Path,
    global_df: pd.DataFrame,
    best: pd.DataFrame,
    skill: pd.DataFrame,
    wins: pd.DataFrame,
    station_best: pd.DataFrame,
    oracle: dict | None = None,
    realistic: dict | None = None,
) -> Path:
    lines = [
        "# Multi-model comparison (zone 2 · PM10)",
        "",
        "Protocol: `seq_len=48`, horizon=24, primary metric **MA24 RMSE** (lower better).",
        "Fair global means use stations common to all *available learned models* within each F-config.",
        "",
        "## Best configuration per model",
        "",
    ]
    if not best.empty:
        show = best[
            [
                "model_label",
                "feature_config",
                "n_stations",
                "hourly_mae",
                "hourly_rmse",
                "ma24_mae",
                "ma24_rmse",
            ]
        ].copy()
        for c in METRICS:
            show[c] = show[c].map(lambda x: f"{x:.4f}")
        lines.append(show.to_string(index=False))
    lines += ["", "## Global mean (all F × model)", ""]
    if not global_df.empty:
        g = global_df.copy()
        for c in METRICS:
            g[c] = g[c].map(lambda x: f"{x:.4f}")
        lines.append(
            g[
                [
                    "feature_config",
                    "model_label",
                    "n_stations",
                    *METRICS,
                ]
            ].to_string(index=False)
        )
    if not skill.empty:
        lines += ["", "## Skill vs persistence MA24", ""]
        s = skill.copy()
        s["skill"] = s["skill"].map(lambda x: f"{x:.4f}")
        s["improvement_pct"] = s["improvement_pct"].map(lambda x: f"{x:.2f}%")
        lines.append(
            s[
                [
                    "model_label",
                    "feature_config",
                    "ma24_rmse",
                    "skill",
                    "improvement_pct",
                ]
            ].to_string(index=False)
        )
    if not wins.empty:
        lines += ["", "## Station wins by F-config", "", wins.to_string(index=False)]
    if not station_best.empty:
        lines += [
            "",
            "## Best model×config per station (MA24 RMSE)",
            "",
            station_best.head(20).to_string(index=False),
        ]
    if oracle and not oracle.get("summary", pd.DataFrame()).empty:
        lines += [
            "",
            "## Oracle routing (best model×F per station)",
            "",
            "Upper bound: at each station pick the lowest test MA24 RMSE among learned models.",
            "Not a deployable selector (uses test labels); use as a ceiling vs single global model.",
            "",
        ]
        sm = oracle["summary"].copy()
        for c in (PRIMARY, "vs_oracle_delta", "improvement_vs_oracle_pct"):
            if c in sm.columns:
                sm[c] = sm[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
        cols = [
            c
            for c in (
                "model_label",
                "feature_config",
                "n_stations",
                PRIMARY,
                "vs_oracle_delta",
                "improvement_vs_oracle_pct",
                "improvement_vs_pers_pct",
            )
            if c in sm.columns
        ]
        lines.append(sm[cols].to_string(index=False))
        if not oracle.get("wins", pd.DataFrame()).empty:
            lines += ["", "### Who wins how many stations", "", oracle["wins"].to_string(index=False)]
    if realistic and not realistic.get("summary", pd.DataFrame()).empty:
        lines += [
            "",
            "## Realistic station routing (proxy: 1st half → 2nd half)",
            "",
            str(realistic.get("note", "")),
            "",
            f"Agreement with oracle on holdout: {realistic.get('agreement_pct', float('nan')):.1f}% of stations.",
            "",
        ]
        sm = realistic["summary"].copy()
        for c in ("eval_rmse", "vs_best_single_delta", "vs_oracle_delta"):
            if c in sm.columns:
                sm[c] = sm[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
        cols = [
            c
            for c in (
                "model_label",
                "feature_config",
                "n_stations",
                "eval_rmse",
                "vs_best_single_delta",
                "vs_oracle_delta",
            )
            if c in sm.columns
        ]
        lines.append(sm[cols].to_string(index=False))
        if not realistic.get("wins", pd.DataFrame()).empty:
            lines += [
                "",
                "### Models chosen by realistic selector",
                "",
                realistic["wins"].to_string(index=False),
            ]
    path = out_dir / "REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_comparison(zone: int = 2, out_dir: Path | None = None) -> dict:
    out_dir = out_dir or ROOT / f"results/comparison/zone_{zone}/all_models"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    fig_dir.mkdir(exist_ok=True)
    tab_dir.mkdir(exist_ok=True)

    df = load_all(zone)
    models = available_models(df)
    print("Available:", models)

    global_df = fair_global_table(df)
    best = best_config_per_model(global_df)
    pers = df[df["model"].str.startswith("persistence")]
    skill = skill_vs_persistence(global_df, pers)
    wins = wins_by_station(df)
    station_best = per_station_best(df)
    oracle = oracle_routing_analysis(df, PRIMARY)
    realistic = realistic_routing_analysis(zone, df)

    # Save tables
    df.to_csv(tab_dir / "metrics_long.csv", index=False)
    global_df.to_csv(tab_dir / "global_mean_by_config.csv", index=False)
    best.to_csv(tab_dir / "best_config_per_model.csv", index=False)
    if not skill.empty:
        skill.to_csv(tab_dir / "skill_vs_persistence.csv", index=False)
    if not wins.empty:
        wins.to_csv(tab_dir / "wins_by_config.csv", index=False)
    if not station_best.empty:
        station_best.to_csv(tab_dir / "best_per_station.csv", index=False)
    if not oracle["summary"].empty:
        oracle["summary"].to_csv(tab_dir / "oracle_vs_single_model.csv", index=False)
        oracle["oracle"].to_csv(tab_dir / "oracle_per_station.csv", index=False)
        oracle["wins"].to_csv(tab_dir / "oracle_wins_by_model.csv", index=False)
    if not realistic["summary"].empty:
        realistic["summary"].to_csv(tab_dir / "realistic_vs_single_model.csv", index=False)
        realistic["selection"].to_csv(tab_dir / "realistic_per_station.csv", index=False)
        realistic["wins"].to_csv(tab_dir / "realistic_wins_by_model.csv", index=False)

    # Paper-style overall table (best per model + baselines)
    overall_rows = []
    for _, r in best.iterrows():
        overall_rows.append(r.to_dict())
    if not pers.empty:
        for m in ("persistence_last", "persistence_ma24"):
            block = pers[pers["model"] == m]
            if block.empty:
                continue
            overall_rows.append(
                {
                    "feature_config": "baseline",
                    "description": "baseline",
                    "model": m,
                    "model_label": MODEL_LABELS[m],
                    "n_stations": int(block["station_key"].nunique()),
                    **{met: float(block[met].mean()) for met in METRICS},
                }
            )
    overall = pd.DataFrame(overall_rows)
    overall.to_csv(tab_dir / "table1_overall.csv", index=False)

    # Plots (no in-figure titles; captions live in the manuscript)
    plot_grouped_bars(
        global_df,
        PRIMARY,
        fig_dir / "fig_ma24_rmse_by_config.png",
        "",
    )
    plot_grouped_bars(
        global_df,
        "ma24_mae",
        fig_dir / "fig_ma24_mae_by_config.png",
        "",
    )
    plot_grouped_bars(
        global_df,
        "hourly_rmse",
        fig_dir / "fig_hourly_rmse_by_config.png",
        "",
    )
    plot_grouped_bars(
        global_df,
        "hourly_mae",
        fig_dir / "fig_hourly_mae_by_config.png",
        "",
    )
    if not skill.empty:
        plot_skill_bars(skill, fig_dir / "fig_skill_vs_persistence.png")
        pers_mean = float(skill["baseline_rmse"].iloc[0])
        plot_improvement_over_baseline(
            global_df, pers_mean, PRIMARY, fig_dir / "fig_improvement_pct.png"
        )
    plot_station_heatmap(df, PRIMARY, fig_dir / "fig_heatmap_stations.png", config="best")

    # Pred vs reality for a representative station (best F* per model)
    example_station = "BASAURI"
    if not station_best.empty:
        # Prefer a mid-tier station from station_best for readability
        example_station = str(station_best.iloc[len(station_best) // 2]["station"])
    pred_paths = plot_pred_vs_reality_all_models(
        zone, example_station, best, fig_dir / "pred_vs_real"
    )
    print(f"Pred-vs-real station={example_station} figures={len(pred_paths)}")

    # Lead-time curves (MAE/RMSE @ H+1…H+24) for best F* per model
    try:
        from src.experiments.lead_time_metrics import run_lead_time_analysis

        lead = run_lead_time_analysis(zone=zone, out_dir=out_dir, df=df, best=best)
    except Exception as exc:  # noqa: BLE001 — keep comparison usable if arrays missing
        print(f"Lead-time analysis skipped: {exc}")
        lead = {}

    best_station_paths = plot_best_model_per_station(
        zone, station_best, fig_dir / "pred_vs_real" / "best_per_station"
    )
    print(f"Best-model-per-station figures={len(best_station_paths)}")

    if not oracle["summary"].empty:
        plot_oracle_vs_single(
            oracle["summary"], PRIMARY, fig_dir / "fig_oracle_vs_single_model.png"
        )
        print(
            f"Oracle mean {PRIMARY}={oracle['oracle_mean']:.4f} "
            f"on {oracle['n_stations']} stations"
        )

    if not realistic["summary"].empty:
        plot_realistic_vs_single(
            realistic["summary"], fig_dir / "fig_realistic_vs_single_model.png"
        )
        print(
            f"Realistic selector eval RMSE={realistic['realistic_mean']:.4f} "
            f"| best single={realistic['best_single_mean']:.4f} "
            f"| oracle holdout={realistic['oracle_mean']:.4f} "
            f"| agree={realistic['agreement_pct']:.1f}%"
        )

    write_markdown_report(
        out_dir,
        global_df,
        best,
        skill,
        wins,
        station_best,
        oracle=oracle,
        realistic=realistic,
    )

    verdict = {}
    if not best.empty:
        top = best.iloc[0]
        verdict = {
            "best_model": top["model"],
            "best_config": top["feature_config"],
            "ma24_rmse": float(top[PRIMARY]),
            "models_available": models,
        }
    (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print("Wrote", out_dir)
    print("Verdict:", verdict)
    return {
        "df": df,
        "global": global_df,
        "best": best,
        "skill": skill,
        "wins": wins,
        "station_best": station_best,
        "overall": overall,
        "out_dir": out_dir,
        "verdict": verdict,
        "pred_vs_real_station": example_station,
        "pred_vs_real_paths": [str(p) for p in pred_paths],
        "oracle": oracle,
        "realistic": realistic,
        "lead_time": lead,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--zone", type=int, default=2)
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir) if args.out_dir else None
    run_comparison(args.zone, out)


if __name__ == "__main__":
    main()
