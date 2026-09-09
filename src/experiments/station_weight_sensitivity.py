"""
Sensitivity of zone-level MA24 RMSE to how stations are aggregated.

Schemes (weights sum to 1):
  equal              — 1/N (current protocol)
  difficulty         — w ∝ persistence MA24 RMSE (hard stations weigh more)
  inv_difficulty     — w ∝ 1 / persistence MA24 RMSE (easy stations weigh more)
  tertile_equal      — mean within difficulty tertiles, then mean of 3 tertiles

Usage:
  python src/experiments/station_weight_sensitivity.py --zone 2
"""

from __future__ import annotations

import argparse
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
)

LEARNED = ("airformer", "xgboost", "informer", "gat_informer")
PERS = "persistence_ma24"


def load_long(zone: int) -> pd.DataFrame:
    path = (
        ROOT
        / "results"
        / "comparison"
        / f"zone_{zone}"
        / "all_models"
        / "tables"
        / "metrics_long.csv"
    )
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run compare_all_models.py first.")
    df = pd.read_csv(path)
    df = df[df["status"] == "ok"].copy()
    # Collapse encoding duplicates
    df = (
        df.sort_values(PRIMARY)
        .drop_duplicates(subset=["model", "feature_config", "station_key"], keep="first")
        .reset_index(drop=True)
    )
    return df


def station_errors_best_f(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Wide table station_key × model (best F each) + persistence series."""
    learned = df[df["model"].isin(LEARNED)].copy()
    g_rows = []
    for (cfg, m), g in learned.groupby(["feature_config", "model"]):
        g_rows.append(
            {
                "feature_config": cfg,
                "model": m,
                "model_label": MODEL_LABELS.get(m, m),
                "n_stations": g["station_key"].nunique(),
                PRIMARY: float(g[PRIMARY].mean()),
            }
        )
    global_df = pd.DataFrame(g_rows)
    best = best_config_per_model(global_df, PRIMARY)

    # Common stations across all learned + persistence
    common = None
    for m in LEARNED:
        keys = set(learned.loc[learned["model"] == m, "station_key"])
        common = keys if common is None else common & keys
    pers = df[df["model"] == PERS]
    common &= set(pers["station_key"])
    common = sorted(common)
    if len(common) < 3:
        raise SystemExit(f"Too few common stations: {len(common)}")

    pers_s = (
        pers.sort_values(PRIMARY)
        .drop_duplicates("station_key", keep="first")
        .set_index("station_key")[PRIMARY]
        .reindex(common)
        .astype(float)
    )

    wide = {}
    meta = []
    for _, row in best.iterrows():
        m, cfg = row["model"], row["feature_config"]
        if m not in LEARNED:
            continue
        sub = learned[
            (learned["model"] == m)
            & (learned["feature_config"] == cfg)
            & (learned["station_key"].isin(common))
        ]
        s = (
            sub.sort_values(PRIMARY)
            .drop_duplicates("station_key", keep="first")
            .set_index("station_key")[PRIMARY]
            .reindex(common)
            .astype(float)
        )
        wide[m] = s
        meta.append({"model": m, "feature_config": cfg, "model_label": MODEL_LABELS.get(m, m)})

    return pd.DataFrame(wide), pers_s, pd.DataFrame(meta)


def weights_equal(n: int) -> np.ndarray:
    return np.ones(n) / n


def weights_difficulty(pers: np.ndarray) -> np.ndarray:
    w = np.asarray(pers, dtype=float)
    w = np.clip(w, 1e-6, None)
    return w / w.sum()


def weights_inv_difficulty(pers: np.ndarray) -> np.ndarray:
    w = 1.0 / np.clip(np.asarray(pers, dtype=float), 1e-6, None)
    return w / w.sum()


def aggregate_tertile_equal(err: np.ndarray, pers: np.ndarray) -> float:
    """Mean within each persistence tertile, then mean of tertile means."""
    q = np.quantile(pers, [1 / 3, 2 / 3])
    bins = np.digitize(pers, q, right=True)  # 0,1,2
    means = []
    for b in (0, 1, 2):
        mask = bins == b
        if mask.any():
            means.append(float(err[mask].mean()))
    return float(np.mean(means)) if means else float("nan")


def weighted_mean(err: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * err))


def skill(err: float, pers: float) -> float:
    if not np.isfinite(pers) or pers <= 0:
        return float("nan")
    return 1.0 - err / pers


def run(zone: int) -> Path:
    df = load_long(zone)
    wide, pers_s, meta = station_errors_best_f(df)
    stations = list(pers_s.index)
    pers = pers_s.to_numpy()
    n = len(stations)

    schemes = {
        "equal": weights_equal(n),
        "difficulty": weights_difficulty(pers),
        "inv_difficulty": weights_inv_difficulty(pers),
    }

    rows = []
    for _, mrow in meta.iterrows():
        m = mrow["model"]
        err = wide[m].to_numpy()
        for scheme, w in schemes.items():
            e = weighted_mean(err, w)
            p = weighted_mean(pers, w)
            rows.append(
                {
                    "scheme": scheme,
                    "model": m,
                    "model_label": mrow["model_label"],
                    "feature_config": mrow["feature_config"],
                    "n_stations": n,
                    "ma24_rmse": e,
                    "pers_ma24_rmse": p,
                    "skill": skill(e, p),
                    "improvement_pct": 100.0 * skill(e, p),
                }
            )
        # tertile scheme (not a simple weight vector)
        e_t = aggregate_tertile_equal(err, pers)
        p_t = aggregate_tertile_equal(pers, pers)
        rows.append(
            {
                "scheme": "tertile_equal",
                "model": m,
                "model_label": mrow["model_label"],
                "feature_config": mrow["feature_config"],
                "n_stations": n,
                "ma24_rmse": e_t,
                "pers_ma24_rmse": p_t,
                "skill": skill(e_t, p_t),
                "improvement_pct": 100.0 * skill(e_t, p_t),
            }
        )

    out = pd.DataFrame(rows)
    # Rank within scheme
    out["rank"] = out.groupby("scheme")["ma24_rmse"].rank(method="min").astype(int)

    out_dir = ROOT / "results" / "comparison" / f"zone_{zone}" / "all_models"
    tab_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    tab_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tab_dir / "station_weight_sensitivity.csv"
    out.to_csv(csv_path, index=False)

    # Weights for transparency
    w_df = pd.DataFrame(
        {
            "station_key": stations,
            "pers_ma24_rmse": pers,
            "w_equal": schemes["equal"],
            "w_difficulty": schemes["difficulty"],
            "w_inv_difficulty": schemes["inv_difficulty"],
        }
    )
    w_path = tab_dir / "station_weight_sensitivity_weights.csv"
    w_df.to_csv(w_path, index=False)

    # Figure: grouped bars MA24 RMSE by scheme
    scheme_order = ["equal", "difficulty", "inv_difficulty", "tertile_equal"]
    scheme_labels = {
        "equal": "Equal (1/N)",
        "difficulty": "∝ pers. RMSE",
        "inv_difficulty": "∝ 1/pers.",
        "tertile_equal": "Tertile equal",
    }
    model_order = [m for m in LEARNED if m in out["model"].unique()]
    colors = {
        "airformer": "#1d3557",
        "xgboost": "#e76f51",
        "informer": "#2a9d8f",
        "gat_informer": "#e9c46a",
    }

    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(scheme_order))
    width = 0.18
    for i, m in enumerate(model_order):
        vals = []
        for sch in scheme_order:
            hit = out[(out["scheme"] == sch) & (out["model"] == m)]
            vals.append(float(hit["ma24_rmse"].iloc[0]) if len(hit) else np.nan)
        ax.bar(
            x + (i - 1.5) * width,
            vals,
            width,
            label=MODEL_LABELS.get(m, m),
            color=colors.get(m, "gray"),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([scheme_labels[s] for s in scheme_order])
    ax.set_ylabel("Aggregated MA24 RMSE")
    ax.set_title("Sensitivity to station aggregation weights (best F per model)")
    ax.legend(frameon=False, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig_path = fig_dir / "fig_station_weight_sensitivity.png"
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)

    # Rank stability summary
    pivot = out.pivot_table(index="model_label", columns="scheme", values="rank")
    rank_path = tab_dir / "station_weight_sensitivity_ranks.csv"
    pivot.to_csv(rank_path)

    print("Wrote", csv_path)
    print("Wrote", w_path)
    print("Wrote", rank_path)
    print("Wrote", fig_path)
    print("\nRanks (1=best) by scheme:")
    print(pivot[scheme_order].to_string())
    print("\nMA24 RMSE:")
    print(
        out.pivot_table(index="model_label", columns="scheme", values="ma24_rmse")[
            scheme_order
        ]
        .round(3)
        .to_string()
    )
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--zone", type=int, default=2)
    args = p.parse_args()
    run(args.zone)


if __name__ == "__main__":
    main()
