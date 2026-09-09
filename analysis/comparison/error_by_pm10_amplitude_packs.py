"""
Hourly RMSE/MAE by true-PM10 amplitude for key model x feature packs.

Answers the supervisor question: does the global winner (XGBoost F5) still
win in decision-relevant high-PM10 bins?
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

from src.experiments.compare_all_models import MODEL_LABELS, load_all  # noqa: E402
from src.experiments.error_distribution_analysis import (  # noqa: E402
    PM10_BINS,
    PM10_LABELS,
    collect_errors,
    summarize_by_pm10,
)

# Official ladder bests + peak extras + matched F3IL
PAIRS = [
    ("informer", "F1"),
    ("gat_informer", "F2"),
    ("airformer", "F3"),
    ("airformer", "F5"),
    ("airformer", "F3IL"),
    ("xgboost", "F5"),
    ("xgboost", "F5I"),
    ("xgboost", "F3IL"),
]

FIG_DIR = ROOT / "results" / "comparison" / "zone_2" / "all_models" / "figures"
TAB_DIR = ROOT / "results" / "comparison" / "zone_2" / "all_models" / "tables"
PRINT_DIR = ROOT / "docs" / "figures_print"


def main() -> None:
    out_csv = TAB_DIR / "error_by_pm10_amplitude_packs.csv"
    if out_csv.exists():
        by_pm = pd.read_csv(out_csv)
        print("Reusing", out_csv)
        # still build wide for convenience
        focus = by_pm[by_pm["pm10_bin"].isin(["20-40", "40-80", "80+"])].copy()
        focus["label"] = focus["model"].map(MODEL_LABELS) + " " + focus["feature_config"]
        wide = focus.pivot_table(
            index="label", columns="pm10_bin", values="rmse", aggfunc="mean", observed=False
        )
        print(wide.round(2).to_string() if not wide.empty else "(no wide)")
    else:
        df = load_all(2)
        rows = []
        for model, cfg in PAIRS:
            sub = df[(df["model"] == model) & (df["feature_config"] == cfg)]
            if sub.empty:
                print(f"SKIP missing {model} {cfg}")
                continue
            rows.append({"model": model, "feature_config": cfg})
        best = pd.DataFrame(rows)
        print("Collecting hourly errors for", len(best), "packs…")
        err = collect_errors(df, best, zone=2)
        if err.empty:
            raise SystemExit("No errors collected")
        by_pm = summarize_by_pm10(err)
        TAB_DIR.mkdir(parents=True, exist_ok=True)
        by_pm.to_csv(out_csv, index=False)
        print("Wrote", out_csv)

    # Wide RMSE table for reporting
    focus = by_pm.copy()
    focus["label"] = focus["model"].map(MODEL_LABELS) + " " + focus["feature_config"]
    wide = focus.pivot_table(
        index="label", columns="pm10_bin", values="rmse", aggfunc="mean", observed=False
    )
    for col in PM10_LABELS:
        if col not in wide.columns:
            wide[col] = np.nan
    wide = wide.reindex(columns=PM10_LABELS)
    df = load_all(2)
    g = (
        df[df["model"].isin([m for m, _ in PAIRS])]
        .groupby(["model", "feature_config"], as_index=False)["ma24_rmse"]
        .mean()
    )
    g["label"] = g["model"].map(MODEL_LABELS) + " " + g["feature_config"]
    wide["ma24_global"] = np.nan
    for lab, v in zip(g["label"], g["ma24_rmse"]):
        if lab in wide.index:
            wide.loc[lab, "ma24_global"] = v
    wide = wide.sort_values("ma24_global")
    wide_path = TAB_DIR / "error_by_pm10_amplitude_packs_wide.csv"
    wide.to_csv(wide_path)
    print(wide.round(2).to_string())
    print("Wrote", wide_path)

    # Figure: all amplitude bins (low bins show Transformer advantage)
    bin_names = ["0-10", "10-20", "20-40", "40-80", "80+"]
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.2))
    axes_flat = list(axes.ravel())
    for ax in axes_flat[len(bin_names) :]:
        ax.axis("off")
    for ax, bin_name in zip(axes_flat, bin_names):
        short: list[str] = []
        vals: list[float] = []
        colors: list[str] = []
        for m, c in PAIRS:
            lab = f"{MODEL_LABELS.get(m, m)} {c}".replace("Informer2020", "Informer")
            short.append(lab)
            row = by_pm[
                (by_pm["model"] == m)
                & (by_pm["feature_config"] == c)
                & (by_pm["pm10_bin"] == bin_name)
            ]
            vals.append(float(row["rmse"].iloc[0]) if len(row) else float("nan"))
            colors.append("#e76f51" if c in {"F3IL", "F5I"} else "#457b9d")
        ax.barh(range(len(short))[::-1], vals[::-1], color=colors[::-1], height=0.7)
        ax.set_yticks(range(len(short))[::-1])
        ax.set_yticklabels(short[::-1], fontsize=7)
        ax.set_xlabel("Hourly RMSE (µg/m³)", fontsize=7.5)
        ax.set_title(f"True PM10 bin {bin_name}", fontsize=9)
        ax.grid(axis="x", alpha=0.25)
        finite = [v for v in vals if np.isfinite(v)]
        xmax = (max(finite) if finite else 1.0) * 1.18
        ax.set_xlim(0, xmax)
        for y, v in zip(range(len(short))[::-1], vals[::-1]):
            if np.isfinite(v):
                ax.text(v + xmax * 0.01, y, f"{v:.1f}", va="center", fontsize=6.5)
    fig.suptitle(
        "Error by target amplitude — official (blue) vs peak extras (orange)\n"
        "Low bins: AirFormer leads · High bins: XGBoost leads",
        fontsize=11,
    )
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PRINT_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig_error_by_pm10_amplitude_packs.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    dest = PRINT_DIR / out.name
    dest.write_bytes(out.read_bytes())
    print("Wrote", out)


if __name__ == "__main__":
    main()
