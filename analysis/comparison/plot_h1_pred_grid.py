"""Compact all-station pred-vs-real grid: Real vs best Transformer only."""
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
    load_ma24_pred_true,
    load_all,
)

OFFICIAL = ["F1", "F2", "F3", "F4", "F5"]
TRANSFORMERS = ["informer", "airformer", "gat_informer"]
COLORS = {
    "true": "#333333",
    "informer": "#1f77b4",
    "airformer": "#ff7f0e",
    "gat_informer": "#2ca02c",
}
LABELS = {
    "informer": "Informer",
    "airformer": "AirFormer",
    "gat_informer": "GAT-Informer",
}
SHORT = {"informer": "Inf", "airformer": "AF", "gat_informer": "GAT"}


def _subsample(y: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float).reshape(-1)
    if len(y) <= n:
        return np.arange(len(y)), y
    idx = np.linspace(0, len(y) - 1, n, dtype=int)
    return idx, y[idx]


def best_transformer(df: pd.DataFrame, station_key_val: str) -> tuple[str, str]:
    block = df[
        (df["station_key"] == station_key_val)
        & (df["model"].isin(TRANSFORMERS))
        & (df["feature_config"].isin(OFFICIAL))
    ]
    row = block.loc[block["ma24_rmse"].idxmin()]
    return str(row["model"]), str(row["feature_config"])


def main(zone: int = 2) -> Path:
    df = load_all(zone)
    stations = (
        df[~df["model"].str.startswith("persistence")]
        .drop_duplicates("station_key")
        .sort_values("station")
    )
    names = list(zip(stations["station"], stations["station_key"]))
    assert len(names) == 18, f"expected 18 stations, got {len(names)}"

    nrows, ncols = 3, 6
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(11.5, 6.0),
        sharex=True,
        sharey=False,
        constrained_layout=True,
    )
    axes_flat = axes.ravel()
    max_points = 140

    # Fixed legend proxies (one entry per architecture that appears)
    used_models: set[str] = set()
    legend_handles: list = []

    for i, (station, sk) in enumerate(names):
        ax = axes_flat[i]
        win_m, win_f = best_transformer(df, sk)
        loaded = load_ma24_pred_true(win_m, zone, win_f, station)
        if loaded is None:
            ax.set_visible(False)
            continue
        pred, true = loaded
        used_models.add(win_m)

        x, y_true = _subsample(true, max_points)
        _, y_hat = _subsample(pred, max_points)
        n = min(len(y_hat), len(x))
        ax.plot(x[:n], y_true[:n], color=COLORS["true"], lw=0.75, label="Real")
        ax.plot(
            x[:n],
            y_hat[:n],
            color=COLORS[win_m],
            lw=0.9,
            alpha=0.9,
            label=LABELS[win_m],
        )

        short = (
            str(station)
            .replace(" (BBIZI2)", "")
            .replace(" (Puerto)", "")
            .replace(" (Monte)", "")
        )
        ax.set_title(f"{short[:11]} · {SHORT[win_m]} {win_f}", fontsize=6.5, pad=2)
        ax.tick_params(labelsize=5, length=2)
        ax.grid(True, alpha=0.2, lw=0.4)

    for j in range(len(names), nrows * ncols):
        axes_flat[j].set_visible(False)

    # Build a clean figure-level legend
    from matplotlib.lines import Line2D

    legend_handles = [Line2D([0], [0], color=COLORS["true"], lw=1.2, label="Real")]
    for m in ("airformer", "gat_informer", "informer"):
        if m in used_models:
            legend_handles.append(
                Line2D([0], [0], color=COLORS[m], lw=1.2, label=LABELS[m])
            )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        fontsize=8,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )

    out = (
        ROOT
        / f"results/comparison/zone_{zone}/all_models/figures"
        / "fig_pred_vs_real_h1_all_stations.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)

    print_dir = ROOT / "docs" / "figures_print"
    print_dir.mkdir(parents=True, exist_ok=True)
    dest = print_dir / out.name
    dest.write_bytes(out.read_bytes())
    print(f"Wrote {out}")
    print(f"Wrote {dest}")
    print("Winners:", {SHORT[m]: sum(1 for _, sk in names if best_transformer(df, sk)[0] == m) for m in TRANSFORMERS})
    return out


if __name__ == "__main__":
    main()
