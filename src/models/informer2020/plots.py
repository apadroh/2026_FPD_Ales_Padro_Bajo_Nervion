"""
Diagnostic plots for Informer2020 runs (saved next to results.json).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curves(
    history: list[dict[str, Any]],
    out_path: Path,
    *,
    title: str = "",
) -> Path:
    """Train / val / test loss vs epoch."""
    epochs = [h["epoch"] for h in history]
    train = [h["train_loss"] for h in history]
    val = [h["val_loss"] for h in history]
    test = [h.get("test_loss") for h in history]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, train, "o-", label="train", color="#1f77b4")
    ax.plot(epochs, val, "s-", label="val", color="#ff7f0e")
    if all(v is not None for v in test):
        ax.plot(epochs, test, "^-", label="test", color="#2ca02c")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title(title or "Train / val / test loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_pred_vs_reality(
    preds: np.ndarray,
    trues: np.ndarray,
    out_path: Path,
    *,
    title: str = "",
    ylabel: str = "PM10",
    max_points: int = 600,
) -> Path:
    """
    Prediction vs ground truth on test (original units).

    Panel 1: time series
    Panel 2: error (pred - truth)
    Panel 3: scatter pred vs truth (+ 1:1 line)
    """
    y_hat = np.asarray(preds, dtype=float).reshape(-1)
    y = np.asarray(trues, dtype=float).reshape(-1)
    n = len(y)
    if n == 0:
        raise ValueError("Empty pred/true arrays")

    if n > max_points:
        idx = np.linspace(0, n - 1, max_points, dtype=int)
        y_hat_s, y_s = y_hat[idx], y[idx]
        x = idx
        subsampled = True
    else:
        y_hat_s, y_s = y_hat, y
        x = np.arange(n)
        subsampled = False

    rmse = float(np.sqrt(np.mean((y_hat - y) ** 2)))
    mae = float(np.mean(np.abs(y_hat - y)))

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 9),
        gridspec_kw={"height_ratios": [1.2, 0.7, 1.0]},
    )

    axes[0].plot(x, y_s, label="ground truth", color="#222222", linewidth=1.3)
    axes[0].plot(
        x,
        y_hat_s,
        label="prediction",
        color="#d62728",
        alpha=0.85,
        linewidth=1.0,
    )
    axes[0].set_ylabel(ylabel)
    axes[0].set_title(
        (title or "Prediction vs ground truth (test)")
        + f"\nMAE={mae:.2f}  RMSE={rmse:.2f}"
    )
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    err = y_hat_s - y_s
    axes[1].fill_between(x, err, 0, color="#9467bd", alpha=0.35)
    axes[1].plot(x, err, color="#9467bd", linewidth=0.8)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Error")
    axes[1].set_xlabel(
        "Test window (subsampled)" if subsampled else "Test window"
    )
    axes[1].grid(True, alpha=0.3)

    lo = float(min(y.min(), y_hat.min()))
    hi = float(max(y.max(), y_hat.max()))
    axes[2].scatter(y, y_hat, s=8, alpha=0.25, color="#1f77b4", edgecolors="none")
    axes[2].plot([lo, hi], [lo, hi], "k--", linewidth=1.0, label="ideal (y = x)")
    axes[2].set_xlabel(f"Ground truth ({ylabel})")
    axes[2].set_ylabel(f"Prediction ({ylabel})")
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].legend(loc="upper left")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_test_ma24(
    preds_ma24: np.ndarray,
    trues_ma24: np.ndarray,
    out_path: Path,
    *,
    title: str = "",
    max_points: int = 500,
    ylabel: str = "PM10 MA24",
) -> Path:
    """Compat wrapper → prediction vs ground truth."""
    return plot_pred_vs_reality(
        preds_ma24,
        trues_ma24,
        out_path,
        title=title or "Prediction vs ground truth (test MA24)",
        ylabel=ylabel,
        max_points=max_points,
    )


def plot_reality_with_test_pred(
    series: np.ndarray,
    borders: dict[str, list[int] | tuple[int, int]],
    preds_ma24: np.ndarray,
    out_path: Path,
    *,
    title: str = "",
    ylabel: str = "PM10",
) -> Path:
    """
    Full target series (train/val/test) and, on the test span,
    MA24 predictions aligned approximately at the end of each window.
    """
    y = np.asarray(series, dtype=float).reshape(-1)
    pred = np.asarray(preds_ma24, dtype=float).reshape(-1)
    fig, ax = plt.subplots(figsize=(12, 4.5))

    colors = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"}
    for name, color in colors.items():
        if name not in borders:
            continue
        a, b = borders[name]
        ax.axvspan(a, b, color=color, alpha=0.15, label=name)

    ax.plot(y, color="#222222", linewidth=0.7, label="ground truth")

    if "test" in borders and len(pred) > 0:
        t0, t1 = borders["test"]
        # Place each MA24 pred near the end of its forecast window along the test span
        span = max(t1 - t0, 1)
        # Leave room at the start of test for seq_len context; distribute preds over usable span
        usable = max(span - 48, len(pred))
        xs = t0 + 48 + np.linspace(0, usable - 1, num=len(pred))
        xs = np.clip(xs, t0, t1 - 1)
        ax.plot(
            xs,
            pred,
            color="#d62728",
            linewidth=1.0,
            alpha=0.9,
            label="prediction (test MA24)",
        )

    ax.set_xlabel("Time index")
    ax.set_ylabel(ylabel)
    ax.set_title(title or "Ground truth (train/val/test) + test prediction")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_target_splits(
    series: np.ndarray,
    borders: dict[str, list[int] | tuple[int, int]],
    out_path: Path,
    *,
    title: str = "",
    ylabel: str = "target",
) -> Path:
    """Full target series with train / val / test spans highlighted."""
    y = np.asarray(series).reshape(-1)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(y, color="#555555", linewidth=0.6, label="series")

    colors = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"}
    for name, color in colors.items():
        if name not in borders:
            continue
        a, b = borders[name]
        ax.axvspan(a, b, color=color, alpha=0.18, label=name)

    ax.set_xlabel("Time index")
    ax.set_ylabel(ylabel)
    ax.set_title(title or "Train / val / test split")
    # Deduplicate legend labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def save_run_plots(
    out_dir: Path,
    *,
    history: list[dict[str, Any]],
    preds_ma24: np.ndarray | None = None,
    trues_ma24: np.ndarray | None = None,
    target_series: np.ndarray | None = None,
    borders: dict[str, list[int] | tuple[int, int]] | None = None,
    station: str = "",
    feature_config: str = "",
    contaminant: str = "PM10",
) -> dict[str, str]:
    """Write standard PNGs into out_dir; return relative artifact paths."""
    tag = f"{station} · {feature_config}".strip(" ·")
    artifacts: dict[str, str] = {}

    if history:
        p = out_dir / "plot_loss_curves.png"
        plot_loss_curves(history, p, title=f"Loss curves — {tag}")
        artifacts["plot_loss_curves"] = p.as_posix()

    if preds_ma24 is not None and trues_ma24 is not None:
        p = out_dir / "plot_pred_vs_reality.png"
        plot_pred_vs_reality(
            preds_ma24,
            trues_ma24,
            p,
            title=f"Prediction vs ground truth (test MA24) — {tag}",
            ylabel=f"{contaminant} MA24",
        )
        artifacts["plot_pred_vs_reality"] = p.as_posix()
        # Keep legacy filename for older notebooks/docs
        p_legacy = out_dir / "plot_test_ma24.png"
        plot_test_ma24(
            preds_ma24,
            trues_ma24,
            p_legacy,
            title=f"Prediction vs ground truth (test MA24) — {tag}",
            ylabel=f"{contaminant} MA24",
        )
        artifacts["plot_test_ma24"] = p_legacy.as_posix()

    if target_series is not None and borders is not None:
        p = out_dir / "plot_train_val_test_split.png"
        plot_target_splits(
            target_series,
            borders,
            p,
            title=f"Split — {tag}",
            ylabel=contaminant,
        )
        artifacts["plot_train_val_test_split"] = p.as_posix()

        if preds_ma24 is not None:
            p2 = out_dir / "plot_reality_and_test_pred.png"
            plot_reality_with_test_pred(
                target_series,
                borders,
                preds_ma24,
                p2,
                title=f"Ground truth + test prediction — {tag}",
                ylabel=contaminant,
            )
            artifacts["plot_reality_and_test_pred"] = p2.as_posix()

    return artifacts
