"""
H2 figure matching the mental model:

  GAT       = fixed % that station i takes from each station j (PM10 Adj)
  AirFormer = same stations as grouped horizontal bars at 08:00, 12:00, 18:00
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ATT_DIR = ROOT / "results" / "comparison" / "zone_2" / "spatial_attention"
FIG_DIR = ROOT / "results" / "comparison" / "zone_2" / "all_models" / "figures"
PRINT_DIR = ROOT / "docs" / "figures_print"

FOCUS = ["ABANTO", "SAN JULIAN", "SANTURCE"]
HOURS = [8, 12, 18]
HOUR_COLORS = ["#2a9d8f", "#e9c46a", "#e76f51"]
TOP_K = 6


def short(name: str) -> str:
    return (
        str(name)
        .replace(" (BBIZI2)", "")
        .replace(" (Puerto)", "")
        .replace(" (Monte)", "")
        .replace("Mª ", "M.")
        .replace("M� ", "M.")
    )


def main() -> Path:
    meta = json.loads((ATT_DIR / "meta.json").read_text(encoding="utf-8"))
    stations: list[str] = meta["stations"]
    A_gat = np.load(ATT_DIR / "A_gat.npy")
    A_by_hour = np.load(ATT_DIR / "A_by_hour.npy")  # (24, N, N)

    fig = plt.figure(figsize=(12.0, 7.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.1])
    n_h = len(HOURS)

    for col, st in enumerate(FOCUS):
        i = stations.index(st)
        order = list(np.argsort(-A_gat[i])[:TOP_K])
        names = [short(stations[j]) for j in order]
        y = np.arange(len(names), dtype=float)

        # ----- Top: GAT fixed % -----
        ax = fig.add_subplot(gs[0, col])
        gat_vals = 100.0 * A_gat[i, order]
        colors = ["#e76f51" if stations[j] == st else "#457b9d" for j in order]
        ax.barh(y[::-1], gat_vals[::-1], color=colors[::-1], height=0.72)
        ax.set_yticks(y[::-1])
        ax.set_yticklabels(names[::-1], fontsize=7.5)
        ax.set_xlabel("% of mix (fixed forever)", fontsize=8)
        ax.set_xlim(0, max(gat_vals.max() * 1.25, 1))
        for yy, v in zip(y[::-1], gat_vals[::-1]):
            ax.text(v + 0.35, yy, f"{v:.0f}%", va="center", fontsize=6.5)
        ax.set_title(f"GAT fixed · mix into {short(st)}", fontsize=9)
        ax.grid(axis="x", alpha=0.25)

        # ----- Bottom: AirFormer same stations @ 08 / 12 / 18 -----
        ax = fig.add_subplot(gs[1, col])
        vals = [100.0 * A_by_hour[h, i, order] for h in HOURS]  # list of (K,)
        h_bar = 0.22
        offsets = np.linspace(-(n_h - 1) / 2, (n_h - 1) / 2, n_h) * h_bar
        y_plot = y[::-1]
        vmax = max(float(v.max()) for v in vals)

        for t, (hour, color, off) in enumerate(zip(HOURS, HOUR_COLORS, offsets)):
            ax.barh(
                y_plot + off,
                vals[t][::-1],
                height=h_bar * 0.92,
                color=color,
                label=f"{hour:02d}:00",
            )

        ax.set_yticks(y_plot)
        ax.set_yticklabels(names[::-1], fontsize=7.5)
        ax.set_xlabel("% of mix (changes with data)", fontsize=8)
        ax.set_xlim(0, vmax * 1.28)
        # label: 8→12→18
        for yy, *row in zip(y_plot, *[v[::-1] for v in vals]):
            txt = "→".join(f"{v:.0f}" for v in row)
            ax.text(max(row) + 0.5, yy, txt, va="center", fontsize=5.8)
        ax.set_title("AirFormer dynamic · same stations @ 08 / 12 / 18 h", fontsize=9)
        ax.grid(axis="x", alpha=0.25)
        if col == 0:
            ax.legend(fontsize=7.5, frameon=True, loc="lower right", title="hour of day")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PRINT_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig_af_vs_gat_spatial_adjacency.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    dest = PRINT_DIR / out.name
    dest.write_bytes(out.read_bytes())
    print(f"saved {out}")
    print(f"copied {dest}")
    return out


if __name__ == "__main__":
    main()
