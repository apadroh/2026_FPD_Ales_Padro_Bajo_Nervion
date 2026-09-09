"""
H4 skill bars: best official F1--F5 per model, plus peak extras F3IL / F5I.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "results" / "comparison" / "zone_2" / "all_models" / "tables" / "skill_vs_persistence.csv"
FIG_DIR = ROOT / "results" / "comparison" / "zone_2" / "all_models" / "figures"
PRINT_DIR = ROOT / "docs" / "figures_print"
OUT_NAME = "fig_skill_vs_persistence.png"

# Equal station-mean MA24 from REPORT / peak tables; same persistence base as CSV
PERS = 7.2035369932072415
EXTRAS = [
    # (label, ma24_rmse, color, is_extra)
    ("AirFormer F3IL", 5.4199, "#e76f51", True),
    ("XGBoost F5I", 5.4216, "#f4a261", True),
]


def main() -> Path:
    skill = pd.read_csv(TAB)
    # Best official F per model (F1--F5 ladder only)
    best = skill.loc[skill.groupby("model")["skill"].idxmax()].copy()
    best = best.sort_values("improvement_pct")

    labels: list[str] = []
    vals: list[float] = []
    colors: list[str] = []

    official_colors = {
        "informer": "#457b9d",
        "gat_informer": "#1d3557",
        "airformer": "#2a9d8f",
        "xgboost": "#264653",
    }
    for _, row in best.iterrows():
        m = row["model"]
        cfg = row["feature_config"]
        lab = f"{row['model_label']} {cfg}"
        labels.append(lab)
        vals.append(float(row["improvement_pct"]))
        colors.append(official_colors.get(m, "#2a6f97"))

    for lab, rmse, col, _ in EXTRAS:
        labels.append(lab)
        vals.append(100.0 * (1.0 - rmse / PERS))
        colors.append(col)

    fig, ax = plt.subplots(figsize=(10, 5.0))
    y = range(len(labels))
    ax.barh(list(y), vals, color=colors, height=0.7)
    ax.axvline(0, color="gray", lw=0.8)
    for yi, v in zip(y, vals):
        ax.text(v + 0.35, yi, f"{v:.1f}%", va="center", fontsize=8)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Skill vs persistence MA24 (%)")
    ax.set_xlim(0, max(vals) * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Legend note
    ax.text(
        0.98,
        0.02,
        "Teal/navy = best official F1–F5 · Orange = peak extras (H3)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#333",
    )
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PRINT_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / OUT_NAME
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    dest = PRINT_DIR / OUT_NAME
    dest.write_bytes(out.read_bytes())
    print(f"saved {out}")
    for lab, v in zip(labels, vals):
        print(f"  {lab}: {v:.2f}%")
    return out


if __name__ == "__main__":
    main()
