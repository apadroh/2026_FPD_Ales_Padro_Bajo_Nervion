"""
Compare Informer2020 feature configs (F1–F5) from batch_summary_*.csv.

Produces station-level and global tables for all metrics
(hourly_mae, hourly_rmse, ma24_mae, ma24_rmse).

Global tables use the **mean** (and also median) across stations.
The fair global mean uses only stations present in every ready config.

Usage:
  python src/experiments/compare_informer2020_configs.py --zone 2
  python src/experiments/compare_informer2020_configs.py --zone 2 --require-all

Outputs under results/informer2020/zone_{Z}/comparison/:
  tables/by_station_{metric}.csv
  tables/by_station_all_metrics.csv
  tables/global_mean.csv
  tables/global_median.csv
  TABLES.md
  REPORT.md, plots, snapshot JSON
"""

from __future__ import annotations

import argparse
import json
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

from src.data.feature_configs import ABLATION_LEVELS, LEVEL_DESCRIPTIONS

METRICS = ("hourly_mae", "hourly_rmse", "ma24_mae", "ma24_rmse")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare Informer2020 F* configs")
    p.add_argument("--zone", type=int, default=2)
    p.add_argument("--results-root", default="results/informer2020")
    p.add_argument(
        "--configs",
        nargs="*",
        default=None,
        choices=ABLATION_LEVELS,
        help="Subset of configs (default: all with usable batch_summary)",
    )
    p.add_argument(
        "--metric",
        default="ma24_rmse",
        choices=list(METRICS),
        help="Primary ranking metric (lower is better)",
    )
    p.add_argument(
        "--require-all",
        action="store_true",
        help="Fail if any of F1–F5 is missing or has zero ok stations",
    )
    p.add_argument(
        "--min-ok",
        type=int,
        default=1,
        help="Minimum ok stations for a config to enter the comparison",
    )
    p.add_argument(
        "--decimals",
        type=int,
        default=4,
        help="Decimals in markdown tables",
    )
    return p.parse_args()


def load_summaries(
    zone_dir: Path,
    configs: list[str] | None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    notes: dict[str, str] = {}
    frames: list[pd.DataFrame] = []
    wanted = configs or list(ABLATION_LEVELS)

    for cfg in wanted:
        path = zone_dir / f"batch_summary_{cfg}.csv"
        if not path.exists():
            notes[cfg] = "missing batch_summary"
            continue
        df = pd.read_csv(path)
        if df.empty:
            notes[cfg] = "empty summary"
            continue
        df["feature_config"] = cfg
        for m in METRICS:
            if m in df.columns:
                df[m] = pd.to_numeric(df[m], errors="coerce")
        n_ok = int((df["status"] == "ok").sum())
        n_fail = int((df["status"] == "failed").sum())
        if n_ok == 0:
            notes[cfg] = f"0 ok / {len(df)} rows ({n_fail} failed)"
            continue
        notes[cfg] = f"{n_ok} ok / {len(df)} rows"
        frames.append(df)

    if not frames:
        raise SystemExit(f"No usable batch summaries under {zone_dir}")
    return pd.concat(frames, ignore_index=True), notes


def ready_configs(ok: pd.DataFrame, min_ok: int) -> list[str]:
    counts = ok.groupby("feature_config").size()
    return [
        c
        for c in ABLATION_LEVELS
        if c in counts.index and int(counts[c]) >= min_ok
    ]


def pivot_metric(ok: pd.DataFrame, metric: str, configs: list[str]) -> pd.DataFrame:
    pivot = ok.pivot_table(
        index="station",
        columns="feature_config",
        values=metric,
        aggfunc="first",
    )
    cols = [c for c in configs if c in pivot.columns]
    return pivot.reindex(columns=cols).sort_index()


def common_stations(ok: pd.DataFrame, configs: list[str]) -> list[str]:
    sets = []
    for cfg in configs:
        s = set(ok.loc[ok["feature_config"] == cfg, "station"].astype(str))
        sets.append(s)
    if not sets:
        return []
    return sorted(set.intersection(*sets))


def by_station_all_metrics(
    ok: pd.DataFrame, configs: list[str], metrics: tuple[str, ...]
) -> pd.DataFrame:
    """Wide table: station | F1_hourly_mae | F1_hourly_rmse | ... | best_ma24_rmse."""
    pieces = []
    for metric in metrics:
        p = pivot_metric(ok, metric, configs)
        p = p.rename(columns={c: f"{c}_{metric}" for c in p.columns})
        pieces.append(p)
    wide = pd.concat(pieces, axis=1)
    # Best config on primary metric if columns exist
    primary_cols = [f"{c}_ma24_rmse" for c in configs if f"{c}_ma24_rmse" in wide.columns]
    if primary_cols:
        sub = wide[primary_cols]
        wide["best_ma24_rmse_config"] = (
            sub.idxmin(axis=1).str.replace("_ma24_rmse", "", regex=False)
        )
        wide["best_ma24_rmse"] = sub.min(axis=1)
    return wide.reset_index()


def global_agg(
    ok: pd.DataFrame,
    configs: list[str],
    metrics: tuple[str, ...],
    stations: list[str],
    how: str = "mean",
) -> pd.DataFrame:
    """One row per config; columns = metrics. Aggregation over `stations`."""
    sub = ok[ok["station"].isin(stations) & ok["feature_config"].isin(configs)]
    rows = []
    for cfg in configs:
        g = sub[sub["feature_config"] == cfg]
        row: dict[str, object] = {
            "feature_config": cfg,
            "description": LEVEL_DESCRIPTIONS.get(cfg, ""),
            "n_stations": int(g["station"].nunique()),
            "agg": how,
        }
        for metric in metrics:
            vals = pd.to_numeric(g[metric], errors="coerce").dropna()
            if vals.empty:
                row[metric] = np.nan
            elif how == "mean":
                row[metric] = float(vals.mean())
            else:
                row[metric] = float(vals.median())
        rows.append(row)
    return pd.DataFrame(rows)


def wins_table(ok: pd.DataFrame, metric: str, configs: list[str]) -> pd.DataFrame:
    pivot = pivot_metric(ok, metric, configs).dropna()
    records = []
    for station, row in pivot.iterrows():
        best = row.astype(float).idxmin()
        records.append(
            {
                "station": station,
                "best_config": best,
                "best_value": float(row[best]),
                **{c: float(row[c]) for c in pivot.columns},
            }
        )
    return pd.DataFrame(records)


def df_to_md(df: pd.DataFrame, decimals: int) -> str:
    show = df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(
                lambda x, d=decimals: "" if pd.isna(x) else f"{x:.{d}f}"
            )
        else:
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else str(x))

    headers = [str(c) for c in show.columns]
    rows = show.astype(str).values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def plot_mean_by_config(summary: pd.DataFrame, metric: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cfgs = summary["feature_config"].tolist()
    means = summary[metric].tolist()
    ns = summary["n_stations"].tolist()
    bars = ax.bar(cfgs, means, color="#4C78A8", edgecolor="none")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_xlabel("Feature config")
    ax.set_title(f"Global mean {metric} by config (lower is better)")
    for bar, n, m in zip(bars, ns, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{m:.3f}\n(n={n})",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_heatmap(pivot: pd.DataFrame, metric: str, out: Path) -> None:
    if pivot.empty:
        return
    data = pivot.to_numpy(dtype=float)
    fig, ax = plt.subplots(
        figsize=(1.4 * max(len(pivot.columns), 3) + 3, 0.35 * len(pivot) + 2)
    )
    im = ax.imshow(data, aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(list(pivot.columns))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index), fontsize=8)
    ax.set_title(f"{metric} by station × config (darker = better)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(metric)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def write_tables_md(
    out_dir: Path,
    *,
    zone: int,
    ready: list[str],
    common: list[str],
    global_mean: pd.DataFrame,
    global_median: pd.DataFrame,
    pivots: dict[str, pd.DataFrame],
    wins: pd.DataFrame,
    decimals: int,
) -> None:
    lines = [
        f"# Informer2020 comparison tables — zone {zone}",
        "",
        "Metrics: `hourly_mae`, `hourly_rmse`, `ma24_mae`, `ma24_rmse` "
        "(original units; **lower is better**).",
        "",
        f"Configs: {', '.join(ready)}.",
        f"Global = **mean** (and median) over the **{len(common)} stations "
        "common to all ready configs (fair comparison).",
        "",
        "## 1. Global — mean by configuration",
        "",
        df_to_md(global_mean, decimals),
        "",
        "## 2. Global — median by configuration",
        "",
        df_to_md(global_median, decimals),
        "",
    ]

    if not wins.empty:
        win_counts = wins["best_config"].value_counts()
        lines += ["## 3. Station wins (best = lowest `ma24_rmse`)", ""]
        for cfg in ready:
            lines.append(f"- **{cfg}**: {int(win_counts.get(cfg, 0))} stations")
        lines += ["", df_to_md(wins, decimals), ""]

    for metric, pivot in pivots.items():
        lines += [
            f"## Per station — `{metric}`",
            "",
            df_to_md(pivot.reset_index(), decimals),
            "",
        ]

    (out_dir / "TABLES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(
    out_dir: Path,
    *,
    zone: int,
    metric: str,
    notes: dict[str, str],
    global_mean: pd.DataFrame,
    wins: pd.DataFrame,
    ready: list[str],
    pending: list[str],
    n_common: int,
) -> None:
    lines = [
        f"# Informer2020 config comparison — zone {zone}",
        "",
        f"Primary ranking metric: **`{metric}`** (lower is better).",
        f"Global tables = **mean** over **{n_common}** common stations "
        "(see also `TABLES.md` and `tables/`).",
        "",
        "## Config readiness",
        "",
    ]
    for cfg in ABLATION_LEVELS:
        status = notes.get(cfg, "not scanned")
        mark = "ready" if cfg in ready else "pending"
        lines.append(
            f"- **{cfg}** ({LEVEL_DESCRIPTIONS.get(cfg, '')}): "
            f"`{mark}` — {status}"
        )

    lines += ["", "## Global mean (all metrics)", "", df_to_md(global_mean, 4), ""]

    if not wins.empty:
        win_counts = wins["best_config"].value_counts().to_dict()
        lines += [
            f"## Wins by station (`{metric}`, common stations)",
            "",
        ]
        for cfg in ready:
            lines.append(f"- {cfg}: **{win_counts.get(cfg, 0)}**")

    if pending:
        lines += [
            "",
            "## Pending",
            "",
            f"{', '.join(pending)} — re-run after pull.",
        ]
    else:
        lines += ["", "## Status", "", "Full F1–F5 ablation tables ready."]

    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    zone_dir = Path(args.results_root) / f"zone_{args.zone}"
    if not zone_dir.exists():
        raise SystemExit(f"Missing {zone_dir}")

    long, notes = load_summaries(zone_dir, args.configs)
    ok = long[long["status"] == "ok"].copy()
    ready = ready_configs(ok, args.min_ok)
    pending = [c for c in ABLATION_LEVELS if c not in ready]

    if args.require_all and pending:
        raise SystemExit(
            "require-all: still pending " + ", ".join(pending) + f" ({notes})"
        )
    if not ready:
        raise SystemExit("No ready configs")

    ok = ok[ok["feature_config"].isin(ready)]
    out_dir = zone_dir / "comparison"
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    long.to_csv(out_dir / "metrics_long.csv", index=False)

    common = common_stations(ok, ready)
    if not common:
        raise SystemExit("No common stations across ready configs")

    # Per-metric station pivots
    pivots: dict[str, pd.DataFrame] = {}
    for metric in METRICS:
        piv = pivot_metric(ok, metric, ready)
        # Fair subset first in TABLES; save full pivot too
        piv_common = piv.loc[piv.index.isin(common)]
        pivots[metric] = piv_common
        piv.to_csv(tables_dir / f"by_station_{metric}_all.csv")
        piv_common.to_csv(tables_dir / f"by_station_{metric}.csv")
        # Keep legacy name for primary metric
        if metric == args.metric:
            piv_common.to_csv(out_dir / f"pivot_{metric}.csv")

    wide = by_station_all_metrics(ok, ready, METRICS)
    wide_common = wide[wide["station"].isin(common)].copy()
    wide.to_csv(tables_dir / "by_station_all_metrics_all.csv", index=False)
    wide_common.to_csv(tables_dir / "by_station_all_metrics.csv", index=False)

    global_mean = global_agg(ok, ready, METRICS, common, how="mean")
    global_median = global_agg(ok, ready, METRICS, common, how="median")
    global_mean.to_csv(tables_dir / "global_mean.csv", index=False)
    global_median.to_csv(tables_dir / "global_median.csv", index=False)
    # Convenience copies at comparison root
    global_mean.to_csv(out_dir / "global_mean.csv", index=False)
    global_median.to_csv(out_dir / "global_median.csv", index=False)

    wins = wins_table(ok[ok["station"].isin(common)], args.metric, ready)
    if not wins.empty:
        wins.to_csv(out_dir / "wins_by_station.csv", index=False)
        wins.to_csv(tables_dir / "wins_by_station.csv", index=False)

    plot_mean_by_config(
        global_mean, args.metric, out_dir / f"plot_mean_{args.metric}.png"
    )
    plot_heatmap(
        pivots[args.metric],
        args.metric,
        out_dir / f"plot_heatmap_{args.metric}.png",
    )

    write_tables_md(
        out_dir,
        zone=args.zone,
        ready=ready,
        common=common,
        global_mean=global_mean,
        global_median=global_median,
        pivots=pivots,
        wins=wins,
        decimals=args.decimals,
    )
    write_report(
        out_dir,
        zone=args.zone,
        metric=args.metric,
        notes=notes,
        global_mean=global_mean,
        wins=wins,
        ready=ready,
        pending=pending,
        n_common=len(common),
    )

    payload = {
        "zone": args.zone,
        "metric": args.metric,
        "ready_configs": ready,
        "pending_configs": pending,
        "n_common_stations": len(common),
        "global_agg": "mean over common stations",
        "global_mean": global_mean.to_dict(orient="records"),
        "global_median": global_median.to_dict(orient="records"),
        "win_counts": (
            wins["best_config"].value_counts().to_dict() if not wins.empty else {}
        ),
    }
    (out_dir / "comparison_snapshot.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print(f"Ready configs : {', '.join(ready)}")
    print(f"Pending       : {', '.join(pending) or '(none)'}")
    print(f"Common stns   : {len(common)}")
    print(f"Wrote         : {out_dir}")
    print("\n=== GLOBAL MEAN ===")
    print(global_mean.to_string(index=False))
    print(f"\nMarkdown tables: {out_dir / 'TABLES.md'}")
    if not wins.empty:
        print("Wins:", wins["best_config"].value_counts().to_dict())


if __name__ == "__main__":
    main()
