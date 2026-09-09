"""Spatial vs temporal influence by station (zone 2).

Proxy (not causal):
  Informer  = temporal-only baseline (best F per station)
  AirFormer = temporal + dynamic spatial
  GAT       = temporal + fixed spatial graph

  delta_air = Informer_MA24_RMSE - AirFormer
  delta_gat = Informer_MA24_RMSE - GAT
  > 0  => that spatial model helps vs Informer
  < 0  => Informer better

Usage:
  python analysis/comparison/spatial_vs_temporal_zone2.py --zone 2
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

from src.experiments.compare_all_models import station_key  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Spatial vs temporal by station")
    p.add_argument("--zone", type=int, default=2)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: results/comparison/zone_<n>/all_models",
    )
    p.add_argument(
        "--spatial-threshold",
        type=float,
        default=0.30,
        help="MA24 RMSE gap (Informer - spatial) to call 'spatial'",
    )
    p.add_argument(
        "--temporal-threshold",
        type=float,
        default=-0.15,
        help="Gap below this => temporal (Informer better)",
    )
    return p.parse_args()


def load_metrics(zone: int) -> pd.DataFrame:
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
    return df[df["model"].isin(["informer", "airformer", "gat_informer"])].copy()


def load_coords(zone: int) -> pd.DataFrame:
    meta = pd.read_csv(
        ROOT / "data" / "metadata" / "stations_metadata.csv",
        sep=";",
        dtype=str,
    )
    meta["zone_int"] = pd.to_numeric(meta["zone"], errors="coerce")
    meta = meta[meta["zone_int"] == zone].copy()
    meta["lat"] = meta["lat"].str.replace(",", ".", regex=False).astype(float)
    meta["lon"] = meta["lon"].str.replace(",", ".", regex=False).astype(float)
    meta["station_key"] = meta["name"].map(station_key)
    return meta[["station_key", "name", "lat", "lon", "town", "type"]]


def classify(row: pd.Series, spatial_thr: float, temporal_thr: float) -> str:
    d = float(row["delta_spatial"])
    d_air = float(row["delta_air"])
    d_gat = float(row["delta_gat"])
    if d >= spatial_thr:
        if d_air >= d_gat + 0.15:
            return "spatial_dynamic"
        if d_gat >= d_air + 0.15:
            return "spatial_fixed"
        return "spatial_both"
    if d <= temporal_thr:
        return "temporal"
    return "mixed"


def build_table(df: pd.DataFrame, spatial_thr: float, temporal_thr: float) -> pd.DataFrame:
    idx = df.groupby(["station_key", "model"])["ma24_rmse"].idxmin()
    best = df.loc[
        idx,
        ["station", "station_key", "model", "feature_config", "ma24_rmse", "hourly_rmse"],
    ]
    piv = best.pivot(index="station_key", columns="model", values="ma24_rmse")
    piv_h = best.pivot(index="station_key", columns="model", values="hourly_rmse")
    piv_f = best.pivot(index="station_key", columns="model", values="feature_config")
    names = best.drop_duplicates("station_key").set_index("station_key")["station"]

    out = pd.DataFrame(
        {
            "station": names,
            "informer_ma24_rmse": piv["informer"],
            "airformer_ma24_rmse": piv["airformer"],
            "gat_ma24_rmse": piv["gat_informer"],
            "informer_hourly_rmse": piv_h["informer"],
            "airformer_hourly_rmse": piv_h["airformer"],
            "gat_hourly_rmse": piv_h["gat_informer"],
            "informer_best_F": piv_f["informer"],
            "airformer_best_F": piv_f["airformer"],
            "gat_best_F": piv_f["gat_informer"],
        }
    )
    out["spatial_best_ma24_rmse"] = out[["airformer_ma24_rmse", "gat_ma24_rmse"]].min(
        axis=1
    )
    out["delta_spatial"] = out["informer_ma24_rmse"] - out["spatial_best_ma24_rmse"]
    out["delta_air"] = out["informer_ma24_rmse"] - out["airformer_ma24_rmse"]
    out["delta_gat"] = out["informer_ma24_rmse"] - out["gat_ma24_rmse"]
    out["pct_vs_informer"] = 100.0 * out["delta_spatial"] / out["informer_ma24_rmse"]
    out["winner"] = out[
        ["informer_ma24_rmse", "airformer_ma24_rmse", "gat_ma24_rmse"]
    ].idxmin(axis=1)
    out["winner"] = out["winner"].str.replace("_ma24_rmse", "", regex=False)
    out["profile"] = out.apply(
        lambda r: classify(r, spatial_thr, temporal_thr), axis=1
    )
    return out.sort_values("delta_spatial", ascending=False).reset_index()


def plot_delta_bars(out: pd.DataFrame, path: Path) -> None:
    """Grouped bars: AirFormer and GAT each vs Informer (best F per model)."""
    # Sort by AirFormer gain (main spatial story); keep Algorta at bottom naturally
    plot_df = out.sort_values("delta_air", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 7.5))
    y = np.arange(len(plot_df))
    h = 0.38
    ax.barh(
        y - h / 2,
        plot_df["delta_air"],
        height=h,
        label="AirFormer vs Informer",
        color="#264653",
        edgecolor="none",
    )
    ax.barh(
        y + h / 2,
        plot_df["delta_gat"],
        height=h,
        label="GAT vs Informer",
        color="#e9c46a",
        edgecolor="none",
    )
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["station"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Δ MA24 RMSE = Informer − model  (>0 spatial helps)")
    ax.set_title("Spatial vs temporal by station (best F per model)")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _scatter_delta(
    ax: plt.Axes,
    merged: pd.DataFrame,
    value_col: str,
    title: str,
    vmin: float = -0.6,
    vmax: float = 0.8,
    *,
    with_basemap: bool = False,
    pad: float = 0.02,
) -> object:
    # Fix extent before basemap so tiles match the station cloud (Bajo Nervión)
    lon0, lon1 = float(merged["lon"].min()), float(merged["lon"].max())
    lat0, lat1 = float(merged["lat"].min()), float(merged["lat"].max())
    ax.set_xlim(lon0 - pad, lon1 + pad)
    ax.set_ylim(lat0 - pad, lat1 + pad)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(merged["lat"].mean())))

    if with_basemap:
        # Subtle free tiles under stations (no Stadia/Stamen: they watermark
        # "API KEY REQUIRED"). Prefer Esri gray canvas, then CartoDB.
        try:
            import contextily as cx

            source = None
            for candidate in (
                "Esri.WorldGrayCanvas",
                "CartoDB.PositronNoLabels",
                "CartoDB.Positron",
            ):
                try:
                    cur: object = cx.providers
                    for part in candidate.split("."):
                        cur = getattr(cur, part)
                    url = str(getattr(cur, "url", "") or "")
                    if "apikey" in url.lower() or "{apikey}" in url.lower():
                        continue
                    if "stadia" in url.lower() or "stamen" in url.lower():
                        continue
                    source = cur
                    break
                except AttributeError:
                    continue
            if source is not None:
                cx.add_basemap(
                    ax,
                    crs="EPSG:4326",
                    source=source,
                    attribution=False,
                    zoom="auto",
                    zorder=0,
                    alpha=0.9,
                )
            else:
                ax.set_facecolor("#f4f4f4")
        except Exception as exc:  # noqa: BLE001
            print(f"Basemap skipped ({exc}); plotting stations only.")
            ax.set_facecolor("#f4f4f4")
    else:
        ax.set_facecolor("#f7f7f7")

    sc = ax.scatter(
        merged["lon"],
        merged["lat"],
        c=merged[value_col],
        cmap="RdYlGn",
        s=200,
        edgecolors="k",
        linewidths=0.55,
        vmin=vmin,
        vmax=vmax,
        zorder=5,
    )
    for _, r in merged.iterrows():
        ax.annotate(
            str(r["station"]).replace(" (BBIZI2)", "").replace(" (Puerto)", "")[:12],
            (r["lon"], r["lat"]),
            textcoords="offset points",
            xytext=(5, 3),
            fontsize=6.5,
            color="#111",
            zorder=6,
            path_effects=None,
        )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.25, zorder=1)
    return sc


def plot_geo_map(out: pd.DataFrame, coords: pd.DataFrame, path: Path) -> None:
    if {"lat", "lon"}.issubset(out.columns):
        merged = out.copy()
    else:
        merged = out.merge(coords, on="station_key", how="left")
    missing = merged[merged["lat"].isna()]
    if not missing.empty:
        print(f"Warning: no coords for {missing['station'].tolist()}")
    merged = merged.dropna(subset=["lat", "lon"])
    if merged.empty:
        print("Warning: no stations with coordinates; skipping map.")
        return

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 6.8), sharex=True, sharey=True, constrained_layout=True
    )
    sc0 = _scatter_delta(
        axes[0],
        merged,
        "delta_air",
        "AirFormer vs Informer\n(green = AirFormer better)",
        with_basemap=True,
    )
    sc1 = _scatter_delta(
        axes[1],
        merged,
        "delta_gat",
        "GAT vs Informer\n(green = GAT better)",
        with_basemap=True,
    )
    cbar = fig.colorbar(sc1, ax=axes.ravel().tolist(), fraction=0.035, pad=0.02)
    cbar.set_label("Δ MA24 RMSE (Informer − model)")
    # No in-figure suptitle; caption lives in the manuscript.
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    _ = sc0

    # Single-panel AirFormer map (useful for slides / print guion)
    path_air = path.with_name(path.stem + "_airformer.png")
    fig2, ax2 = plt.subplots(figsize=(8.2, 7.2), constrained_layout=True)
    sc_a = _scatter_delta(
        ax2,
        merged,
        "delta_air",
        "AirFormer vs Informer · Bajo Nervión\n(green = AirFormer better)",
        with_basemap=True,
    )
    cbar2 = fig2.colorbar(sc_a, ax=ax2, fraction=0.045, pad=0.02)
    cbar2.set_label("Δ MA24 RMSE (Informer − AirFormer)")
    fig2.savefig(path_air, dpi=180, bbox_inches="tight")
    plt.close(fig2)
    print(f"Wrote {path_air}")


def plot_f3_same_config(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Same-F control: F3 only (cleaner spatial vs temporal contrast)."""
    f3 = df[df["feature_config"] == "F3"]
    piv = f3.pivot(index="station_key", columns="model", values="ma24_rmse")
    names = f3.drop_duplicates("station_key").set_index("station_key")["station"]
    out = pd.DataFrame(
        {
            "station": names,
            "informer": piv["informer"],
            "airformer": piv["airformer"],
            "gat": piv["gat_informer"],
        }
    )
    out["delta_air"] = out["informer"] - out["airformer"]
    out["delta_gat"] = out["informer"] - out["gat"]
    out = out.sort_values("delta_air", ascending=False).reset_index()

    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(out))
    h = 0.35
    ax.barh(y - h / 2, out["delta_air"], height=h, label="vs AirFormer", color="#264653")
    ax.barh(y + h / 2, out["delta_gat"], height=h, label="vs GAT", color="#e9c46a")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(out["station"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Δ MA24 RMSE = Informer − model  (>0 spatial better)")
    ax.set_title("Same config F3 · Informer vs spatial models")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()
    out_root = args.out_dir or (
        ROOT / "results" / "comparison" / f"zone_{args.zone}" / "all_models"
    )
    tab_dir = out_root / "tables"
    fig_dir = out_root / "figures"
    tab_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_metrics(args.zone)
    coords = load_coords(args.zone)
    table = build_table(df, args.spatial_threshold, args.temporal_threshold)
    table = table.merge(
        coords[["station_key", "lat", "lon", "town", "type"]],
        on="station_key",
        how="left",
    )

    csv_path = tab_dir / "spatial_vs_temporal_by_station.csv"
    table.to_csv(csv_path, index=False)

    plot_delta_bars(table, fig_dir / "fig_spatial_vs_temporal_delta.png")
    plot_geo_map(table, coords, fig_dir / "fig_spatial_vs_temporal_map.png")
    f3 = plot_f3_same_config(df, fig_dir / "fig_spatial_vs_temporal_F3.png")
    f3.to_csv(tab_dir / "spatial_vs_temporal_F3.csv", index=False)

    print(f"Wrote {csv_path}")
    print(f"Wrote {fig_dir / 'fig_spatial_vs_temporal_delta.png'}")
    print(f"Wrote {fig_dir / 'fig_spatial_vs_temporal_map.png'}")
    print(f"Wrote {fig_dir / 'fig_spatial_vs_temporal_F3.png'}")
    print("\nProfile counts:")
    print(table["profile"].value_counts().to_string())
    print("\nTop spatial benefit:")
    print(
        table.head(6)[
            ["station", "delta_spatial", "pct_vs_informer", "profile", "winner"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
