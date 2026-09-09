"""
EDA parametrizable por zona y contaminante.

Ejecutar desde la raiz del proyecto:
    python analysis/eda/eda_zone_contaminant.py --zone 2 --contaminant PM10

Salida:
    analysis/eda/zone_<n>/<CONT>/
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import seaborn as sns

    HAS_SNS = True
except ImportError:
    HAS_SNS = False

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONTAMINANTS = ["NO2", "SO2", "PM10", "CO", "O3", "NO", "PM25", "NOx"]
METEO_VARS = ["DV", "VV", "T", "H", "PR", "LL", "RA"]
ANALYSIS_VARS = CONTAMINANTS + METEO_VARS + ["Sah"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EDA por zona y contaminante")
    parser.add_argument("--zone", type=int, required=True)
    parser.add_argument(
        "--contaminant",
        type=str,
        required=True,
        choices=CONTAMINANTS,
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Ruta opcional al CSV definitivo (override)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directorio de salida (default: analysis/eda/zone_<n>/<CONT>)",
    )
    return parser.parse_args()


def resolve_paths(zone: int, contaminant: str, dataset: str | None, output_dir: str | None):
    ds = (
        Path(dataset)
        if dataset
        else ROOT / f"data/training/zone_{zone}/dataset_zone_{zone}_{contaminant}.csv"
    )
    out = (
        Path(output_dir)
        if output_dir
        else ROOT / f"analysis/eda/zone_{zone}/{contaminant}"
    )
    seq = ROOT / f"analysis/acf/zone_{zone}/{contaminant}/seq_len_recommendation.json"
    return ds, out, seq


def setup_style() -> None:
    plt.style.use("ggplot")
    if HAS_SNS:
        sns.set_palette("husl")


def save_fig(output_dir: Path, name: str) -> None:
    path = output_dir / "figures" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figura: {path.relative_to(ROOT)}")


def save_csv(output_dir: Path, df: pd.DataFrame, name: str) -> None:
    path = output_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  CSV   : {path.relative_to(ROOT)}")


def plot_heatmap(output_dir: Path, data: pd.DataFrame, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    if HAS_SNS:
        sns.heatmap(data, annot=True, fmt=".0f", cmap="YlGnBu", vmin=0, vmax=100, ax=ax)
    else:
        im = ax.imshow(data.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)
        ax.set_xticks(range(len(data.columns)))
        ax.set_xticklabels(data.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(data.index)))
        ax.set_yticklabels(data.index)
        plt.colorbar(im, ax=ax, label="%")
    ax.set_title(title)
    save_fig(output_dir, filename)


def plot_corr_heatmap(output_dir: Path, corr: pd.DataFrame, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    if HAS_SNS:
        sns.heatmap(corr, cmap="coolwarm", center=0.5, vmin=0, vmax=1, ax=ax)
    else:
        im = ax.imshow(corr.values, cmap="coolwarm", vmin=0, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index, fontsize=7)
        plt.colorbar(im, ax=ax)
    ax.set_title(title)
    save_fig(output_dir, filename)


def compute_acf(series: np.ndarray, max_lag: int = 168) -> np.ndarray:
    centered = series - series.mean()
    acf = np.ones(max_lag + 1)
    for lag in range(1, max_lag + 1):
        if lag >= len(centered):
            break
        acf[lag] = np.corrcoef(centered[:-lag], centered[lag:])[0, 1]
    return acf


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)


def run_eda(zone: int, contaminant: str, dataset_path: Path, output_dir: Path, seq_rec_path: Path) -> None:
    setup_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    target = contaminant

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"No existe {dataset_path}. "
            f"Genera primero: python src/data/build_definitive_datasets.py --zones {zone}"
        )

    section(f"EDA zone {zone} / {contaminant}")
    print(f"Dataset: {dataset_path.relative_to(ROOT)}")
    print(f"Salida : {output_dir.relative_to(ROOT)}")

    section("1. Load and structure")
    df = pd.read_csv(dataset_path, parse_dates=["time"], low_memory=False)
    df = df.sort_values(["time", "ID"]).reset_index(drop=True)

    if target not in df.columns:
        raise ValueError(f"El dataset no tiene columna {target}")

    print(f"Filas       : {len(df):,}")
    print(f"Estaciones  : {df['station_name'].nunique()}")
    print(f"Horas       : {df['time'].nunique():,}")
    print(f"Periodo     : {df['time'].min()} -> {df['time'].max()}")

    stations = (
        df[["ID", "station_name", "lat", "lon"]]
        .drop_duplicates()
        .sort_values("station_name")
    )
    save_csv(output_dir, stations, "01_stations.csv")

    section("2. Availability")
    avail_rows = []
    for station_name, group in df.groupby("station_name"):
        for var in ANALYSIS_VARS:
            if var not in df.columns:
                continue
            avail_rows.append(
                {
                    "station_name": station_name,
                    "variable": var,
                    "availability_pct": group[var].notna().mean() * 100,
                }
            )
    availability = pd.DataFrame(avail_rows)
    save_csv(output_dir, availability, "02_availability.csv")
    avail_pivot = availability.pivot(
        index="station_name", columns="variable", values="availability_pct"
    )
    plot_heatmap(
        output_dir,
        avail_pivot,
        "Disponibilidad (%) por estacion y variable",
        "02_availability_heatmap.png",
    )

    section(f"3. Temporal coverage {target}")
    coverage_rows = []
    for station_name, group in df.groupby("station_name"):
        valid = group[group[target].notna()]
        coverage_rows.append(
            {
                "station_name": station_name,
                "start": valid["time"].min() if not valid.empty else pd.NaT,
                "end": valid["time"].max() if not valid.empty else pd.NaT,
                "n_valid": len(valid),
                "availability_pct": group[target].notna().mean() * 100,
            }
        )
    coverage = pd.DataFrame(coverage_rows).sort_values("start")
    save_csv(output_dir, coverage, f"03_{target.lower()}_coverage.csv")

    fig, ax = plt.subplots(figsize=(12, 6))
    for _, row in coverage.iterrows():
        if pd.isna(row["start"]):
            continue
        ax.barh(
            row["station_name"],
            (row["end"] - row["start"]).total_seconds() / 3600,
            left=row["start"],
            height=0.6,
        )
    ax.set_title(f"Cobertura temporal de {target}")
    save_fig(output_dir, f"03_{target.lower()}_coverage_gantt.png")

    section(f"4. {target} statistics")
    desc = (
        df.groupby("station_name")[target]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .round(2)
        .sort_values("mean", ascending=False)
    )
    save_csv(output_dir, desc.reset_index(), f"04_{target.lower()}_descriptives.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if HAS_SNS:
        sns.boxplot(data=df, x="station_name", y=target, ax=axes[0])
    else:
        labels = sorted(df["station_name"].unique())
        data_box = [df.loc[df["station_name"] == s, target].dropna() for s in labels]
        axes[0].boxplot(data_box, labels=labels)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=90, fontsize=7)
    axes[0].set_title(f"Boxplot {target}")
    df[target].dropna().hist(bins=60, ax=axes[1], edgecolor="white")
    axes[1].set_title(f"Distribucion global {target}")
    save_fig(output_dir, f"04_{target.lower()}_distributions.png")

    section("5. Time series")
    target_wide = df.pivot(index="time", columns="station_name", values=target)
    zonal_mean = target_wide.mean(axis=1)

    fig, ax = plt.subplots(figsize=(14, 4))
    zonal_mean.plot(ax=ax, color="black", linewidth=0.8)
    ax.set_title(f"{target} — media zonal horaria")
    ax.set_ylabel("ug/m3")
    save_fig(output_dir, "05_zonal_mean_timeseries.png")

    sample_stations = sorted(df["station_name"].unique())[:6]
    n = max(len(sample_stations), 1)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.2 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, station in zip(axes, sample_stations):
        series = df.loc[df["station_name"] == station].set_index("time")[target]
        series.plot(ax=ax, linewidth=0.5)
        ax.set_ylabel(station, rotation=0, ha="right", fontsize=8)
    fig.suptitle(f"{target} — muestra de estaciones")
    save_fig(output_dir, "05_sample_stations_timeseries.png")

    section("5b. STL decomposition")
    try:
        from src.utils.decomposition_plot import save_stl_html

        html_path = save_stl_html(
            df,
            output_dir / "figures" / f"{target.lower()}_stl_decomposition.html",
            target=target,
            period=24,
            last_days=730,
        )
        print(f"  HTML  : {html_path.relative_to(ROOT)}")
    except ImportError as exc:
        print(f"  [SKIP] descomposicion STL: {exc}")

    section("6. Temporal patterns")
    df_ts = df[["time", "station_name", target]].copy()
    df_ts["hour"] = df_ts["time"].dt.hour
    df_ts["dow"] = df_ts["time"].dt.dayofweek
    df_ts["month"] = df_ts["time"].dt.month
    df_ts["year"] = df_ts["time"].dt.year

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    df_ts.groupby("hour")[target].mean().plot(kind="bar", ax=axes[0], color="steelblue")
    axes[0].set_title("Patron diario")
    weekly = df_ts.groupby("dow")[target].mean()
    weekly.index = ["Lun", "Mar", "Mie", "Jue", "Vie", "Sab", "Dom"]
    weekly.plot(kind="bar", ax=axes[1], color="darkorange")
    axes[1].set_title("Patron semanal")
    df_ts.groupby("month")[target].mean().plot(kind="bar", ax=axes[2], color="seagreen")
    axes[2].set_title("Patron mensual")
    save_fig(output_dir, "06_temporal_patterns.png")

    yearly = df_ts.groupby(["year", "station_name"])[target].mean().reset_index()
    save_csv(output_dir, yearly, "06_yearly_means.csv")
    fig, ax = plt.subplots(figsize=(12, 5))
    for station in yearly["station_name"].unique():
        sub = yearly[yearly["station_name"] == station]
        ax.plot(sub["year"], sub[target], alpha=0.35, linewidth=1)
    yearly.groupby("year")[target].mean().plot(
        ax=ax, color="black", linewidth=2.5, label="Media zonal"
    )
    ax.legend()
    ax.set_title(f"Evolucion anual {target}")
    save_fig(output_dir, "06_yearly_evolution.png")

    section("7. Spatial correlation")
    spatial_corr = target_wide.corr()
    save_csv(output_dir, spatial_corr.reset_index(), "07_spatial_correlation.csv")
    plot_corr_heatmap(
        output_dir,
        spatial_corr,
        f"Correlacion espacial de {target}",
        "07_spatial_correlation.png",
    )

    section("8. Correlation with meteo")
    meteo_cols = [c for c in METEO_VARS + ["Sah"] if c in df.columns]
    corr_meteo = (
        df[[target] + meteo_cols].corr()[target].drop(target).sort_values(ascending=False)
    )
    save_csv(output_dir, corr_meteo.reset_index(name="correlation"), "08_meteo_correlation.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    corr_meteo.plot(kind="barh", ax=ax, color="teal")
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_title(f"Correlacion {target} vs meteo/Sah")
    save_fig(output_dir, "08_meteo_correlation.png")

    section("9. ACF and input window (seq_len, all models)")
    max_lag = 168
    acf_by_station = {}
    for station_name, group in df.groupby("station_name"):
        values = group.sort_values("time")[target].dropna().to_numpy()
        if len(values) > max_lag + 10:
            acf_by_station[station_name] = compute_acf(values, max_lag)

    if acf_by_station:
        acf_matrix = np.vstack(list(acf_by_station.values()))
        median_acf = np.median(acf_matrix, axis=0)
        lags = np.arange(len(median_acf))

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(lags, median_acf, linewidth=2, label="ACF mediana")
        ax.fill_between(
            lags,
            np.percentile(acf_matrix, 25, axis=0),
            np.percentile(acf_matrix, 75, axis=0),
            alpha=0.2,
            label="P25-P75",
        )
        ax.axhline(0.3, color="red", linestyle="--", label="Umbral 0.3")
        if seq_rec_path.exists():
            with open(seq_rec_path, encoding="utf-8") as f:
                rec = json.load(f)
            ax.axvline(
                rec["recommended_seq_len"],
                color="green",
                linestyle="--",
                label=f"seq_len={rec['recommended_seq_len']} h",
            )
            print(f"seq_len recomendado: {rec['recommended_seq_len']} h")
        ax.legend()
        ax.set_title(f"Autocorrelacion {target}")
        save_fig(output_dir, "09_acf.png")

        key_lags = [1, 6, 12, 24, 48, 72, 168]
        acf_table = pd.DataFrame(
            {
                "lag_h": key_lags,
                "acf_median": [
                    median_acf[lag] if lag < len(median_acf) else np.nan
                    for lag in key_lags
                ],
            }
        )
        save_csv(output_dir, acf_table, "09_acf_key_lags.csv")

    section("10. Saharan days")
    if "Sah" in df.columns:
        sah = df.copy()
        sah["Sah"] = sah["Sah"].fillna(0)
        sah["episodio"] = sah["Sah"].astype(bool)
        sah_summary = (
            sah.groupby("episodio")[target].agg(["count", "mean", "median", "max"]).round(2)
        )
        save_csv(output_dir, sah_summary.reset_index(), "10_sah_summary.csv")
        fig, ax = plt.subplots(figsize=(8, 4))
        if HAS_SNS:
            sns.boxplot(data=sah, x="episodio", y=target, ax=ax)
        else:
            data_box = [
                sah.loc[~sah["episodio"], target].dropna(),
                sah.loc[sah["episodio"], target].dropna(),
            ]
            ax.boxplot(data_box, labels=["No Sahariano", "Sahariano"])
        ax.set_title(f"{target} en dias sahariano vs normales")
        save_fig(output_dir, "10_sah_boxplot.png")

    section("11. Extremes and quality")
    thresholds = [50, 100, 150]
    extreme_rows = []
    for station_name, group in df.groupby("station_name"):
        s = group[target].dropna()
        row = {"station_name": station_name, "n": len(s)}
        for thr in thresholds:
            row[f"pct_above_{thr}"] = (s > thr).mean() * 100 if len(s) else np.nan
        extreme_rows.append(row)
    save_csv(output_dir, pd.DataFrame(extreme_rows), "11_extremes.csv")

    dup_check = df.groupby(["time", "ID"]).size()
    gaps = df.groupby("station_name").apply(
        lambda g: g.sort_values("time")["time"].diff().dt.total_seconds().div(3600).median(),
        include_groups=False,
    )
    print(f"Duplicados (time, ID): {(dup_check > 1).sum()}")
    print(f"Cadencia mediana (h): {gaps.median():.1f}")

    section("12. Modeling summary")
    summary = {
        "zone": zone,
        "contaminant": contaminant,
        "dataset": str(dataset_path.relative_to(ROOT)),
        "periodo": f"{df['time'].min()} -> {df['time'].max()}",
        "n_estaciones": int(df["station_name"].nunique()),
        "n_horas": int(df["time"].nunique()),
        f"{target.lower()}_media_zonal": round(float(df[target].mean()), 2),
        f"{target.lower()}_std_zonal": round(float(df[target].std()), 2),
    }
    if seq_rec_path.exists():
        with open(seq_rec_path, encoding="utf-8") as f:
            rec = json.load(f)
        summary["seq_len_recomendado_h"] = rec.get("recommended_seq_len")
        summary["horizon_default_h"] = rec.get("horizon")

    summary_path = output_dir / "12_modeling_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  JSON  : {summary_path.relative_to(ROOT)}")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print(f"\nEDA completado. Resultados en: {output_dir.relative_to(ROOT)}")


def main() -> None:
    args = parse_args()
    dataset_path, output_dir, seq_rec_path = resolve_paths(
        args.zone, args.contaminant, args.dataset, args.output_dir
    )
    run_eda(args.zone, args.contaminant, dataset_path, output_dir, seq_rec_path)


if __name__ == "__main__":
    main()
