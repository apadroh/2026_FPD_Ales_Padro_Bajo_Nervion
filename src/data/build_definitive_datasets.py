"""
Build definitive per-contaminant datasets from dataset_zone_*.csv.

For each zone:
  1. Analyze common periods (100% station coverage)
  2. Trim to the recommended valid period per contaminant
  3. Write data/training/zone_<n>/dataset_zone_<n>_<CONT>.csv
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONTAMINANTS = [
    "NO2",
    "SO2",
    "PM10",
    "CO",
    "O3",
    "NO",
    "PM25",
    "NOx",
]
METEO_VARS = ["DV", "VV", "T", "H", "PR", "LL", "RA"]
DATA_COLS = CONTAMINANTS + METEO_VARS + ["Sah", "Sah_int"]


@dataclass(frozen=True)
class StrictPeriod:
    contaminant: str
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    n_stations: int
    n_valid_in_period: int
    n_valid_total: int
    data_pct: float
    duration_hours: int
    is_valid: bool
    limiting_start_stations: str
    limiting_end_stations: str


# ---------------------------------------------------------------------------
# Common periods (formerly analyze_common_periods.py)
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"], low_memory=False)
    return df.sort_values(["time", "ID"])


def station_periods(df: pd.DataFrame, contaminant: str) -> pd.DataFrame:
    if contaminant not in df.columns:
        return pd.DataFrame()

    subset = df[["station_name", "time", contaminant]].copy()
    subset = subset[subset[contaminant].notna()]
    if subset.empty:
        return pd.DataFrame()

    rows = []
    for station_name, group in subset.groupby("station_name", sort=True):
        rows.append(
            {
                "contaminant": contaminant,
                "station_name": station_name,
                "start": group["time"].min(),
                "end": group["time"].max(),
                "n_valid": int(group[contaminant].notna().sum()),
            }
        )

    periods = pd.DataFrame(rows)
    total_hours = df["time"].nunique()
    periods["availability_pct"] = (
        periods["n_valid"] / total_hours * 100
    ).round(2)
    return periods.sort_values(["start", "station_name"])


def count_valid_in_window(
    df: pd.DataFrame,
    contaminant: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> int:
    mask = (
        df[contaminant].notna()
        & (df["time"] >= start)
        & (df["time"] <= end)
    )
    return int(mask.sum())


def strict_period_100(periods: pd.DataFrame) -> StrictPeriod:
    contaminant = periods["contaminant"].iloc[0]
    n_stations = len(periods)
    n_valid_total = int(periods["n_valid"].sum())

    start = periods["start"].max()
    end = periods["end"].min()
    valid = bool(start <= end)

    limiting_start = periods.loc[
        periods["start"] == start, "station_name"
    ].tolist()
    limiting_end = periods.loc[
        periods["end"] == end, "station_name"
    ].tolist()

    n_valid_in_period = 0
    duration_hours = 0
    data_pct = 0.0

    if valid:
        duration_hours = int(((end - start) / pd.Timedelta(hours=1)) + 1)
        n_valid_in_period = n_stations * duration_hours
        data_pct = (
            round(n_valid_in_period / n_valid_total * 100, 2)
            if n_valid_total
            else 0.0
        )

    return StrictPeriod(
        contaminant=contaminant,
        start=start,
        end=end,
        n_stations=n_stations,
        n_valid_in_period=n_valid_in_period,
        n_valid_total=n_valid_total,
        data_pct=data_pct,
        duration_hours=duration_hours,
        is_valid=valid,
        limiting_start_stations="; ".join(limiting_start),
        limiting_end_stations="; ".join(limiting_end),
    )


def analyze_contaminant(
    df: pd.DataFrame,
    contaminant: str,
) -> tuple[pd.DataFrame, StrictPeriod | None]:
    periods = station_periods(df, contaminant)
    if periods.empty:
        return periods, None

    result = strict_period_100(periods)

    if result.is_valid:
        actual_valid = count_valid_in_window(
            df, contaminant, result.start, result.end
        )
        n_total = int(df[contaminant].notna().sum())
        result = StrictPeriod(
            contaminant=result.contaminant,
            start=result.start,
            end=result.end,
            n_stations=result.n_stations,
            n_valid_in_period=actual_valid,
            n_valid_total=n_total,
            data_pct=round(actual_valid / n_total * 100, 2) if n_total else 0.0,
            duration_hours=result.duration_hours,
            is_valid=True,
            limiting_start_stations=result.limiting_start_stations,
            limiting_end_stations=result.limiting_end_stations,
        )

    return periods, result


def station_retention_in_period(
    df: pd.DataFrame,
    contaminant: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    if contaminant not in df.columns:
        return pd.DataFrame()

    subset = df[["station_name", "time", contaminant]].copy()
    rows = []
    for station_name, group in subset.groupby("station_name", sort=True):
        n_valid_total = int(group[contaminant].notna().sum())
        if n_valid_total == 0:
            continue

        window = group[(group["time"] >= start) & (group["time"] <= end)]
        n_valid_in_period = int(window[contaminant].notna().sum())
        n_hours_in_period = int(window["time"].nunique())
        coverage_in_period_pct = (
            round(n_valid_in_period / n_hours_in_period * 100, 2)
            if n_hours_in_period
            else 0.0
        )

        rows.append(
            {
                "contaminant": contaminant,
                "station_name": station_name,
                "period_start": start,
                "period_end": end,
                "n_valid_total": n_valid_total,
                "n_valid_in_period": n_valid_in_period,
                "data_retention_pct": round(
                    n_valid_in_period / n_valid_total * 100, 2
                ),
                "n_hours_in_period": n_hours_in_period,
                "coverage_in_period_pct": coverage_in_period_pct,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["contaminant", "data_retention_pct", "station_name"],
        ascending=[True, False, True],
    )


def save_period_outputs(
    output_dir: Path,
    station_periods_all: pd.DataFrame,
    strict_periods: pd.DataFrame,
    station_retention: pd.DataFrame | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "station_periods_by_contaminant.csv": station_periods_all,
        "common_periods_summary.csv": strict_periods,
        "recommended_periods.csv": strict_periods,
    }
    if station_retention is not None and not station_retention.empty:
        paths["station_data_retention.csv"] = station_retention

    for filename, frame in paths.items():
        path = output_dir / filename
        frame.to_csv(path, index=False)
        print(f"Guardado: {path}")


def run_common_periods_analysis(
    df: pd.DataFrame,
    output_dir: Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    all_periods: list[pd.DataFrame] = []
    strict_rows: list[dict] = []
    retention_frames: list[pd.DataFrame] = []

    for contaminant in CONTAMINANTS:
        periods, result = analyze_contaminant(df, contaminant)
        if not periods.empty:
            all_periods.append(periods)
        if result is not None:
            row = result.__dict__.copy()
            row["period_type"] = "estricto_100pct_cobertura"
            strict_rows.append(row)
            if verbose:
                status = "OK" if result.is_valid else "NO VALIDO"
                print(
                    f"  {contaminant}: {status} | "
                    f"estaciones={result.n_stations} | "
                    f"{result.start} -> {result.end}"
                )

        if result is not None and result.is_valid:
            retention = station_retention_in_period(
                df, contaminant, result.start, result.end
            )
            if not retention.empty:
                retention_frames.append(retention)

    station_periods_all = (
        pd.concat(all_periods, ignore_index=True)
        if all_periods
        else pd.DataFrame()
    )
    strict_periods = pd.DataFrame(strict_rows)
    station_retention = (
        pd.concat(retention_frames, ignore_index=True)
        if retention_frames
        else pd.DataFrame()
    )

    if output_dir is not None:
        save_period_outputs(
            output_dir,
            station_periods_all,
            strict_periods,
            station_retention,
        )

    return strict_periods


# ---------------------------------------------------------------------------
# Definitive build
# ---------------------------------------------------------------------------

def parse_zones(text: str) -> list[int]:
    """Acepta listas y rangos: '1,3-8' -> [1,3,4,5,6,7,8]."""
    zones: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            zones.extend(range(int(start_s), int(end_s) + 1))
        else:
            zones.append(int(part))
    return sorted(set(zones))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze common periods and build definitive datasets "
            "por zona y contaminante"
        )
    )
    parser.add_argument(
        "--zones",
        type=str,
        default="1-8",
        help="Zonas a procesar. Ej: '2', '1,3-8', '1-8'. Default: 1-8",
    )
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Directory with dataset_zone_<n>.csv",
    )
    parser.add_argument(
        "--training-dir",
        default="data/training",
        help="Output directory for definitive CSVs (zone_<n>/)",
    )
    parser.add_argument(
        "--analysis-dir",
        default="analysis/common_periods",
        help="Output directory for period analysis (dataset_zone_<n>/)",
    )
    parser.add_argument(
        "--quiet-periods",
        action="store_true",
        help="Less verbose period analysis",
    )
    return parser.parse_args()


def availability_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        rows.append(
            {
                "variable": col,
                "availability_pct": round(df[col].notna().mean() * 100, 2),
                "n_valid": int(df[col].notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def build_contaminant_dataset(
    df: pd.DataFrame,
    contaminant: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, list[str]]:
    stations = station_periods(df, contaminant)["station_name"].tolist()
    if not stations:
        return pd.DataFrame(), []

    mask = (
        df["station_name"].isin(stations)
        & (df["time"] >= start)
        & (df["time"] <= end)
    )
    subset = df.loc[mask].sort_values(["time", "ID"]).reset_index(drop=True)
    return subset, stations


def build_zone(
    zone: int,
    processed_dir: Path,
    training_dir: Path,
    analysis_dir: Path,
    quiet_periods: bool,
) -> None:
    dataset_path = processed_dir / f"dataset_zone_{zone}.csv"
    if not dataset_path.exists():
        print(f"[SKIP] Zona {zone}: no existe {dataset_path}")
        return

    periods_dir = analysis_dir / f"dataset_zone_{zone}"
    output_dir = training_dir / f"zone_{zone}"
    manifest_path = periods_dir / "definitive_manifest.csv"

    print("\n" + "#" * 80)
    print(f"# ZONA {zone}")
    print("#" * 80)
    print(f"Dataset base: {dataset_path}")

    df = load_dataset(dataset_path)

    print(f"\nAnalizando periodos comunes -> {periods_dir}")
    recommended = run_common_periods_analysis(
        df,
        output_dir=periods_dir,
        verbose=not quiet_periods,
    )

    if recommended.empty:
        print(f"[SKIP] Zona {zone}: sin resultados de periodos")
        return

    valid = recommended[recommended["is_valid"]].copy()
    if valid.empty:
        print(f"[SKIP] Zona {zone}: ningun periodo estricto valido")
        return

    valid["start"] = pd.to_datetime(valid["start"])
    valid["end"] = pd.to_datetime(valid["end"])

    manifest_rows: list[dict] = []
    print(f"\nGenerando {len(valid)} datasets definitivos en {output_dir}")
    print("=" * 80)

    for _, period in valid.iterrows():
        contaminant = period["contaminant"]
        start = period["start"]
        end = period["end"]

        subset, stations = build_contaminant_dataset(
            df, contaminant, start, end
        )
        if subset.empty:
            print(f"[SKIP] {contaminant}: sin datos tras filtrar")
            continue

        output_file = output_dir / f"dataset_zone_{zone}_{contaminant}.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        subset.to_csv(output_file, index=False)

        avail = availability_summary(subset, DATA_COLS)
        target_avail = avail.loc[
            avail["variable"] == contaminant, "availability_pct"
        ]
        target_pct = float(target_avail.iloc[0]) if not target_avail.empty else 0.0

        manifest_rows.append(
            {
                "zone": zone,
                "contaminant": contaminant,
                "output_file": str(output_file.as_posix()),
                "period_start": start,
                "period_end": end,
                "n_rows": len(subset),
                "n_stations": len(stations),
                "n_timestamps": subset["time"].nunique(),
                "target_contaminant_availability_pct": target_pct,
                "stations": "; ".join(stations),
            }
        )

        print(f"\n{contaminant}")
        print(f"  Archivo : {output_file}")
        print(f"  Periodo : {start} -> {end}")
        print(f"  Filas   : {len(subset):,}")
        print(f"  Estaciones: {len(stations)}")
        print(f"  Disponibilidad {contaminant}: {target_pct}%")

    skipped = set(CONTAMINANTS) - set(valid["contaminant"])
    if skipped:
        print("\nContaminantes omitidos (periodo estricto no valido):")
        for contaminant in sorted(skipped):
            print(f"  - {contaminant}")

    manifest = pd.DataFrame(manifest_rows)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    print(f"\nManifiesto: {manifest_path}")


def main() -> None:
    args = parse_args()
    zones = parse_zones(args.zones)
    processed_dir = Path(args.processed_dir)
    training_dir = Path(args.training_dir)
    analysis_dir = Path(args.analysis_dir)

    print(f"Zonas a procesar: {zones}")

    for zone in zones:
        build_zone(
            zone=zone,
            processed_dir=processed_dir,
            training_dir=training_dir,
            analysis_dir=analysis_dir,
            quiet_periods=args.quiet_periods,
        )

    print("\nGeneracion definitiva finalizada.")


if __name__ == "__main__":
    main()
