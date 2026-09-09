"""Saharan dust intensity (MITECO-style net load) from saharianos.csv."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAHARIAN_CSV = ROOT / "data" / "raw" / "saharianos.csv"


def load_sah_intensity_by_day(
    path: Path | str | None = None,
    *,
    source: str = "Valderejo",
    clip_min: float = 0.0,
) -> pd.Series:
    """
    Daily net African dust load (µg/m³) keyed by normalized date.

    Uses the Valderejo (or Pagoeta) column from the official episode table —
    validated against MITECO P40 methodology on zone-8 Valderejo PM10
    (corr ≈ 0.93). Days absent from the table → 0.
    """
    csv_path = Path(path) if path is not None else DEFAULT_SAHARIAN_CSV
    if not csv_path.exists():
        return pd.Series(dtype=float)

    df = pd.read_csv(csv_path, sep=";")
    df.columns = df.columns.str.strip()
    if "Fecha" not in df.columns or source not in df.columns:
        raise ValueError(
            f"{csv_path} must have Fecha and {source}; got {list(df.columns)}"
        )

    df["date"] = pd.to_datetime(df["Fecha"], dayfirst=True, errors="coerce").dt.normalize()
    df[source] = pd.to_numeric(df[source], errors="coerce")
    df = df.dropna(subset=["date"])
    # One value per day; keep last if duplicates
    s = (
        df.groupby("date", sort=True)[source]
        .last()
        .clip(lower=clip_min)
        .fillna(0.0)
        .astype(float)
    )
    return s


def attach_sah_int(
    df: pd.DataFrame,
    *,
    time_col: str = "time",
    intensity: pd.Series | None = None,
) -> pd.DataFrame:
    """Add Sah_int column (hourly rows inherit the day's intensity)."""
    out = df.copy()
    times = pd.to_datetime(out[time_col], errors="coerce")
    days = times.dt.normalize()
    if intensity is None:
        intensity = load_sah_intensity_by_day()
    out["Sah_int"] = days.map(intensity).fillna(0.0).astype(float)
    return out


SAH_INT_CONTEXT_COLS = [
    "Sah_int_lag1",
    "Sah_int_lag2",
    "Sah_int_lag3",
    "Sah_int_dur",
    "Sah_int_max3",
]


def add_sah_int_context_features(
    df: pd.DataFrame,
    *,
    time_col: str = "time",
    sah_col: str = "Sah_int",
) -> pd.DataFrame:
    """
    Add daily context around Sah_int (same value for all stations at a timestamp).

    - Sah_int_lag{1,2,3}: intensity 1–3 calendar days earlier
    - Sah_int_dur: consecutive active days (Sah_int>0) ending today
    - Sah_int_max3: max(Sah_int) over today and previous 2 days
    """
    out = df.copy()
    if sah_col not in out.columns:
        for c in SAH_INT_CONTEXT_COLS:
            out[c] = 0.0
        return out

    times = pd.to_datetime(out[time_col], errors="coerce")
    day = times.dt.normalize()
    daily = out.assign(_day=day).groupby("_day", sort=True)[sah_col].first().astype(float)

    lag1 = daily.shift(1).fillna(0.0)
    lag2 = daily.shift(2).fillna(0.0)
    lag3 = daily.shift(3).fillna(0.0)
    max3 = pd.concat([daily, lag1, lag2], axis=1).max(axis=1)

    active = (daily > 0).astype(int)
    # Run length of active days ending at each day
    dur = active.copy()
    run = 0
    vals = []
    for a in active.to_numpy():
        run = run + 1 if a else 0
        vals.append(float(run))
    dur = pd.Series(vals, index=active.index, dtype=float)

    out["Sah_int_lag1"] = day.map(lag1).fillna(0.0).astype(float)
    out["Sah_int_lag2"] = day.map(lag2).fillna(0.0).astype(float)
    out["Sah_int_lag3"] = day.map(lag3).fillna(0.0).astype(float)
    out["Sah_int_dur"] = day.map(dur).fillna(0.0).astype(float)
    out["Sah_int_max3"] = day.map(max3).fillna(0.0).astype(float)
    return out
