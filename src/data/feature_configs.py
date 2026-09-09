"""
Feature variable configurations for pollutant forecasting (F1–F5 ablation).

F1: autoregressive (target only)
F2: + temporal features
F3: + meteorology
F4: + co-pollutants
F5: F4 + Saharan intensity + episode memory (no binary Sah)

Any Saharan-associated pack uses continuous Valderejo net load (Sah_int)
plus short episode context (lags 1–3, duration, max3). The binary flag
``Sah`` is not used in active configs.

Extras (aliases / historical names; same Saharan block as F3/F5 paths):
F3S / F3I / F3IL: F3 + Sah_int + episode memory
F5I: same feature list as F5 (F4 + intensity + memory)
"""

from __future__ import annotations

from dataclasses import dataclass

CONTAMINANTS = ["NO2", "SO2", "PM10", "CO", "O3", "NO", "PM25", "NOx"]

AQ_CANDIDATES = [
    "PM10",
    "PM25",
    "NO2",
    "NO",
    "NOx",
    "SO2",
    "CO",
    "O3",
]

METEO_CANDIDATES = ["VV", "wind_sin", "wind_cos", "T", "H", "PR", "LL", "RA"]

TIME_FEATURES = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
]

METEO_F3 = ["VV", "wind_sin", "wind_cos", "T", "H", "PR"]

# Continuous intensity + short episode memory (replaces binary Sah).
SAHARAN_INTENSITY_MEMORY = [
    "Sah_int",
    "Sah_int_lag1",
    "Sah_int_lag2",
    "Sah_int_lag3",
    "Sah_int_dur",
    "Sah_int_max3",
]

F4_ADDITIONS: dict[str, list[str]] = {
    "PM10": ["PM25", "NO2", "NOx", "SO2"],
    "PM25": ["PM10", "NO2", "NOx", "SO2"],
    "NO2": ["PM10", "PM25", "NOx", "SO2"],
    "NOx": ["PM10", "PM25", "NO2", "SO2"],
    "SO2": ["PM10", "PM25", "NO2", "NOx"],
    "CO": ["PM10", "PM25", "NO2", "NOx"],
    "O3": ["PM10", "NO2", "NOx", "SO2"],
    "NO": ["PM10", "NO2", "NOx", "SO2"],
}

DEFAULT_F4 = ["PM25", "NO2", "NOx", "SO2"]

# Main TFM cumulative ablation (do not reorder).
ABLATION_LEVELS = ["F1", "F2", "F3", "F4", "F5"]
# Extra configs outside the cumulative ladder.
EXTRA_FEATURE_LEVELS = ["F3S", "F3I", "F5I", "F3IL"]
ALL_FEATURE_LEVELS = ABLATION_LEVELS + EXTRA_FEATURE_LEVELS

# Levels that attach Saharan intensity + episode memory (never binary Sah).
SAHARAN_LEVELS = frozenset({"F5", "F3S", "F3I", "F5I", "F3IL"})

LEVEL_DESCRIPTIONS = {
    "F1": "Autoregressive (target only)",
    "F2": "Target + temporal features",
    "F3": "F2 + meteorology",
    "F4": "F3 + co-pollutants",
    "F5": "F4 + Sah_int + episode memory (lags/dur/max3); no binary Sah",
    "F3S": "F3 + Sah_int + episode memory (alias of F3IL path)",
    "F3I": "F3 + Sah_int + episode memory",
    "F5I": "F4 + Sah_int + episode memory (same as F5)",
    "F3IL": "F3 + Sah_int + episode memory (lags/dur/max3)",
}


@dataclass(frozen=True)
class FeatureConfig:
    contaminant: str
    level: str
    target: str
    features: tuple[str, ...]
    description: str


def f4_pollutants_for(contaminant: str) -> list[str]:
    extras = F4_ADDITIONS.get(contaminant, DEFAULT_F4)
    return [c for c in extras if c != contaminant]


def build_feature_list(contaminant: str, level: str) -> list[str]:
    if level not in ALL_FEATURE_LEVELS:
        raise ValueError(
            f"Invalid level '{level}'. Options: {ALL_FEATURE_LEVELS}"
        )

    target = contaminant
    features: list[str] = [target]

    if level in {"F2", "F3", "F4", "F5", "F3S", "F3I", "F5I", "F3IL"}:
        features.extend(TIME_FEATURES)

    if level in {"F3", "F4", "F5", "F3S", "F3I", "F5I", "F3IL"}:
        features.extend(METEO_F3)

    if level in {"F4", "F5", "F5I"}:
        features.extend(f4_pollutants_for(contaminant))

    # Saharan packs: intensity + episode memory only (never binary Sah).
    if level in SAHARAN_LEVELS:
        features.extend(SAHARAN_INTENSITY_MEMORY)

    seen: set[str] = set()
    ordered: list[str] = []
    for col in features:
        if col not in seen:
            seen.add(col)
            ordered.append(col)
    return ordered


def get_feature_config(contaminant: str, level: str) -> FeatureConfig:
    features = build_feature_list(contaminant, level)
    return FeatureConfig(
        contaminant=contaminant,
        level=level,
        target=contaminant,
        features=tuple(features),
        description=LEVEL_DESCRIPTIONS[level],
    )


def resolve_features_in_dataset(
    df,
    contaminant: str,
    level: str,
    min_availability_pct: float = 80.0,
) -> tuple[list[str], list[str], dict[str, float]]:
    """
    Return (features_used, features_excluded, global_availability_pct).
    """
    config = get_feature_config(contaminant, level)
    availability = {
        col: float(df[col].notna().mean() * 100)
        for col in config.features
        if col in df.columns
    }

    used = [
        col
        for col in config.features
        if col in df.columns and availability.get(col, 0.0) >= min_availability_pct
    ]
    missing_cols = [col for col in config.features if col not in df.columns]
    low_avail = [
        col
        for col in config.features
        if col in df.columns and availability.get(col, 0.0) < min_availability_pct
    ]
    excluded = missing_cols + low_avail

    if config.target not in used:
        raise ValueError(
            f"Target {config.target} does not reach {min_availability_pct}% "
            f"availability ({availability.get(config.target, 0):.1f}%)"
        )

    return used, excluded, availability


def default_dataset_path(zone: int, contaminant: str) -> str:
    """Definitive training CSV (new layout)."""
    return (
        f"data/training/zone_{zone}/"
        f"dataset_zone_{zone}_{contaminant}.csv"
    )


def default_output_dir(zone: int, contaminant: str, level: str) -> str:
    """Legacy tensors (simplified Informer). Prefer Informer2020."""
    return (
        f"data/training/zone_{zone}/informer_legacy/"
        f"{contaminant}/{level}"
    )


def default_informer2020_data_root() -> str:
    return "data/training"


def default_informer2020_results_root() -> str:
    return "results/informer2020"
