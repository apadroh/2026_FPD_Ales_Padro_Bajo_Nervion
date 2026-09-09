"""Canonical project paths (Informer2020 layout)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Definitive training datasets
TRAINING_ROOT = ROOT / "data" / "training"
ANALYSIS_ROOT = ROOT / "analysis"
RESULTS_ROOT = ROOT / "results"
MODELS_NOTEBOOKS_ROOT = ROOT / "models"

BASELINE_SEQ_LEN = 48
BASELINE_HORIZON = 24


def seq_len_tag(seq_len: int | None) -> str | None:
    """Subfolder name for non-baseline history lengths; None → keep legacy paths."""
    if seq_len is None or int(seq_len) == BASELINE_SEQ_LEN:
        return None
    return f"seq_len_{int(seq_len)}"


def horizon_tag(horizon: int | None) -> str | None:
    """Subfolder name for non-baseline forecast horizons; None → keep legacy paths."""
    if horizon is None or int(horizon) == BASELINE_HORIZON:
        return None
    return f"horizon_{int(horizon)}"


def apply_layout_tags(
    base: Path,
    *,
    seq_len: int | None = None,
    horizon: int | None = None,
) -> Path:
    """Insert seq_len_L / horizon_H subdirs (baseline values add no segment)."""
    tag = seq_len_tag(seq_len)
    if tag:
        base = base / tag
    htag = horizon_tag(horizon)
    if htag:
        base = base / htag
    return base


def definitive_csv(zone: int, contaminant: str) -> Path:
    return TRAINING_ROOT / f"zone_{zone}" / f"dataset_zone_{zone}_{contaminant}.csv"


# Backward-compatible alias
quasi_definitive_csv = definitive_csv


def informer2020_station_dir(
    zone: int,
    station: str,
    feature_config: str,
) -> Path:
    station_key = station.replace(" ", "_").replace("(", "").replace(")", "")
    return (
        TRAINING_ROOT
        / f"zone_{zone}"
        / "informer2020"
        / station_key
        / feature_config
    )


def informer2020_results_dir(
    zone: int,
    station: str,
    feature_config: str,
    seq_len: int | None = None,
    horizon: int | None = None,
) -> Path:
    station_key = station.replace(" ", "_").replace("(", "").replace(")", "")
    base = apply_layout_tags(
        RESULTS_ROOT / "informer2020" / f"zone_{zone}",
        seq_len=seq_len,
        horizon=horizon,
    )
    return base / station_key / feature_config


def analysis_acf_dir(zone: int, contaminant: str) -> Path:
    """ACF / seq_len shared across all models."""
    return ANALYSIS_ROOT / "acf" / f"zone_{zone}" / contaminant


def analysis_feature_selection_dir(zone: int, contaminant: str) -> Path:
    """Availability, lag correlations, and F1–F5 ablation."""
    return ANALYSIS_ROOT / "feature_selection" / f"zone_{zone}" / contaminant


def airformer_upstream() -> Path:
    return ROOT / "src" / "models" / "airformer" / "_upstream"


def airformer_results_dir(
    zone: int,
    dataset: str = "ZONE2_PM10",
    feature_config: str | None = None,
    seq_len: int | None = None,
    horizon: int | None = None,
) -> Path:
    base = apply_layout_tags(
        RESULTS_ROOT / "airformer" / f"zone_{zone}" / dataset,
        seq_len=seq_len,
        horizon=horizon,
    )
    if feature_config:
        return base / feature_config
    return base


def airformer_data_dir(
    zone: int,
    feature_config: str,
    dataset: str = "ZONE2_PM10",
    seq_len: int | None = None,
    horizon: int | None = None,
) -> Path:
    base = apply_layout_tags(
        TRAINING_ROOT / f"zone_{zone}" / "airformer" / dataset,
        seq_len=seq_len,
        horizon=horizon,
    )
    return base / feature_config


def gat_informer_upstream() -> Path:
    return ROOT / "src" / "models" / "gat_informer" / "_upstream"


def gat_informer_data_dir(zone: int, feature_config: str) -> Path:
    return TRAINING_ROOT / f"zone_{zone}" / "gat_informer" / feature_config


def gat_informer_results_dir(zone: int, feature_config: str = "F3") -> Path:
    return RESULTS_ROOT / "gat_informer" / f"zone_{zone}" / feature_config


def gat_informer_mf_data_dir(
    zone: int,
    feature_config: str,
    *,
    seq_len: int | None = None,
    horizon: int | None = None,
) -> Path:
    base = apply_layout_tags(
        TRAINING_ROOT / f"zone_{zone}" / "gat_informer_mf",
        seq_len=seq_len,
        horizon=horizon,
    )
    return base / feature_config


def gat_informer_mf_results_dir(
    zone: int,
    feature_config: str = "F3",
    *,
    seq_len: int | None = None,
    horizon: int | None = None,
) -> Path:
    base = apply_layout_tags(
        RESULTS_ROOT / "gat_informer_mf" / f"zone_{zone}",
        seq_len=seq_len,
        horizon=horizon,
    )
    return base / feature_config


def gnn_informer_results_dir(zone: int, feature_config: str = "F3") -> Path:
    """Sequential GNN→Informer results (reuses gat_informer_mf data packs)."""
    return RESULTS_ROOT / "gnn_informer" / f"zone_{zone}" / feature_config


def gat_informer_mf_pm10graph_data_dir(
    zone: int,
    feature_config: str,
    *,
    seq_len: int | None = None,
    horizon: int | None = None,
) -> Path:
    base = apply_layout_tags(
        TRAINING_ROOT / f"zone_{zone}" / "gat_informer_mf_pm10graph",
        seq_len=seq_len,
        horizon=horizon,
    )
    return base / feature_config


def gat_informer_mf_pm10graph_results_dir(
    zone: int,
    feature_config: str = "F1",
    seq_len: int | None = None,
    horizon: int | None = None,
) -> Path:
    base = apply_layout_tags(
        RESULTS_ROOT / "gat_informer_mf_pm10graph" / f"zone_{zone}",
        seq_len=seq_len,
        horizon=horizon,
    )
    return base / feature_config


def xgboost_results_zone_dir(
    zone: int,
    seq_len: int | None = None,
    horizon: int | None = None,
) -> Path:
    return apply_layout_tags(
        RESULTS_ROOT / "xgboost" / f"zone_{zone}",
        seq_len=seq_len,
        horizon=horizon,
    )
