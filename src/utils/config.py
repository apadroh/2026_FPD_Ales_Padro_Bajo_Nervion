from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Centralized project configuration."""

    root_dir: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = root_dir / "data"

    # API
    api_base_url: str = os.getenv("API_BASE_URL", "https://api.euskadi.eus/air-quality")

    # Extraction defaults
    output_stations_csv_dir: Path = data_dir / "raw" / "stations_csv"
    extract_start_year: int = int(os.getenv("EXTRACT_START_YEAR", "2012"))
    extract_end_year: int = int(os.getenv("EXTRACT_END_YEAR", "2025"))

    # Metadata
    stations_metadata_csv: Path = data_dir / "metadata" / "metadatos_estaciones.csv"
    stations_meteo_check_csv: Path = data_dir / "metadata" / "stations_meteo_check.csv"
    saharian_days_csv: Path = data_dir / "raw" / "saharianos.csv"
    saharian_start_year: int = int(os.getenv("SAHARIAN_START_YEAR", "2006"))

    # Dataset build paths
    stations_merged_dir: Path = data_dir / "intermediate" / "stations_merged"
    processed_dir: Path = data_dir / "processed"


settings = Settings()
