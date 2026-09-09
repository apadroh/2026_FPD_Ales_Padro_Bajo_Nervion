"""
AirFormer (AAAI-23) — official vendored code.

Source: https://github.com/yoshall/AirFormer

The full repo lives in `_upstream/`. Import its API with
PYTHONPATH = `_upstream` (not MASTER `src/`, which would collide).
"""

from __future__ import annotations

from pathlib import Path

UPSTREAM_ROOT = Path(__file__).resolve().parent / "_upstream"
MAIN_SCRIPT = UPSTREAM_ROOT / "experiments" / "airformer" / "main.py"
DATA_ZIP = UPSTREAM_ROOT / "data" / "data.zip"
TINY_DATA_DIR = UPSTREAM_ROOT / "data" / "AIR_TINY"

__all__ = [
    "UPSTREAM_ROOT",
    "MAIN_SCRIPT",
    "DATA_ZIP",
    "TINY_DATA_DIR",
]
