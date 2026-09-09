"""
GAT-Informer — hybrid static spatial GNN + temporal Informer.

Source: https://github.com/ChengqingYu/GAT-Informer
Official code vendored in `_upstream/`.
"""

from __future__ import annotations

from pathlib import Path

UPSTREAM_ROOT = Path(__file__).resolve().parent / "_upstream"
MAIN_SCRIPT = UPSTREAM_ROOT / "main_GAT_Informer.py"
BLOCK_DIR = UPSTREAM_ROOT / "block"
METRIC_DIR = UPSTREAM_ROOT / "metric"

__all__ = [
    "UPSTREAM_ROOT",
    "MAIN_SCRIPT",
    "BLOCK_DIR",
    "METRIC_DIR",
]
