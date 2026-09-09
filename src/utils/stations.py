"""Station name normalization and encoding-duplicate cleanup."""

from __future__ import annotations

import re
import shutil
import unicodedata
from pathlib import Path

CANONICAL_DIAZ_HARO = "Mª_DIAZ_HARO"


def station_key(name: str) -> str:
    """Normalize station folder names for grouping (handles Mª / mojibake)."""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.upper()
    s = re.sub(r"[^A-Z0-9]+", "_", s).strip("_")
    if "DIAZ" in s and "HARO" in s:
        return "M_DIAZ_HARO"
    return s


def is_encoding_duplicate(name: str) -> bool:
    """True if *name* is a mojibake alias of Mª_DIAZ_HARO."""
    if station_key(name) != "M_DIAZ_HARO":
        return False
    return name not in {CANONICAL_DIAZ_HARO, "M_DIAZ_HARO"}


def pick_canonical_station_name(names: list[str]) -> str:
    """Pick the preferred folder name among encoding variants."""
    if CANONICAL_DIAZ_HARO in names:
        return CANONICAL_DIAZ_HARO
    if "M_DIAZ_HARO" in names:
        return "M_DIAZ_HARO"
    # Fallback: shortest name without obvious mojibake bytes.
    clean = [n for n in names if not is_encoding_duplicate(n)]
    if clean:
        return sorted(clean, key=len)[0]
    return sorted(names, key=len)[0]


def dedupe_station_names(names: list[str]) -> list[str]:
    """Collapse encoding duplicates; preserve first-seen order."""
    groups: dict[str, list[str]] = {}
    order: list[str] = []
    for name in names:
        key = station_key(name)
        if key not in groups:
            order.append(key)
            groups[key] = []
        groups[key].append(name)
    out: list[str] = []
    for key in order:
        out.append(pick_canonical_station_name(groups[key]))
    return out


def _station_dirs(parent: Path) -> list[Path]:
    skip = {"comparison", "__pycache__"}
    out: list[Path] = []
    if not parent.is_dir():
        return out
    for child in sorted(parent.iterdir()):
        if not child.is_dir():
            continue
        if child.name in skip or child.name.startswith("seq_len_"):
            continue
        if child.name.startswith("batch_summary_"):
            continue
        out.append(child)
    return out


def _safe_path_label(path: Path | str) -> str:
    return str(path).encode("ascii", "backslashreplace").decode("ascii")


def dedupe_station_directories(
    root: Path,
    *,
    dry_run: bool = False,
) -> list[tuple[Path, Path]]:
    """Remove encoding-duplicate station folders under *root* (and seq_len_*)."""
    removed: list[tuple[Path, Path]] = []
    if not root.exists():
        return removed

    parents = [root]
    for child in sorted(root.iterdir()):
        if child.is_dir() and child.name.startswith("seq_len_"):
            parents.append(child)

    for parent in parents:
        dirs = _station_dirs(parent)
        groups: dict[str, list[Path]] = {}
        for d in dirs:
            groups.setdefault(station_key(d.name), []).append(d)
        for key, variants in groups.items():
            if len(variants) <= 1:
                continue
            keep_name = pick_canonical_station_name([v.name for v in variants])
            for d in variants:
                if d.name == keep_name:
                    continue
                removed.append((d, parent / keep_name))
                if dry_run:
                    print(
                        f"[dry-run] would remove duplicate {_safe_path_label(d)} "
                        f"(keep {_safe_path_label(keep_name)})"
                    )
                else:
                    print(
                        f"Removing duplicate station dir {_safe_path_label(d)} "
                        f"(keep {_safe_path_label(keep_name)})"
                    )
                    shutil.rmtree(d)
    return removed
