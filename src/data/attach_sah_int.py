"""Attach Sah_int to definitive zone CSVs (no full rebuild)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.saharan_intensity import attach_sah_int, load_sah_intensity_by_day
from src.utils.paths import definitive_csv


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--zones", type=int, nargs="+", default=[2])
    p.add_argument("--contaminant", default="PM10")
    args = p.parse_args()

    intensity = load_sah_intensity_by_day()
    print(f"Sah_int calendar days with >0: {(intensity > 0).sum()} / {len(intensity)}")

    for zone in args.zones:
        path = definitive_csv(zone, args.contaminant)
        if not path.exists():
            print(f"SKIP missing {path}")
            continue
        import pandas as pd

        df = pd.read_csv(path)
        before = "Sah_int" in df.columns
        df = attach_sah_int(df, intensity=intensity)
        df.to_csv(path, index=False)
        nz = float((df["Sah_int"] > 0).mean() * 100)
        print(
            f"OK zone_{zone}: wrote Sah_int "
            f"(replaced={before})  rows={len(df)}  pct>0={nz:.2f}%  "
            f"max={df['Sah_int'].max():.1f}"
        )


if __name__ == "__main__":
    main()
