#!/usr/bin/env python3
"""Remove mojibake duplicate station folders (Mª_DIAZ_HARO vs M┬к_DIAZ_HARO).

Examples:
  python scripts/hpc/cleanup_duplicate_stations.py --zone 2
  python scripts/hpc/cleanup_duplicate_stations.py --zone 2 --remote
  python scripts/hpc/cleanup_duplicate_stations.py --path results/xgboost/zone_2
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.stations import dedupe_station_directories  # noqa: E402

DEFAULT_HOST = os.environ.get("HPC_HOST", "ales.padro@hpc.tri.lan")
DEFAULT_REMOTE = os.environ.get("HPC_REMOTE_DIR", "~/MASTER")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zone", type=int, default=2)
    p.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Explicit local results root (e.g. results/xgboost/zone_2)",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=["informer2020", "xgboost"],
        choices=["informer2020", "xgboost", "airformer", "gat_informer_mf_pm10graph"],
    )
    p.add_argument("--remote", action="store_true", help="Run cleanup on HPC via ssh")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--remote-dir", default=DEFAULT_REMOTE)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def cleanup_local(args: argparse.Namespace) -> int:
    roots: list[Path] = []
    if args.path:
        roots.append(args.path if args.path.is_absolute() else ROOT / args.path)
    else:
        for model in args.models:
            if model == "informer2020":
                roots.append(ROOT / f"results/informer2020/zone_{args.zone}")
            elif model == "xgboost":
                roots.append(ROOT / f"results/xgboost/zone_{args.zone}")
            elif model == "airformer":
                roots.append(
                    ROOT / f"results/airformer/zone_{args.zone}/ZONE2_PM10"
                )
            elif model == "gat_informer_mf_pm10graph":
                roots.append(ROOT / f"results/gat_informer_mf_pm10graph/zone_{args.zone}")

    n = 0
    for root in roots:
        if not root.exists():
            print(f"Skip missing {root}")
            continue
        print(f"=== {root} ===")
        removed = dedupe_station_directories(root, dry_run=args.dry_run)
        n += len(removed)
    print(f"Done. {'Would remove' if args.dry_run else 'Removed'} {n} duplicate folder(s).")
    return 0


def cleanup_remote(args: argparse.Namespace) -> int:
    models = " ".join(args.models)
    cmd = (
        f"cd {args.remote_dir} && "
        f"python3 scripts/hpc/cleanup_duplicate_stations.py "
        f"--zone {args.zone} --models {models}"
    )
    if args.dry_run:
        cmd += " --dry-run"
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=25", args.host, cmd]
    print(">>", " ".join(ssh))
    return subprocess.run(ssh, check=False).returncode


def main() -> int:
    args = parse_args()
    if args.remote:
        return cleanup_remote(args)
    return cleanup_local(args)


if __name__ == "__main__":
    raise SystemExit(main())
