#!/usr/bin/env python3
"""Pull only metrics artifacts (results.json, batch summaries) from HPC."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.stations import dedupe_station_directories  # noqa: E402

DEFAULT_HOST = os.environ.get("HPC_HOST", "ales.padro@hpc.tri.lan")
DEFAULT_REMOTE = os.environ.get("HPC_REMOTE_DIR", "~/MASTER")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--remote-dir", default=DEFAULT_REMOTE)
    p.add_argument("--zone", type=int, default=2)
    p.add_argument("--key", default=str(Path.home() / ".ssh" / "id_rsa"))
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def ssh_opts(key: str) -> list[str]:
    opts = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=25"]
    if key and Path(key).exists():
        opts += ["-i", key]
    return opts


def run(cmd: list[str], *, dry_run: bool = False) -> None:
    print(">>", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()
    remote_tar = f"/tmp/zone_{args.zone}_metrics.tgz"
    rel_roots = [
        f"results/informer2020/zone_{args.zone}",
        f"results/xgboost/zone_{args.zone}",
        f"results/airformer/zone_{args.zone}",
        f"results/gat_informer_mf_pm10graph/zone_{args.zone}",
    ]
    find_cmd = (
        f"cd {args.remote_dir} && "
        f"find {' '.join(rel_roots)} "
        r"\( -name results.json -o -name 'batch_summary_*.csv' -o -name per_station_summary.csv \) "
        f"| tar czf {remote_tar} -T - && ls -lh {remote_tar}"
    )
    run(["ssh", *ssh_opts(args.key), args.host, find_cmd], dry_run=args.dry_run)

    local_tar = Path(tempfile.gettempdir()) / f"zone_{args.zone}_metrics.tgz"
    run(
        ["scp", *ssh_opts(args.key), f"{args.host}:{remote_tar}", str(local_tar)],
        dry_run=args.dry_run,
    )
    if args.dry_run:
        return 0

    print(f"Extracting into {ROOT} ...")
    with tarfile.open(local_tar, "r:gz") as tar:
        tar.extractall(ROOT, filter="data")

    for model_root in [
        ROOT / f"results/informer2020/zone_{args.zone}",
        ROOT / f"results/xgboost/zone_{args.zone}",
        ROOT / f"results/airformer/zone_{args.zone}/ZONE2_PM10",
        ROOT / f"results/gat_informer_mf_pm10graph/zone_{args.zone}",
    ]:
        if model_root.exists():
            dedupe_station_directories(model_root)

    run(["ssh", *ssh_opts(args.key), args.host, f"rm -f {remote_tar}"], dry_run=False)
    print("Metrics pull complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
