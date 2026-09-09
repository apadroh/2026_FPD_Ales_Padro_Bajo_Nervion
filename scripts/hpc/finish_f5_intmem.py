#!/usr/bin/env python3
"""Wait for XGB F5 L=96, pull metrics, dedupe, regenerate comparison tables."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DEFAULT_HOST = "ales.padro@hpc.tri.lan"
JOB_ID = "223963"
POLL_SEC = 300


def run(cmd: list[str], *, check: bool = True) -> None:
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=check)


def hpc_xgb96_count(host: str) -> int:
    out = subprocess.check_output(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=25",
            host,
            "ls ~/MASTER/results/xgboost/zone_2/seq_len_96/*/F5/results.json 2>/dev/null | wc -l",
        ],
        text=True,
    )
    return int(out.strip().splitlines()[-1])


def job_running(host: str, job_id: str) -> bool:
    out = subprocess.check_output(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=25",
            host,
            f"squeue -j {job_id} -h -o %T 2>/dev/null || true",
        ],
        text=True,
    ).strip()
    return bool(out) and "RUNNING" in out or "PENDING" in out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--job-id", default=JOB_ID)
    p.add_argument("--poll-sec", type=int, default=POLL_SEC)
    p.add_argument("--skip-wait", action="store_true")
    return p.parse_args()


def regenerate_tables() -> None:
    run([sys.executable, "src/experiments/compare_all_models.py", "--zone", "2"])
    run([sys.executable, "scripts/analysis/seq_len_metrics_table.py"])


def main() -> int:
    args = parse_args()
    if not args.skip_wait:
        while True:
            n = hpc_xgb96_count(args.host)
            running = job_running(args.host, args.job_id)
            print(f"XGB L=96: {n} results on HPC; job {args.job_id} running={running}", flush=True)
            if n >= 18 and not running:
                break
            if not running and n >= 17:
                # allow 17 real + job done (duplicate may inflate count briefly)
                break
            time.sleep(args.poll_sec)

    run([sys.executable, "scripts/hpc/pull_metrics_only.py", "--zone", "2"])
    run(
        [
            sys.executable,
            "scripts/hpc/cleanup_duplicate_stations.py",
            "--zone",
            "2",
            "--models",
            "informer2020",
            "xgboost",
            "airformer",
            "gat_informer_mf_pm10graph",
        ]
    )
    regenerate_tables()
    print("F5 int+mem campaign: pull + tables complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
