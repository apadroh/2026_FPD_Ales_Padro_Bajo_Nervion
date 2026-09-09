"""
Export training packs for the H=48 h campaign (L=48, horizon=48).

AirFormer + GAT-MF NPZ under horizon_48/; clones GAT to pm10graph adjacency.
Informer / XGBoost reuse existing CSV packs (pred_len set at train time).

Usage:
  python scripts/hpc/export_horizon48_packs.py
  python scripts/hpc/export_horizon48_packs.py --feature-configs F5
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GRAPH = ROOT / "data/graphs/pm10_graph_zone_2_top5.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export H=48 data packs (zone 2)")
    p.add_argument("--zone", type=int, default=2)
    p.add_argument(
        "--feature-configs",
        nargs="+",
        default=["F1", "F2", "F3", "F4", "F5"],
    )
    p.add_argument("--seq-len", type=int, default=48)
    p.add_argument("--horizon", type=int, default=48)
    p.add_argument("--skip-clone", action="store_true")
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    args = parse_args()
    cfgs = " ".join(args.feature_configs)
    py = sys.executable

    run(
        [
            py,
            "src/data/export_airformer_zone2.py",
            "--zone",
            str(args.zone),
            "--seq-len",
            str(args.seq_len),
            "--horizon",
            str(args.horizon),
            "--feature-configs",
            *args.feature_configs,
        ]
    )
    run(
        [
            py,
            "src/data/export_gat_informer_mf_zone2.py",
            "--zone",
            str(args.zone),
            "--seq-len",
            str(args.seq_len),
            "--horizon",
            str(args.horizon),
            "--feature-configs",
            *args.feature_configs,
        ]
    )

    if args.skip_clone:
        return

    if not GRAPH.exists():
        raise SystemExit(f"Missing PM10 graph: {GRAPH}")

    for cfg in args.feature_configs:
        run(
            [
                py,
                "src/data/clone_gat_pack_alt_graph.py",
                "--zone",
                str(args.zone),
                "--src-config",
                cfg,
                "--graph-csv",
                str(GRAPH),
                "--tag",
                "pm10graph",
                "--seq-len",
                str(args.seq_len),
                "--horizon",
                str(args.horizon),
            ]
        )

    print(
        f"\nDone. Packs under horizon_{args.horizon}/ for AirFormer and "
        f"gat_informer_mf_pm10graph. Launch with:\n"
        f"  python scripts/hpc/launch_from_pc.py --model airformer --all-configs "
        f"--horizon {args.horizon} --upload-raw-data"
    )


if __name__ == "__main__":
    main()
