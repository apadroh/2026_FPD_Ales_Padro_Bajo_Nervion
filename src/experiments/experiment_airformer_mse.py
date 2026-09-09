"""
Optional AirFormer ablation: masked MSE training loss (separate from official MAE).

Official ZONE2 results stay under results/airformer/.../F5/.
This entry point forces --train-loss mse and --results-tag _mse so outputs land in
e.g. results/airformer/zone_2/ZONE2_PM10/F5_mse/ without touching F5/.

Usage:
  python src/experiments/experiment_airformer_mse.py --feature-config F5
  python src/experiments/experiment_airformer_mse.py --feature-config F1 F5 --max-epochs 2
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WRAPPER = ROOT / "src" / "experiments" / "experiment_airformer.py"


def _has_flag(argv: list[str], name: str) -> bool:
    prefix = f"--{name}"
    return any(a == prefix or a.startswith(prefix + "=") for a in argv)


def main() -> None:
    argv = sys.argv[1:]
    injected: list[str] = []
    if not _has_flag(argv, "train-loss"):
        injected.extend(["--train-loss", "mse"])
    if not _has_flag(argv, "results-tag"):
        injected.extend(["--results-tag", "_mse"])
    # Peak-weighted loss is MAE-only upstream.
    if not _has_flag(argv, "peak-weight-alpha"):
        injected.extend(["--peak-weight-alpha", "0"])
    if not _has_flag(argv, "peak-asym-beta"):
        injected.extend(["--peak-asym-beta", "0"])

    cmd = [sys.executable, str(WRAPPER), *argv, *injected]
    print("AirFormer MSE ablation ->", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT))


if __name__ == "__main__":
    main()
