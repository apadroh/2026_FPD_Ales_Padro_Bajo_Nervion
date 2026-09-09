from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
INGEST_SCRIPT = ROOT_DIR / "src" / "data" / "ingest_stations.py"

PIPELINE_STEPS = {
    "ingest": INGEST_SCRIPT,
    "metadata": INGEST_SCRIPT,
    "extract": INGEST_SCRIPT,
    "merge": INGEST_SCRIPT,
    "build-meteo-graph": ROOT_DIR / "src" / "graphs" / "build_meteo_graph.py",
    "build-topk-graph": ROOT_DIR / "src" / "graphs" / "build_topk_graph.py",
    "build-pollution-graph": ROOT_DIR / "src" / "graphs" / "build_pollution_graph.py",
    "check-meteo": ROOT_DIR / "src" / "data" / "check_meteo_graph.py",
    "build": ROOT_DIR / "src" / "data" / "build_dataset_by_zone.py",
    "definitive": ROOT_DIR / "src" / "data" / "build_definitive_datasets.py",
}

DEFAULT_ORDER = [
    "ingest",
    "build-meteo-graph",
    "build-topk-graph",
    "build-pollution-graph",
    "check-meteo",
    "build",
    "definitive",
]

INGEST_STEP_MAP = {
    "ingest": "all",
    "metadata": "metadata",
    "extract": "extract",
    "merge": "merge",
}


def run_step(step: str, extra_args: list[str] | None = None) -> int:
    script_path = PIPELINE_STEPS[step]

    if not script_path.exists():
        print(f"[ERROR] Script no encontrado para '{step}': {script_path}")
        return 1

    cmd = [sys.executable, str(script_path)]

    if step in INGEST_STEP_MAP:
        cmd.extend(["--step", INGEST_STEP_MAP[step]])

    if extra_args:
        cmd.extend(extra_args)

    print(f"[INFO] Ejecutando paso '{step}' -> {script_path}")
    if extra_args or step in INGEST_STEP_MAP:
        print(f"[INFO] Args: {cmd[2:]}")
    result = subprocess.run(cmd, cwd=ROOT_DIR)
    return result.returncode


def run_pipeline(steps: list[str], step_extra: dict[str, list[str]] | None = None) -> int:
    step_extra = step_extra or {}
    for step in steps:
        code = run_step(step, step_extra.get(step))
        if code != 0:
            print(f"[ERROR] Paso '{step}' falló con código {code}.")
            return code

    print("[OK] Pipeline finalizado correctamente.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Orquestador del pipeline de calidad del aire."
    )
    parser.add_argument(
        "--step",
        choices=list(PIPELINE_STEPS.keys()),
        help="Ejecuta solo un paso concreto.",
    )
    parser.add_argument(
        "--from-step",
        choices=DEFAULT_ORDER,
        help="Ejecuta desde este paso hasta el final del pipeline.",
    )
    parser.add_argument(
        "--zones",
        type=str,
        default=None,
        help=(
            "Zonas para build / definitive (ej: 1,3-8 o 1-8). "
            "Se reenvía a los scripts correspondientes."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Con build: también genera dataset_all.csv",
    )
    parser.add_argument(
        "--test-all",
        action="store_true",
        help="Con build: también genera dataset_test_all.csv",
    )
    return parser.parse_args()


def build_extra_args(args: argparse.Namespace) -> list[str]:
    extra: list[str] = []
    if args.zones:
        extra.extend(["--zones", args.zones])
    if args.all:
        extra.append("--all")
    if args.test_all:
        extra.append("--test-all")
    return extra


def definitive_extra_args(args: argparse.Namespace) -> list[str]:
    extra: list[str] = []
    if args.zones:
        extra.extend(["--zones", args.zones])
    return extra


def main() -> int:
    args = parse_args()
    step_extra = {
        "build": build_extra_args(args),
        "definitive": definitive_extra_args(args),
    }

    if args.step:
        return run_step(args.step, step_extra.get(args.step))

    if args.from_step:
        start_idx = DEFAULT_ORDER.index(args.from_step)
        return run_pipeline(DEFAULT_ORDER[start_idx:], step_extra)

    return run_pipeline(DEFAULT_ORDER, step_extra)


if __name__ == "__main__":
    raise SystemExit(main())
