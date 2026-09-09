"""
Launch MASTER trainings on KATEA HPC from your PC / Cursor terminal.

Headline models: informer, airformer, airformer_mse, gat_mf_pm10 (GAT-Informer adopted), xgboost.

Examples:
  python scripts/hpc/launch_from_pc.py --model xgboost --all-configs --upload-raw-data
  python scripts/hpc/launch_from_pc.py --model informer --feature-config F5 --stations SAN_MIGUEL --upload-raw-data
  python scripts/hpc/launch_from_pc.py --model gat_mf_pm10 --feature-config F1 --upload-raw-data
  python scripts/hpc/launch_from_pc.py --model airformer --feature-config F1 --upload-raw-data
  python scripts/hpc/launch_from_pc.py --status
  python scripts/hpc/launch_from_pc.py --pull-results --model xgboost
  python scripts/hpc/launch_from_pc.py --pull-all-models
  python scripts/hpc/launch_from_pc.py --local --model gat_mf_pm10 --feature-config F1

See also: scripts/hpc/README.md

Requires OpenSSH (scp/ssh). Passphrase is prompted in the terminal.
Env overrides: HPC_HOST, HPC_KEY, HPC_REMOTE_DIR
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.paths import airformer_data_dir as _airformer_data_dir
from src.utils.paths import apply_layout_tags, gat_informer_mf_pm10graph_data_dir
from src.utils.stations import dedupe_station_directories  # noqa: E402

DEFAULT_HOST = os.environ.get("HPC_HOST", "ales.padro@hpc.tri.lan")
DEFAULT_KEY = os.environ.get(
    "HPC_KEY", str(Path.home() / ".ssh" / "id_rsa")
)
DEFAULT_REMOTE = os.environ.get("HPC_REMOTE_DIR", "~/MASTER")

MODEL_SCRIPTS = {
    "informer": "scripts/hpc/slurm_informer.sh",
    "airformer": "scripts/hpc/slurm_airformer.sh",
    "airformer_mse": "scripts/hpc/slurm_airformer_mse.sh",
    "gat_mf_pm10": "scripts/hpc/slurm_gat_informer_mf_pm10graph.sh",
    "xgboost": "scripts/hpc/slurm_xgboost.sh",
}

CPU_MODEL_SCRIPTS = {
    "airformer": "scripts/hpc/slurm_airformer_cpu.sh",
    "airformer_mse": "scripts/hpc/slurm_airformer_mse_cpu.sh",
    "informer": "scripts/hpc/slurm_informer_cpu.sh",
    "gat_mf_pm10": "scripts/hpc/slurm_gat_informer_mf_pm10graph_cpu.sh",
}


def slurm_script_for_model(args: argparse.Namespace) -> str:
    rel = MODEL_SCRIPTS[args.model]
    if getattr(args, "partition", None) == "cpu":
        rel = CPU_MODEL_SCRIPTS.get(args.model, rel)
    return rel

# Extra source files to scp before sbatch (so HPC has latest train/experiment code).
MODEL_CODE_SYNC = {
    "informer": [
        "src/experiments/run_informer2020_batch.py",
        "src/experiments/experiment_informer2020.py",
        "src/data/feature_configs.py",
        "src/data/saharan_intensity.py",
        "src/utils/paths.py",
        "src/utils/stations.py",
        "src/models/informer2020/metrics.py",
        "scripts/hpc/cleanup_duplicate_stations.py",
        "scripts/hpc/slurm_informer.sh",
        "scripts/hpc/slurm_informer_cpu.sh",
    ],
    "airformer": [
        "src/experiments/experiment_airformer.py",
        "src/data/feature_configs.py",
        "src/data/saharan_intensity.py",
        "src/utils/paths.py",
        "src/utils/stations.py",
        "src/models/informer2020/metrics.py",
        "scripts/hpc/cleanup_duplicate_stations.py",
        "src/data/export_airformer_zone2.py",
        "scripts/hpc/slurm_airformer.sh",
        "scripts/hpc/slurm_airformer_cpu.sh",
    ],
    "airformer_mse": [
        "src/experiments/experiment_airformer.py",
        "src/experiments/experiment_airformer_mse.py",
        "src/models/airformer/_upstream/src/base/trainer.py",
        "src/models/airformer/_upstream/experiments/airformer/main.py",
        "src/data/feature_configs.py",
        "src/data/saharan_intensity.py",
        "src/utils/paths.py",
        "src/utils/stations.py",
        "src/models/informer2020/metrics.py",
        "scripts/hpc/cleanup_duplicate_stations.py",
        "src/data/export_airformer_zone2.py",
        "scripts/hpc/slurm_airformer_mse.sh",
        "scripts/hpc/slurm_airformer_mse_cpu.sh",
    ],
    "gat_mf_pm10": [
        "src/experiments/experiment_gat_informer_mf.py",
        "src/models/gat_informer/zone2_train_mf.py",
        "src/data/feature_configs.py",
        "src/data/saharan_intensity.py",
        "src/utils/paths.py",
        "src/utils/stations.py",
        "src/models/informer2020/metrics.py",
        "scripts/hpc/cleanup_duplicate_stations.py",
        "scripts/hpc/slurm_gat_informer_mf_pm10graph.sh",
        "scripts/hpc/slurm_gat_informer_mf_pm10graph_cpu.sh",
    ],
    "xgboost": [
        "src/experiments/experiment_xgboost.py",
        "src/models/informer2020/metrics.py",
        "src/data/air_quality_dataset.py",
        "src/data/feature_configs.py",
        "src/data/saharan_intensity.py",
        "src/utils/paths.py",
        "src/utils/stations.py",
        "src/models/informer2020/metrics.py",
        "scripts/hpc/cleanup_duplicate_stations.py",
        "scripts/hpc/slurm_xgboost.sh",
        "docs/xgboost_baseline.md",
    ],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Launch MASTER trainings from PC")
    p.add_argument(
        "--model",
        choices=[
            "informer",
            "airformer",
            "airformer_mse",
            "gat_mf_pm10",
            "xgboost",
        ],
        default="informer",
    )
    p.add_argument(
        "--feature-config",
        choices=["F1", "F2", "F3", "F4", "F5", "F3S", "F3I", "F5I", "F3IL"],
        default="F5",
    )
    p.add_argument(
        "--feature-configs",
        nargs="+",
        choices=["F1", "F2", "F3", "F4", "F5", "F3S", "F3I", "F5I", "F3IL"],
        default=None,
        help="Submit several configs (e.g. F2 F4)",
    )
    p.add_argument(
        "--all-configs",
        action="store_true",
        help="Submit F1..F5 (Informer) or available configs in one go",
    )
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--key", default=DEFAULT_KEY)
    p.add_argument("--remote-dir", default=DEFAULT_REMOTE)
    p.add_argument(
        "--upload-data",
        action="store_true",
        help="Deprecated alias of --upload-raw-data (old zip packs removed)",
    )
    p.add_argument(
        "--upload-raw-data",
        action="store_true",
        help=(
            "scp data/training/... packs only (AirFormer / GAT / GAT-MF / "
            "gat_mf_pm10 / Informer CSVs). Requires repo already on HPC."
        ),
    )
    p.add_argument(
        "--status",
        action="store_true",
        help="Only show squeue / recent sacct (no submit)",
    )
    p.add_argument(
        "--pull-results",
        action="store_true",
        help="Download results/ for this model from HPC",
    )
    p.add_argument(
        "--pull-all-models",
        action="store_true",
        help="Download results for informer, airformer, gat_mf_pm10 and xgboost",
    )
    p.add_argument(
        "--dedupe-stations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove mojibake duplicate station folders before/after pull (default: on)",
    )
    p.add_argument(
        "--pull-logs",
        action="store_true",
        help="Download recent logs/slurm/{model}_*.{out,err} for debugging",
    )
    p.add_argument(
        "--local",
        action="store_true",
        help="Run batch on this PC instead of HPC",
    )
    p.add_argument("--preset", choices=["paper", "fast"], default="paper")
    p.add_argument("--zone", type=int, default=2)
    p.add_argument(
        "--seq-len",
        type=int,
        default=48,
        help="History length (48=baseline paths; 24/72/96 → seq_len_L subdirs)",
    )
    p.add_argument(
        "--horizon",
        type=int,
        default=24,
        help="Forecast horizon (24=baseline; 48 → horizon_48/ subdirs, MA24_D1/D2)",
    )
    p.add_argument("--train-epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument(
        "--stations",
        nargs="+",
        default=None,
        help=(
            "Informer only: station folder names under "
            "data/training/zone_*/informer2020/ (e.g. SAN_MIGUEL). "
            "Passed to Slurm as STATIONS=..."
        ),
    )
    p.add_argument(
        "--partition",
        default=None,
        help="Override Slurm partition (e.g. gpu-small, gpu-fast). Default: script #SBATCH",
    )
    p.add_argument(
        "--force-retrain",
        action="store_true",
        help="Set SKIP_EXISTING=0 so existing results.json are overwritten",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def skip_existing_flag(args: argparse.Namespace) -> str:
    return "0" if args.force_retrain else "1"


def _ssh_common_opts(args: argparse.Namespace) -> list[str]:
    """Timeouts / keepalives. (No ControlMaster: broken on Windows OpenSSH.)"""
    return [
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=20",
        "-o",
        "ConnectionAttempts=3",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
    ]


def ssh_base(args: argparse.Namespace) -> list[str]:
    cmd = ["ssh"]
    if args.key and Path(args.key).exists():
        cmd += ["-i", args.key]
    cmd += _ssh_common_opts(args)
    cmd.append(args.host)
    return cmd


def scp_base(args: argparse.Namespace) -> list[str]:
    cmd = ["scp"]
    if args.key and Path(args.key).exists():
        cmd += ["-i", args.key]
    cmd += _ssh_common_opts(args)
    return cmd


def run(cmd: list[str], *, dry_run: bool = False, check: bool = True) -> int:
    printable = " ".join(cmd)
    print(f">> {printable}")
    if dry_run:
        return 0
    return subprocess.run(cmd, check=check).returncode


def ensure_lf(path: Path) -> Path:
    """Rewrite script with Unix newlines into a temp file next to it."""
    text = path.read_text(encoding="utf-8")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    out = path.with_suffix(path.suffix + ".lf")
    out.write_bytes(text.encode("utf-8"))
    return out


def remote_bash(args: argparse.Namespace, script: str) -> list[str]:
    # Login shells print banners; keep command simple.
    remote = (
        f"cd {args.remote_dir} && "
        f"mkdir -p logs/slurm scripts/hpc && "
        f"{script}"
    )
    return ssh_base(args) + [remote]


def upload_script(args: argparse.Namespace, rel: str) -> None:
    local = ROOT / rel
    if not local.exists():
        raise SystemExit(f"Missing script: {local}")
    lf = ensure_lf(local)
    try:
        dest = f"{args.host}:{args.remote_dir}/{rel.replace(chr(92), '/')}"
        run(scp_base(args) + [str(lf), dest], dry_run=args.dry_run)
    finally:
        lf.unlink(missing_ok=True)


def airformer_data_dir(
    zone: int,
    feature_config: str,
    dataset: str = "ZONE2_PM10",
    seq_len: int = 48,
    horizon: int = 24,
) -> Path:
    return _airformer_data_dir(
        zone, feature_config, dataset, seq_len=seq_len, horizon=horizon
    )


def gat_mf_pm10_data_dir(
    zone: int,
    feature_config: str,
    seq_len: int = 48,
    horizon: int = 24,
) -> Path:
    return gat_informer_mf_pm10graph_data_dir(
        zone, feature_config, seq_len=seq_len, horizon=horizon
    )


def upload_code_sync(args: argparse.Namespace) -> None:
    """Upload experiment/train modules so HPC matches the PC repo."""
    rels = list(MODEL_CODE_SYNC.get(args.model, []))
    rels.append(MODEL_SCRIPTS[args.model])
    # unique preserve order
    seen: set[str] = set()
    ordered = []
    for r in rels:
        if r not in seen:
            seen.add(r)
            ordered.append(r)
    for rel in ordered:
        local = ROOT / rel
        if not local.exists():
            raise SystemExit(f"Missing code to sync: {local}")
        remote_parent = f"{args.remote_dir}/{Path(rel).parent.as_posix()}"
        run(remote_bash(args, f"mkdir -p {remote_parent}"), dry_run=args.dry_run)
        if rel.endswith(".sh"):
            upload_script(args, rel)
        else:
            dest = f"{args.host}:{args.remote_dir}/{Path(rel).as_posix()}"
            print(f"Sync code {rel} -> {dest}")
            run(scp_base(args) + [str(local), dest], dry_run=args.dry_run)


def informer_station_dir(zone: int, station: str, feature_config: str) -> Path:
    return (
        ROOT
        / "data"
        / "training"
        / f"zone_{zone}"
        / "informer2020"
        / station
        / feature_config
    )


def upload_raw_data_many(args: argparse.Namespace, configs: list[str]) -> None:
    """Upload npz packs without re-shipping code (HPC repo already cloned)."""
    if args.model == "xgboost":
        # One tar + one scp (avoids dozens of SSH handshakes that reset the gateway).
        import tarfile
        import tempfile

        local_root = ROOT / "data" / "training" / f"zone_{args.zone}" / "informer2020"
        if not local_root.exists():
            raise SystemExit(f"Missing {local_root}")
        stations = args.stations or [
            d.name for d in sorted(local_root.iterdir()) if d.is_dir()
        ]
        remote_parent = (
            f"{args.remote_dir}/data/training/zone_{args.zone}/informer2020"
        )
        with tempfile.TemporaryDirectory(prefix="xgb_upload_") as tmp:
            tar_path = Path(tmp) / "informer2020_xgb_packs.tgz"
            print(f"Packing {len(stations)} stations × {len(configs)} configs -> {tar_path.name}")
            with tarfile.open(tar_path, "w:gz") as tar:
                for station in stations:
                    for cfg in configs:
                        local = local_root / station / cfg
                        if not (local / "data.csv").exists():
                            print(f"  skip missing {station}/{cfg}")
                            continue
                        tar.add(local, arcname=f"{station}/{cfg}")
            run(remote_bash(args, f"mkdir -p {remote_parent}"), dry_run=args.dry_run)
            remote_tar = f"{args.remote_dir}/informer2020_xgb_packs.tgz"
            print(f"Upload tar -> {args.host}:{remote_tar}")
            run(
                scp_base(args) + [str(tar_path), f"{args.host}:{remote_tar}"],
                dry_run=args.dry_run,
            )
            run(
                remote_bash(
                    args,
                    f"mkdir -p {remote_parent} && "
                    f"tar xzf {remote_tar} -C {remote_parent} && "
                    f"rm -f {remote_tar} && "
                    f"echo unpacked && find {remote_parent} -name data.csv | wc -l",
                ),
                dry_run=args.dry_run,
            )
        return

    if args.model == "informer":
        if not args.stations:
            raise SystemExit(
                "--upload-raw-data for informer requires --stations "
                "(e.g. --stations SAN_MIGUEL). For full packs use --upload-data."
            )
        remote_parent = (
            f"{args.remote_dir}/data/training/zone_{args.zone}/informer2020"
        )
        run(remote_bash(args, f"mkdir -p {remote_parent}"), dry_run=args.dry_run)
        for station in args.stations:
            for cfg in configs:
                local = informer_station_dir(args.zone, station, cfg)
                if not (local / "data.csv").exists():
                    raise SystemExit(
                        f"Missing Informer pack: {local / 'data.csv'}\n"
                        f"Build with:\n"
                        f"  python src/data/export_informer2020_all_stations.py "
                        f"--feature-configs {cfg} --stations \"SAN MIGUEL\""
                    )
                remote_station = f"{remote_parent}/{station}"
                run(
                    remote_bash(args, f"mkdir -p {remote_station}"),
                    dry_run=args.dry_run,
                )
                dest = f"{args.host}:{remote_station}/"
                print(f"Uploading {local} -> {dest}{cfg}/")
                run(scp_base(args) + ["-r", str(local), dest], dry_run=args.dry_run)
        return

    if args.model in ("airformer", "airformer_mse"):
        dataset = "ZONE2_PM10"
        remote_parent = (
            f"{args.remote_dir}/data/training/zone_{args.zone}/airformer/{dataset}"
        )
        run(remote_bash(args, f"mkdir -p {remote_parent}"), dry_run=args.dry_run)
        for cfg in configs:
            local = airformer_data_dir(
                args.zone, cfg, dataset, seq_len=args.seq_len, horizon=args.horizon
            )
            if not (local / "train.npz").exists():
                raise SystemExit(
                    f"Missing AirFormer pack: {local}\n"
                    f"Export with: python src/data/export_airformer_zone2.py "
                    f"--feature-configs {cfg} --seq-len {args.seq_len} "
                    f"--horizon {args.horizon}"
                )
            remote_cfg_parent = apply_layout_tags(
                Path(f"{remote_parent}"),
                seq_len=args.seq_len,
                horizon=args.horizon,
            )
            run(
                remote_bash(args, f"mkdir -p {remote_cfg_parent.as_posix()}"),
                dry_run=args.dry_run,
            )
            dest = f"{args.host}:{remote_cfg_parent.as_posix()}/"
            print(f"Uploading {local} -> {dest}{cfg}/")
            run(scp_base(args) + ["-r", str(local), dest], dry_run=args.dry_run)
        return

    if args.model == "gat_mf_pm10":
        remote_parent = (
            f"{args.remote_dir}/data/training/zone_{args.zone}/gat_informer_mf_pm10graph"
        )
        run(remote_bash(args, f"mkdir -p {remote_parent}"), dry_run=args.dry_run)
        for cfg in configs:
            local = gat_mf_pm10_data_dir(
                args.zone, cfg, seq_len=args.seq_len, horizon=args.horizon
            )
            if not any(local.glob("data*.npz")):
                raise SystemExit(
                    f"Missing PM10-graph pack: {local}\n"
                    f"Build with:\n"
                    f"  python scripts/hpc/export_horizon48_packs.py "
                    f"--feature-configs {cfg}\n"
                    f"  # or for H=24:\n"
                    f"  python src/graphs/build_pm10_graph.py --zone {args.zone}\n"
                    f"  python src/data/clone_gat_pack_alt_graph.py --src-config {cfg} "
                    f"--graph-csv data/graphs/pm10_graph_zone_{args.zone}_top5.csv "
                    f"--tag pm10graph"
                )
            dest = f"{args.host}:{remote_parent}/"
            print(f"Uploading {local} -> {dest}{cfg}/")
            run(scp_base(args) + ["-r", str(local), dest], dry_run=args.dry_run)
        return

    raise SystemExit(
        f"--upload-raw-data is implemented for informer/airformer/airformer_mse/"
        f"gat_mf_pm10, not {args.model}"
    )


def configs_for(args: argparse.Namespace) -> list[str]:
    if args.all_configs:
        return ["F1", "F2", "F3", "F4", "F5"]
    if args.feature_configs:
        return list(args.feature_configs)
    return [args.feature_config]


def sbatch_line(args: argparse.Namespace, rel: str, cfg: str) -> str:
    seq = int(getattr(args, "seq_len", 48))
    horizon = int(getattr(args, "horizon", 24))
    job_name = f"{args.model}_s{seq}_h{horizon}_{cfg.lower()}"
    if len(job_name) > 64:
        job_name = f"{args.model}_{cfg.lower()}"
    max_epochs = 50 if args.preset == "paper" else max(int(args.train_epochs), 1)
    label_len = max(seq // 2, 1)
    export = (
        f"ALL,FEATURE_CONFIG={cfg},"
        f"PRESET={args.preset},ZONE={args.zone},"
        f"SEQ_LEN={seq},LABEL_LEN={label_len},HORIZON={horizon},"
        f"TRAIN_EPOCHS={args.train_epochs},PATIENCE={args.patience},"
        f"MAX_EPOCHS={max_epochs},SKIP_EXISTING={skip_existing_flag(args)}"
    )
    if args.model == "xgboost":
        # Prefer explicit high trial counts; default 30 (not TRAIN_EPOCHS=10).
        n_trials = 30
        if hasattr(args, "train_epochs") and int(args.train_epochs) > 30:
            n_trials = int(args.train_epochs)
        export += f",N_TRIALS={n_trials},SKIP_EXISTING={skip_existing_flag(args)}"
    if args.stations:
        # Comma-separated (Slurm --export cannot carry spaces safely).
        stations = ",".join(args.stations)
        export += f",STATIONS={stations}"
    part = ""
    if getattr(args, "partition", None):
        part = f"--partition={args.partition} "
    return f"sbatch {part}--job-name={job_name} --export={export} {rel}"


def sbatch_xgboost_one(args: argparse.Namespace, rel: str, configs: list[str]) -> str:
    """One CPU job: several F* sequentially, skip stations already done."""
    n_trials = 30
    if hasattr(args, "train_epochs") and int(args.train_epochs) > 30:
        n_trials = int(args.train_epochs)
    cfgs = ":".join(configs)
    seq = int(getattr(args, "seq_len", 48))
    horizon = int(getattr(args, "horizon", 24))
    label_len = max(seq // 2, 1)
    job_name = f"xgboost_s{seq}_h{horizon}_{'_'.join(c.lower() for c in configs)}"
    if len(job_name) > 64:
        job_name = f"xgboost_s{seq}_h{horizon}_resume"
    export = (
        f"ALL,FEATURE_CONFIGS={cfgs},ZONE={args.zone},"
        f"SEQ_LEN={seq},LABEL_LEN={label_len},HORIZON={horizon},"
        f"N_TRIALS={n_trials},SKIP_EXISTING={skip_existing_flag(args)}"
    )
    if args.stations:
        export += f",STATIONS={','.join(args.stations)}"
    return f"sbatch --job-name={job_name} --export={export} {rel}"


def sbatch_airformer_one(args: argparse.Namespace, rel: str, configs: list[str]) -> str:
    """One GPU allocation: several F* in series (one queue wait)."""
    seq = int(getattr(args, "seq_len", 48))
    horizon = int(getattr(args, "horizon", 24))
    cfgs = ":".join(configs)
    max_epochs = 50 if args.preset == "paper" else max(int(args.train_epochs), 1)
    prefix = "airformer_mse" if args.model == "airformer_mse" else "airformer"
    job_name = f"{prefix}_s{seq}_h{horizon}_{'_'.join(c.lower() for c in configs)}"
    if len(job_name) > 64:
        job_name = f"{prefix}_s{seq}_all"
    export = (
        f"ALL,FEATURE_CONFIGS={cfgs},ZONE={args.zone},"
        f"SEQ_LEN={seq},HORIZON={horizon},MAX_EPOCHS={max_epochs},SKIP_EXISTING={skip_existing_flag(args)}"
    )
    part = ""
    if getattr(args, "partition", None):
        part = f"--partition={args.partition} "
    return f"sbatch {part}--job-name={job_name} --export={export} {rel}"


def sbatch_multicfg_gpu(
    args: argparse.Namespace, rel: str, configs: list[str], *, model_tag: str
) -> str:
    """One GPU job looping F* (Informer / GAT), same pattern as AirFormer."""
    seq = int(getattr(args, "seq_len", 48))
    horizon = int(getattr(args, "horizon", 24))
    cfgs = ":".join(configs)
    max_epochs = 50 if args.preset == "paper" else max(int(args.train_epochs), 1)
    label_len = max(seq // 2, 1)
    job_name = f"{model_tag}_s{seq}_h{horizon}_{'_'.join(c.lower() for c in configs)}"
    if len(job_name) > 64:
        job_name = f"{model_tag}_s{seq}_all"
    export = (
        f"ALL,FEATURE_CONFIGS={cfgs},ZONE={args.zone},"
        f"SEQ_LEN={seq},LABEL_LEN={label_len},HORIZON={horizon},"
        f"TRAIN_EPOCHS={args.train_epochs},PATIENCE={args.patience},"
        f"MAX_EPOCHS={max_epochs},PRESET={args.preset},SKIP_EXISTING={skip_existing_flag(args)}"
    )
    if args.stations:
        export += f",STATIONS={','.join(args.stations)}"
    part = ""
    if getattr(args, "partition", None):
        part = f"--partition={args.partition} "
    return f"sbatch {part}--job-name={job_name} --export={export} {rel}"


def submit(args: argparse.Namespace, configs: list[str] | None = None) -> None:
    configs = configs or [args.feature_config]
    upload_code_sync(args)
    rel = slurm_script_for_model(args)
    lines = [f"chmod +x {rel}"]
    if args.model == "xgboost":
        lines.append(sbatch_xgboost_one(args, rel, configs))
    elif args.model in ("airformer", "airformer_mse") and len(configs) > 1:
        lines.append(sbatch_airformer_one(args, rel, configs))
    elif args.model in ("informer", "gat_mf_pm10") and len(configs) > 1:
        tag = "informer" if args.model == "informer" else "gat_mf"
        lines.append(sbatch_multicfg_gpu(args, rel, configs, model_tag=tag))
    else:
        for cfg in configs:
            lines.append(sbatch_line(args, rel, cfg))
    lines.append('squeue -u "$USER"')
    run(remote_bash(args, " && ".join(lines)), dry_run=args.dry_run)


def show_status(args: argparse.Namespace) -> None:
    run(
        remote_bash(
            args,
            'echo "=== squeue ===" && squeue -u "$USER" && '
            'echo "=== last jobs ===" && '
            'sacct -u "$USER" --starttime=now-2days '
            "--format=JobID,JobName%20,State,ExitCode,Elapsed,End -X | tail -20",
        ),
        dry_run=args.dry_run,
        check=False,
    )


PULL_MODELS = ("informer", "airformer", "gat_mf_pm10", "xgboost")


def sync_dedupe_scripts(args: argparse.Namespace) -> None:
    """Ensure HPC has station dedupe helpers before remote cleanup."""
    files = [
        "src/utils/stations.py",
        "src/models/informer2020/metrics.py",
        "scripts/hpc/cleanup_duplicate_stations.py",
    ]
    for rel in files:
        local = ROOT / rel
        if not local.exists():
            continue
        dest = f"{args.host}:{args.remote_dir}/{rel}"
        run(scp_base(args) + [str(local), dest], dry_run=args.dry_run)


def cleanup_duplicate_stations_remote(args: argparse.Namespace, models: list[str]) -> None:
    if not args.dedupe_stations:
        return
    sync_dedupe_scripts(args)
    model_args = " ".join(
        {
            "informer": "informer2020",
            "airformer": "airformer",
            "airformer_mse": "airformer",
            "gat_mf_pm10": "gat_informer_mf_pm10graph",
            "xgboost": "xgboost",
        }[m]
        for m in models
    )
    cmd = (
        f"cd {args.remote_dir} && "
        f"python3 scripts/hpc/cleanup_duplicate_stations.py "
        f"--zone {args.zone} --models {model_args}"
    )
    run(remote_bash(args, cmd), dry_run=args.dry_run, check=False)


def cleanup_duplicate_stations_local(args: argparse.Namespace, model: str) -> None:
    if not args.dedupe_stations:
        return
    if model == "informer":
        root = ROOT / f"results/informer2020/zone_{args.zone}"
    elif model in ("airformer", "airformer_mse"):
        root = ROOT / f"results/airformer/zone_{args.zone}/ZONE2_PM10"
    elif model == "gat_mf_pm10":
        root = ROOT / f"results/gat_informer_mf_pm10graph/zone_{args.zone}"
    elif model == "xgboost":
        root = ROOT / f"results/xgboost/zone_{args.zone}"
    else:
        return
    if root.exists():
        dedupe_station_directories(root, dry_run=args.dry_run)


def pull_results(args: argparse.Namespace, *, model: str | None = None) -> None:
    model = model or args.model
    local_parent = ROOT / "results"
    local_parent.mkdir(parents=True, exist_ok=True)
    if model == "informer":
        remote = (
            f"{args.host}:{args.remote_dir}/results/informer2020/zone_{args.zone}"
        )
        dest = local_parent / "informer2020"
    elif model in ("airformer", "airformer_mse"):
        remote = (
            f"{args.host}:{args.remote_dir}/results/airformer/zone_{args.zone}"
        )
        dest = local_parent / "airformer"
    elif model == "gat_mf_pm10":
        remote = (
            f"{args.host}:{args.remote_dir}/results/gat_informer_mf_pm10graph/zone_{args.zone}"
        )
        dest = local_parent / "gat_informer_mf_pm10graph"
    elif model == "xgboost":
        remote = (
            f"{args.host}:{args.remote_dir}/results/xgboost/zone_{args.zone}"
        )
        dest = local_parent / "xgboost"
    else:
        raise SystemExit(f"Unknown model for --pull-results: {model}")
    dest.mkdir(parents=True, exist_ok=True)
    run(scp_base(args) + ["-r", remote, str(dest)], dry_run=args.dry_run)
    cleanup_duplicate_stations_local(args, model)
    print(f"Pulled into {dest}")


def pull_all_results(args: argparse.Namespace) -> None:
    cleanup_duplicate_stations_remote(args, list(PULL_MODELS))
    for model in PULL_MODELS:
        print(f"\n=== pull {model} ===")
        pull_results(args, model=model)


def pull_slurm_logs(args: argparse.Namespace, pattern: str | None = None) -> None:
    """Download recent slurm out/err for debugging failed jobs."""
    pattern = pattern or {
        "gat_mf_pm10": "gat_mf_pm10graph",
    }.get(args.model, args.model)
    local = ROOT / "logs" / "slurm_pulled"
    local.mkdir(parents=True, exist_ok=True)
    print(f"Listing remote logs matching {pattern}_*")
    run(
        remote_bash(
            args,
            f"cd {args.remote_dir}/logs/slurm && "
            f"ls -1t {pattern}_*.err {pattern}_*.out 2>/dev/null | head -20 || "
            f"ls -1t *.err 2>/dev/null | head -20",
        ),
        dry_run=args.dry_run,
        check=False,
    )
    run(
        remote_bash(
            args,
            f"cd {args.remote_dir}/logs/slurm && "
            f"tar czf /tmp/{pattern}_logs.tgz "
            f"$(ls -1t {pattern}_*.err {pattern}_*.out 2>/dev/null | head -20) "
            f"2>/dev/null || true",
        ),
        dry_run=args.dry_run,
        check=False,
    )
    tgz = local / f"{pattern}_logs.tgz"
    run(
        scp_base(args)
        + [f"{args.host}:/tmp/{pattern}_logs.tgz", str(tgz)],
        dry_run=args.dry_run,
        check=False,
    )
    if tgz.exists() and tgz.stat().st_size > 0:
        import tarfile

        with tarfile.open(tgz, "r:gz") as tar:
            tar.extractall(local)
        print(f"Extracted logs into {local}")
        # Show last err briefly
        errs = sorted(local.glob(f"{pattern}_*.err"), key=lambda p: p.stat().st_mtime)
        if errs:
            print(f"--- tail {errs[-1].name} ---")
            print("\n".join(errs[-1].read_text(encoding="utf-8", errors="replace").splitlines()[-40:]))
    else:
        print("No log archive downloaded (check SSH / remote paths).")


def run_local(args: argparse.Namespace) -> None:
    py = sys.executable
    horizon = int(getattr(args, "horizon", 24))
    h_args = ["--horizon", str(horizon)]
    pred_args = ["--pred-len", str(horizon)]
    if args.model == "informer":
        cmd = [
            py,
            "-u",
            str(ROOT / "src/experiments/run_informer2020_batch.py"),
            "--feature-config",
            args.feature_config,
            "--preset",
            args.preset,
            "--zone",
            str(args.zone),
            "--train-epochs",
            str(args.train_epochs),
            "--patience",
            str(args.patience),
            "--seq-len",
            str(args.seq_len),
            *pred_args,
        ]
        if args.stations:
            cmd += ["--stations", *args.stations]
    elif args.model == "airformer":
        cmd = [
            py,
            "-u",
            str(ROOT / "src/experiments/experiment_airformer.py"),
            "--feature-config",
            args.feature_config,
            "--max-epochs",
            "50",
            "--batch-size",
            "16",
            "--patience",
            str(args.patience),
            "--stochastic-flag",
            "False",
            "--spatial-flag",
            "True",
            "--dartboard",
            "4",
            "--seq-len",
            str(args.seq_len),
            *h_args,
        ]
    elif args.model == "airformer_mse":
        cmd = [
            py,
            "-u",
            str(ROOT / "src/experiments/experiment_airformer_mse.py"),
            "--feature-config",
            args.feature_config,
            "--max-epochs",
            "50",
            "--batch-size",
            "16",
            "--patience",
            str(args.patience),
            "--stochastic-flag",
            "False",
            "--spatial-flag",
            "True",
            "--dartboard",
            "4",
            *h_args,
        ]
    elif args.model == "gat_mf_pm10":
        cmd = [
            py,
            "-u",
            str(ROOT / "src/experiments/experiment_gat_informer_mf.py"),
            "--feature-config",
            args.feature_config,
            "--zone",
            str(args.zone),
            "--max-epochs",
            str(50 if args.preset == "paper" else args.train_epochs),
            "--batch-size",
            "32",
            "--data-dir",
            str(gat_mf_pm10_data_dir(args.zone, args.feature_config, args.seq_len, horizon)),
            "--out-dir",
            str(
                apply_layout_tags(
                    ROOT / "results" / "gat_informer_mf_pm10graph" / f"zone_{args.zone}",
                    seq_len=args.seq_len,
                    horizon=horizon,
                )
                / args.feature_config
            ),
            *h_args,
        ]
    elif args.model == "xgboost":
        n_trials = int(args.train_epochs) if int(args.train_epochs) >= 10 else 30
        cmd = [
            py,
            "-u",
            str(ROOT / "src/experiments/experiment_xgboost.py"),
            "--zone",
            str(args.zone),
            "--feature-configs",
            args.feature_config,
            "--n-trials",
            str(n_trials),
            "--skip-existing",
            "--seq-len",
            str(args.seq_len),
            *pred_args,
        ]
        if args.stations:
            cmd.extend(["--stations", *args.stations])
    else:
        raise SystemExit(f"Unknown model for --local: {args.model}")
    run(cmd, dry_run=args.dry_run)


def main() -> None:
    args = parse_args()
    if shutil.which("ssh") is None and not args.local and not args.dry_run:
        raise SystemExit("OpenSSH `ssh` not found in PATH")

    if args.status:
        show_status(args)
        return
    if args.pull_logs:
        pull_slurm_logs(args)
        return
    if args.pull_all_models:
        pull_all_results(args)
        return
    if args.pull_results:
        cleanup_duplicate_stations_remote(args, [args.model])
        pull_results(args)
        return
    if args.local:
        run_local(args)
        return

    configs = configs_for(args)
    print(
        f"Model={args.model} configs={','.join(configs)} "
        f"seq_len={args.seq_len} horizon={args.horizon} preset={args.preset}"
    )

    if args.upload_raw_data or args.upload_data:
        if args.upload_data and not args.upload_raw_data:
            print(
                "Note: --upload-data now uploads training packs "
                "(same as --upload-raw-data); legacy zip packs were removed."
            )
        upload_raw_data_many(args, configs)

    submit(args, configs)
    print(
        "\nSubmitted. Later: "
        "python scripts/hpc/launch_from_pc.py --status"
    )


if __name__ == "__main__":
    main()
