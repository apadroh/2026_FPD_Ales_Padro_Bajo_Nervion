# KATEA HPC (Tecnalia)

Login: `ales.padro@hpc.tri.lan` (from Tecnalia network/VPN).  
**Do not train on the login node** — use `sbatch` / `srun`.

Docs: [Getting Access](https://katea.digital.tecnalia.dev/docs/hpc/getting-access/), [SSH](https://katea.digital.tecnalia.dev/docs/hpc/ssh-basics/).

## Launch from Cursor (recommended)

From the repo on your PC (venv activated, VPN/Tecnalia network):

| Action | Cursor | Or terminal |
|--------|--------|-------------|
| Upload data + submit | **Terminal → Run Task… → `HPC: Launch train (upload data + sbatch)`** | `python scripts/hpc/launch_from_pc.py --model informer --feature-config F3 --upload-raw-data` |
| Submit only (data already on HPC) | `HPC: Launch train (sbatch only)` | omit `--upload-raw-data` |
| Queue / recent jobs | `HPC: Job status` | `--status` |
| Download results | `HPC: Pull results` | `--pull-results --model informer` |
| Local CPU batch | `Local: Informer batch (CPU)` | `--local --preset fast` |

The launcher uploads the Slurm script with **Unix LF** line endings (avoids the `\r\n` sbatch error on Windows).

```powershell
cd C:\Users\105116\MASTER
.\venv\Scripts\Activate.ps1

# Informer / AirFormer / GAT (PM10 graph) / XGBoost
python scripts/hpc/launch_from_pc.py --model informer --feature-config F3 --upload-raw-data
python scripts/hpc/launch_from_pc.py --model airformer --feature-config F1 --upload-raw-data
python scripts/hpc/launch_from_pc.py --model gat_mf_pm10 --all-configs --upload-raw-data
python scripts/hpc/launch_from_pc.py --model xgboost --all-configs --upload-raw-data

# H=48 h campaign (results under horizon_48/, MA24_D1/D2 metrics)
python scripts/hpc/export_horizon48_packs.py
python scripts/hpc/launch_from_pc.py --model airformer --all-configs --horizon 48 --upload-raw-data
python scripts/hpc/launch_from_pc.py --model xgboost --all-configs --horizon 48

python scripts/hpc/launch_from_pc.py --status
```

**AirFormer data sizes (approx., zone-2 PM10):** F1 ~7 MB total; F2 ~ tens of MB; F3/F5 `train.npz` alone is ~360–480 MB. Prefer `--upload-raw-data` for F3–F5.

**Headline GAT:** `gat_mf_pm10` → packs bajo `gat_informer_mf_pm10graph/`. Variantes antiguas: `archivo_2026-08-03/`.

Env overrides: `HPC_HOST`, `HPC_KEY`, `HPC_REMOTE_DIR` (defaults: `ales.padro@hpc.tri.lan`, `~/.ssh/id_rsa`, `~/MASTER`).

## One-time setup (on login node)

```bash
cd ~
mkdir -p MASTER && cd MASTER
git clone https://github.com/apadroh/MASTER.git .
module avail torch
module load torch/2.9.0   # or listed module / use .venv with pip torch
python -m venv .venv
source .venv/bin/activate
pip install -U pip pandas numpy matplotlib scipy
# + torch with CUDA if not provided by the module
mkdir -p logs/slurm
```

**AirFormer note:** the upstream trainer needs `scipy`. If jobs fail with `No module named 'scipy'`:

```bash
source ~/MASTER/.venv/bin/activate
pip install scipy
```

Also sync the local experiment CLI (F1–F5) before relaunching F2/F4 — those choices are not yet on `origin/main`:

```powershell
scp src/experiments/experiment_airformer.py ales.padro@hpc.tri.lan:~/MASTER/src/experiments/
scp src/models/airformer/_upstream/src/models/airformer.py ales.padro@hpc.tri.lan:~/MASTER/src/models/airformer/_upstream/src/models/
scp scripts/hpc/slurm_airformer.sh ales.padro@hpc.tri.lan:~/MASTER/scripts/hpc/
# then (LF-safe via launcher):
python scripts/hpc/launch_from_pc.py --model airformer --all-configs
```

## Slurm scripts (on HPC)

| Script | Role |
|--------|------|
| `scripts/hpc/slurm_informer.sh` | Parametric Informer (`FEATURE_CONFIG=F1..F5`) |
| `scripts/hpc/slurm_airformer.sh` | Parametric AirFormer (`FEATURE_CONFIG=F1..F5`) |
| `scripts/hpc/slurm_gat_informer_mf_pm10graph.sh` | GAT-Informer adoptado (MF + grafo PM10) |
| `scripts/hpc/slurm_xgboost.sh` | XGBoost baseline |

```bash
FEATURE_CONFIG=F3 sbatch --export=ALL,FEATURE_CONFIG scripts/hpc/slurm_informer.sh
squeue -u "$USER"
```

If `gpu-fast` is rejected, ask in TEAMS#KATEA for the correct `--partition`.

Informer **skips** stations that already have `results.json` (safe to resume).

## Resource requests (vs Erik’s script)

Erik’s template (`gpu-fast`, `gres=gpu:1`, `module load torch`, venv) **is** what we use.
His **sizes** were for **vLLM + Qwen** (24 CPUs, 40 GB, array) — not for Informer/AirFormer.

| | Erik (vLLM) | Ours (current) |
|--|-------------|----------------|
| Partition / GPU | `gpu-fast` + 1 GPU | same |
| CPUs | 24 | **4** (`num_workers=0`) |
| RAM | 40 GB | **16 GB** |
| Time | 4 h | Informer 6 h / AirFormer 4 h |

Jobs already running keep the old `#SBATCH` until you re-submit with the updated scripts.

After a finished job, tighten further with real usage:

```bash
sacct -j JOBID --format=JobID,JobName%16,Elapsed,MaxRSS,ReqMem,State,ExitCode -X
```

Rule of thumb: set `--mem` ≈ 1.5× `MaxRSS`, `--time` ≈ 1.5–2× `Elapsed`.
If OOM / timeout, bump only that knob (e.g. AirFormer F5 → `--mem=24GB`).
