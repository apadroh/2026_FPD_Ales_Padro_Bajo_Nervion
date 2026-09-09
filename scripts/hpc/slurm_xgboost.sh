#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=xgboost
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-12:00:00
#SBATCH --mem=32GB
#SBATCH --output=logs/slurm/xgboost_%j.out
#SBATCH --error=logs/slurm/xgboost_%j.err

# XGBoost baseline — CPU only (tree_method=hist). Do NOT use gpu-fast.
# One job can run several F* sequentially (resume with SKIP_EXISTING=1):
#   FEATURE_CONFIGS=F3,F4,F5 N_TRIALS=30 SEQ_LEN=24 sbatch --export=ALL,FEATURE_CONFIGS,N_TRIALS,SEQ_LEN \
#     scripts/hpc/slurm_xgboost.sh
# Or from PC:
#   python scripts/hpc/launch_from_pc.py --model xgboost --feature-configs F3 F4 F5 --seq-len 24

set -euo pipefail

# Prefer FEATURE_CONFIGS (colon/semicolon-separated; avoid commas — Slurm
# --export uses commas as field separators). Fall back to FEATURE_CONFIG.
FEATURE_CONFIGS="${FEATURE_CONFIGS:-${FEATURE_CONFIG:-F1}}"
ZONE="${ZONE:-2}"
SEQ_LEN="${SEQ_LEN:-48}"
HORIZON="${HORIZON:-24}"
LABEL_LEN="${LABEL_LEN:-$((SEQ_LEN / 2))}"
N_TRIALS="${N_TRIALS:-30}"
MAX_ESTIMATORS="${MAX_ESTIMATORS:-800}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
# Optional: comma-separated station folders, e.g. "ABANTO,SAN_MIGUEL"
# STATIONS="${STATIONS:-}"

module load torch/2.9.0 2>/dev/null || module load torch || true

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
elif [[ -f venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

mkdir -p logs/slurm
echo "host=$(hostname 2>/dev/null || echo unknown)  cwd=$PWD"
echo "FEATURE_CONFIGS=$FEATURE_CONFIGS  ZONE=$ZONE  SEQ_LEN=$SEQ_LEN  HORIZON=$HORIZON  N_TRIALS=$N_TRIALS  STATIONS=${STATIONS:-<all>}"

python - <<'PY'
import importlib.util
import sys
missing = [m for m in ("xgboost", "optuna", "sklearn") if importlib.util.find_spec(m) is None]
if missing:
    print("ERROR: missing packages:", ", ".join(missing), file=sys.stderr)
    print("On HPC venv (via srun): python -m pip install xgboost optuna scikit-learn", file=sys.stderr)
    sys.exit(1)
print("deps ok: xgboost/optuna/sklearn")
PY

DATA_ROOT="data/training/zone_${ZONE}/informer2020"
if [[ ! -d "$DATA_ROOT" ]]; then
  echo "ERROR: missing $DATA_ROOT (upload Informer station CSVs with --upload-raw-data)"
  exit 1
fi

EXTRA_ARGS=()
if [[ -n "${STATIONS:-}" ]]; then
  IFS=',' read -r -a STATION_ARR <<< "${STATIONS}"
  EXTRA_ARGS+=(--stations "${STATION_ARR[@]}")
fi
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  EXTRA_ARGS+=(--skip-existing)
fi

# Expand configs: accept "F3:F4:F5", "F3;F4;F5", or legacy "F3,F4,F5"
CFG_NORM="${FEATURE_CONFIGS//;/:}"
CFG_NORM="${CFG_NORM//,/:}"
IFS=':' read -r -a CFG_ARR <<< "${CFG_NORM}"
CFG_ARGS=()
for cfg in "${CFG_ARR[@]}"; do
  cfg_trimmed="${cfg// /}"
  [[ -n "$cfg_trimmed" ]] && CFG_ARGS+=("$cfg_trimmed")
done

python -u src/experiments/experiment_xgboost.py \
  --zone "$ZONE" \
  --feature-configs "${CFG_ARGS[@]}" \
  --seq-len "$SEQ_LEN" \
  --label-len "$LABEL_LEN" \
  --pred-len "$HORIZON" \
  --n-trials "$N_TRIALS" \
  --max-estimators "$MAX_ESTIMATORS" \
  "${EXTRA_ARGS[@]}"

if [[ "$SEQ_LEN" == "48" && "$HORIZON" == "24" ]]; then
  echo "Done. Summaries under results/xgboost/zone_${ZONE}/batch_summary_F*.csv"
elif [[ "$HORIZON" != "24" ]]; then
  echo "Done. Summaries under results/xgboost/zone_${ZONE}/horizon_${HORIZON}/batch_summary_F*.csv"
else
  echo "Done. Summaries under results/xgboost/zone_${ZONE}/seq_len_${SEQ_LEN}/batch_summary_F*.csv"
fi
