#!/bin/bash
#SBATCH --partition=gpu-fast
#SBATCH --job-name=af_mse
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm/airformer_mse_%j.out
#SBATCH --error=logs/slurm/airformer_mse_%j.err

# AirFormer MSE ablation (journal follow-up; does NOT overwrite official MAE runs).
#   FEATURE_CONFIGS=F1:F2:F3:F4:F5 sbatch --export=ALL,FEATURE_CONFIGS scripts/hpc/slurm_airformer_mse.sh
# From PC:
#   python scripts/hpc/launch_from_pc.py --model airformer_mse --all-configs --upload-raw-data

set -euo pipefail

export PATH="/usr/bin:/bin:${PATH:-}"

FEATURE_CONFIGS="${FEATURE_CONFIGS:-${FEATURE_CONFIG:-F5}}"
ZONE="${ZONE:-2}"
SEQ_LEN="${SEQ_LEN:-48}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PATIENCE="${PATIENCE:-5}"
DARTBOARD="${DARTBOARD:-4}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

module load torch/2.9.0 2>/dev/null || module load torch || true

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
elif [[ -f venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

python -c "import scipy" 2>/dev/null || pip install -q "scipy>=1.10"

mkdir -p logs/slurm
echo "host=${HOSTNAME:-unknown}  cuda=$(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo n/a)"
echo "cwd=$PWD  FEATURE_CONFIGS=$FEATURE_CONFIGS  SEQ_LEN=$SEQ_LEN  loss=MSE  tag=_mse  SKIP_EXISTING=$SKIP_EXISTING"

CFG_NORM="${FEATURE_CONFIGS//;/:}"
CFG_NORM="${CFG_NORM//,/:}"
IFS=':' read -r -a CFG_ARR <<< "${CFG_NORM}"

for cfg in "${CFG_ARR[@]}"; do
  cfg_trimmed="${cfg// /}"
  [[ -z "$cfg_trimmed" ]] && continue

  if [[ "$SEQ_LEN" == "48" ]]; then
    OUT_DIR="results/airformer/zone_${ZONE}/ZONE2_PM10/${cfg_trimmed}_mse"
  else
    OUT_DIR="results/airformer/zone_${ZONE}/ZONE2_PM10/seq_len_${SEQ_LEN}/${cfg_trimmed}_mse"
  fi

  if [[ "$SKIP_EXISTING" == "1" ]] && [[ -f "${OUT_DIR}/results.json" ]]; then
    echo "SKIP ${cfg_trimmed} (exists: ${OUT_DIR}/results.json)"
    continue
  fi

  echo "===== TRAIN ${cfg_trimmed} seq_len=${SEQ_LEN} loss=MSE ====="
  python -u src/experiments/experiment_airformer_mse.py \
    --zone "$ZONE" \
    --feature-config "$cfg_trimmed" \
    --seq-len "$SEQ_LEN" \
    --max-epochs "$MAX_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --patience "$PATIENCE" \
    --stochastic-flag False \
    --spatial-flag True \
    --dartboard "$DARTBOARD"
  echo "Done ${cfg_trimmed} -> ${OUT_DIR}/"
done

echo "All AirFormer MSE configs finished for SEQ_LEN=${SEQ_LEN}."
