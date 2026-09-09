#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=af_mse_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --mem=32GB
#SBATCH --output=logs/slurm/airformer_mse_%j.out
#SBATCH --error=logs/slurm/airformer_mse_%j.err

# AirFormer MSE ablation on CPU (no GPU) — journal follow-up; does not overwrite MAE runs.
#   FEATURE_CONFIGS=F1:F2:F3:F4:F5 sbatch --export=ALL,FEATURE_CONFIGS \
#     scripts/hpc/slurm_airformer_mse_cpu.sh
# From PC:
#   python scripts/hpc/launch_from_pc.py --model airformer_mse --all-configs --partition cpu

set -euo pipefail
export PATH="/usr/bin:/bin:${PATH:-}"
export CUDA_VISIBLE_DEVICES=""

FEATURE_CONFIGS="${FEATURE_CONFIGS:-${FEATURE_CONFIG:-F5}}"
ZONE="${ZONE:-2}"
SEQ_LEN="${SEQ_LEN:-48}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
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
echo "host=${HOSTNAME:-unknown} cuda=$(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo n/a)"
echo "CPU AirFormer MSE FEATURE_CONFIGS=$FEATURE_CONFIGS SEQ_LEN=$SEQ_LEN tag=_mse"

CFG_NORM="${FEATURE_CONFIGS//;/:}"
CFG_NORM="${CFG_NORM//,/:}"
IFS=':' read -r -a CFG_ARR <<< "$CFG_NORM"
for cfg in "${CFG_ARR[@]}"; do
  cfg_trimmed="${cfg// /}"
  [[ -z "$cfg_trimmed" ]] && continue
  out_name="${cfg_trimmed}_mse"
  if [[ "$SEQ_LEN" == "48" ]]; then
    OUT_DIR="results/airformer/zone_${ZONE}/ZONE2_PM10/${out_name}"
  else
    OUT_DIR="results/airformer/zone_${ZONE}/ZONE2_PM10/seq_len_${SEQ_LEN}/${out_name}"
  fi
  if [[ "$SKIP_EXISTING" == "1" ]] && [[ -f "${OUT_DIR}/results.json" ]]; then
    echo "SKIP ${out_name}"
    continue
  fi
  echo "===== TRAIN ${cfg_trimmed} -> ${out_name} seq_len=${SEQ_LEN} loss=MSE (CPU) ====="
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
  echo "Done ${out_name} -> ${OUT_DIR}/"
done
echo "All AirFormer MSE CPU configs finished for SEQ_LEN=${SEQ_LEN}."
