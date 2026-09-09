#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=airformer_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --mem=32GB
#SBATCH --output=logs/slurm/airformer_%j.out
#SBATCH --error=logs/slurm/airformer_%j.err

# AirFormer on CPU (no GPU) — use when gpu-* queues are saturated.
#   FEATURE_CONFIGS=F3IL SEQ_LEN=48 sbatch --export=ALL,FEATURE_CONFIGS,SEQ_LEN \
#     scripts/hpc/slurm_airformer_cpu.sh
# Peak-weighted F3I (separate results folder F3I_pw):
#   FEATURE_CONFIGS=F3I RESULTS_TAG=_pw PEAK_WEIGHT_ALPHA=1 PEAK_ASYM_BETA=0.5 \
#     sbatch --export=ALL,FEATURE_CONFIGS,RESULTS_TAG,PEAK_WEIGHT_ALPHA,PEAK_ASYM_BETA \
#     scripts/hpc/slurm_airformer_cpu.sh

set -euo pipefail
export PATH="/usr/bin:/bin:${PATH:-}"
export CUDA_VISIBLE_DEVICES=""

FEATURE_CONFIGS="${FEATURE_CONFIGS:-${FEATURE_CONFIG:-F1}}"
ZONE="${ZONE:-2}"
SEQ_LEN="${SEQ_LEN:-48}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
PATIENCE="${PATIENCE:-5}"
DARTBOARD="${DARTBOARD:-4}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RESULTS_TAG="${RESULTS_TAG:-}"
PEAK_WEIGHT_ALPHA="${PEAK_WEIGHT_ALPHA:-0}"
PEAK_WEIGHT_THR="${PEAK_WEIGHT_THR:-40}"
PEAK_WEIGHT_THR2="${PEAK_WEIGHT_THR2:-80}"
PEAK_ASYM_BETA="${PEAK_ASYM_BETA:-0}"

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
echo "CPU AirFormer FEATURE_CONFIGS=$FEATURE_CONFIGS SEQ_LEN=$SEQ_LEN RESULTS_TAG=$RESULTS_TAG peak_alpha=$PEAK_WEIGHT_ALPHA asym=$PEAK_ASYM_BETA"

CFG_NORM="${FEATURE_CONFIGS//;/:}"
CFG_NORM="${CFG_NORM//,/:}"
IFS=':' read -r -a CFG_ARR <<< "$CFG_NORM"
for cfg in "${CFG_ARR[@]}"; do
  cfg_trimmed="${cfg// /}"
  [[ -z "$cfg_trimmed" ]] && continue
  out_name="${cfg_trimmed}${RESULTS_TAG}"
  if [[ "$SEQ_LEN" == "48" ]]; then
    OUT_DIR="results/airformer/zone_${ZONE}/ZONE2_PM10/${out_name}"
  else
    OUT_DIR="results/airformer/zone_${ZONE}/ZONE2_PM10/seq_len_${SEQ_LEN}/${out_name}"
  fi
  if [[ "$SKIP_EXISTING" == "1" ]] && [[ -f "${OUT_DIR}/results.json" ]]; then
    echo "SKIP ${out_name}"
    continue
  fi
  echo "===== TRAIN ${cfg_trimmed} -> ${out_name} seq_len=${SEQ_LEN} (CPU) ====="
  EXTRA=()
  if [[ -n "$RESULTS_TAG" ]]; then
    EXTRA+=(--results-tag "$RESULTS_TAG")
  fi
  python -u src/experiments/experiment_airformer.py \
    --feature-config "$cfg_trimmed" \
    --seq-len "$SEQ_LEN" \
    --max-epochs "$MAX_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --patience "$PATIENCE" \
    --stochastic-flag False \
    --spatial-flag True \
    --dartboard "$DARTBOARD" \
    --peak-weight-alpha "$PEAK_WEIGHT_ALPHA" \
    --peak-weight-thr "$PEAK_WEIGHT_THR" \
    --peak-weight-thr2 "$PEAK_WEIGHT_THR2" \
    --peak-asym-beta "$PEAK_ASYM_BETA" \
    "${EXTRA[@]}"
  echo "Done ${out_name} -> ${OUT_DIR}/"
done
echo "All AirFormer CPU configs finished for SEQ_LEN=${SEQ_LEN}."
