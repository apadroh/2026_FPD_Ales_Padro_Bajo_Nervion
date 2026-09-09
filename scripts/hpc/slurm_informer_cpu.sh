#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=informer_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --mem=32GB
#SBATCH --output=logs/slurm/informer_%j.out
#SBATCH --error=logs/slurm/informer_%j.err

# Informer2020 on CPU — use when gpu-* queues are saturated.
#   FEATURE_CONFIGS=F1:F2:F3:F4:F5 SEQ_LEN=96 sbatch --export=ALL,FEATURE_CONFIGS,SEQ_LEN \
#     scripts/hpc/slurm_informer_cpu.sh

set -euo pipefail
export PATH="/usr/bin:/bin:${PATH:-}"
export CUDA_VISIBLE_DEVICES=""

FEATURE_CONFIGS="${FEATURE_CONFIGS:-${FEATURE_CONFIG:-F5}}"
PRESET="${PRESET:-paper}"
ZONE="${ZONE:-2}"
SEQ_LEN="${SEQ_LEN:-48}"
LABEL_LEN="${LABEL_LEN:-$((SEQ_LEN / 2))}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
PATIENCE="${PATIENCE:-3}"
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

mkdir -p logs/slurm
echo "host=$(hostname)  cuda=$(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo n/a)"
echo "CPU Informer FEATURE_CONFIGS=$FEATURE_CONFIGS SEQ_LEN=$SEQ_LEN PRESET=$PRESET"

EXTRA_ARGS=()
if [[ -n "${STATIONS:-}" ]]; then
  IFS=',' read -r -a STATION_ARR <<< "${STATIONS}"
  EXTRA_ARGS+=(--stations "${STATION_ARR[@]}")
fi

CFG_NORM="${FEATURE_CONFIGS//;/:}"
CFG_NORM="${CFG_NORM//,/:}"
IFS=':' read -r -a CFG_ARR <<< "${CFG_NORM}"

for cfg in "${CFG_ARR[@]}"; do
  cfg_trimmed="${cfg// /}"
  [[ -z "$cfg_trimmed" ]] && continue

  if [[ "$SEQ_LEN" == "48" ]]; then
    SUMMARY="results/informer2020/zone_${ZONE}/batch_summary_${cfg_trimmed}.csv"
  else
    SUMMARY="results/informer2020/zone_${ZONE}/seq_len_${SEQ_LEN}/batch_summary_${cfg_trimmed}.csv"
  fi

  if [[ "$SKIP_EXISTING" == "1" ]] && [[ -f "$SUMMARY" ]]; then
    echo "NOTE ${cfg_trimmed}: summary exists ($SUMMARY) — still run to fill gaps"
  fi

  echo "===== TRAIN Informer ${cfg_trimmed} seq_len=${SEQ_LEN} (CPU) ====="
  python -u src/experiments/run_informer2020_batch.py \
    --feature-config "$cfg_trimmed" \
    --preset "$PRESET" \
    --zone "$ZONE" \
    --seq-len "$SEQ_LEN" \
    --label-len "$LABEL_LEN" \
    --train-epochs "$TRAIN_EPOCHS" \
    --patience "$PATIENCE" \
    "${EXTRA_ARGS[@]}"

  echo "Done ${cfg_trimmed} -> $SUMMARY"
done

echo "All Informer CPU configs finished for SEQ_LEN=${SEQ_LEN}."
