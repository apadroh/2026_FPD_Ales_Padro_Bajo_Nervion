#!/bin/bash
#SBATCH --partition=gpu-fast
#SBATCH --job-name=gat_mf_pm10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-12:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm/gat_mf_pm10graph_%j.out
#SBATCH --error=logs/slurm/gat_mf_pm10graph_%j.err

# GAT-MF + PM10 Adj: one GPU, several F* in series.
#   FEATURE_CONFIGS=F1:F2:F3:F4:F5 SEQ_LEN=24 sbatch --export=ALL,FEATURE_CONFIGS,SEQ_LEN \
#     scripts/hpc/slurm_gat_informer_mf_pm10graph.sh

set -euo pipefail

export PATH="/usr/bin:/bin:${PATH:-}"

FEATURE_CONFIGS="${FEATURE_CONFIGS:-${FEATURE_CONFIG:-F1}}"
ZONE="${ZONE:-2}"
SEQ_LEN="${SEQ_LEN:-48}"
HORIZON="${HORIZON:-24}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

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
echo "host=${HOSTNAME:-unknown}  cuda=$(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo n/a)"
echo "cwd=$PWD  FEATURE_CONFIGS=$FEATURE_CONFIGS  SEQ_LEN=$SEQ_LEN  HORIZON=$HORIZON  SKIP_EXISTING=$SKIP_EXISTING"

CFG_NORM="${FEATURE_CONFIGS//;/:}"
CFG_NORM="${CFG_NORM//,/:}"
IFS=':' read -r -a CFG_ARR <<< "${CFG_NORM}"

for cfg in "${CFG_ARR[@]}"; do
  cfg_trimmed="${cfg// /}"
  [[ -z "$cfg_trimmed" ]] && continue

  DATA_DIR="data/training/zone_${ZONE}/gat_informer_mf_pm10graph"
  [[ "$SEQ_LEN" != "48" ]] && DATA_DIR="${DATA_DIR}/seq_len_${SEQ_LEN}"
  [[ "$HORIZON" != "24" ]] && DATA_DIR="${DATA_DIR}/horizon_${HORIZON}"
  DATA_DIR="${DATA_DIR}/${cfg_trimmed}"

  OUT_DIR="results/gat_informer_mf_pm10graph/zone_${ZONE}"
  [[ "$SEQ_LEN" != "48" ]] && OUT_DIR="${OUT_DIR}/seq_len_${SEQ_LEN}"
  [[ "$HORIZON" != "24" ]] && OUT_DIR="${OUT_DIR}/horizon_${HORIZON}"
  OUT_DIR="${OUT_DIR}/${cfg_trimmed}"

  if [[ "$SKIP_EXISTING" == "1" ]] && [[ -f "${OUT_DIR}/results.json" ]]; then
    echo "SKIP ${cfg_trimmed} (exists: ${OUT_DIR}/results.json)"
    continue
  fi

  if [[ ! -f "$DATA_DIR/data${SEQ_LEN}.npz" ]]; then
    echo "ERROR: missing pack $DATA_DIR/data${SEQ_LEN}.npz (export + upload)"
    exit 2
  fi

  echo "===== TRAIN GAT-MF PM10 ${cfg_trimmed} seq_len=${SEQ_LEN} horizon=${HORIZON} ====="
  python -u src/experiments/experiment_gat_informer_mf.py \
    --feature-config "$cfg_trimmed" \
    --zone "$ZONE" \
    --seq-len "$SEQ_LEN" \
    --horizon "$HORIZON" \
    --max-epochs "$MAX_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --data-dir "$DATA_DIR" \
    --out-dir "$OUT_DIR"

  echo "Done ${cfg_trimmed} -> $OUT_DIR/"
done

echo "All GAT-MF PM10 configs finished for SEQ_LEN=${SEQ_LEN}."
