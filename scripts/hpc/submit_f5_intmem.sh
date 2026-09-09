#!/bin/bash
# Unpack F5 intensity+memory packs and submit CPU trains for all models.
set -euo pipefail
cd "$HOME/MASTER"
mkdir -p logs/slurm data/training/zone_2/informer2020
mkdir -p data/training/zone_2/airformer/ZONE2_PM10/F5
mkdir -p data/training/zone_2/gat_informer_mf_pm10graph/F5

if [[ -f dist/f5_informer_only.tgz ]]; then
  tar xzf dist/f5_informer_only.tgz -C data/training/zone_2/informer2020
  echo "unpacked informer F5"
fi
if [[ -f dist/f5_airformer.tgz ]]; then
  tar xzf dist/f5_airformer.tgz -C data/training/zone_2/airformer/ZONE2_PM10/F5
  echo "unpacked airformer F5"
fi
if [[ -f dist/f5_gat48.tgz ]]; then
  tar xzf dist/f5_gat48.tgz -C data/training/zone_2/gat_informer_mf_pm10graph/F5
  echo "unpacked gat F5"
fi

# sanity: Sah_int in metadata / csv header
python - <<'PY'
import json
from pathlib import Path
m=json.loads(Path('data/training/zone_2/airformer/ZONE2_PM10/F5/metadata.json').read_text())
feats=m.get('features') or m.get('feature_cols') or []
assert 'Sah_int' in feats and 'Sah_int_lag1' in feats and 'Sah' not in feats, feats
print('AF features OK', len(feats))
head=Path('data/training/zone_2/informer2020/ABANTO/F5/data.csv').read_text(encoding='utf-8', errors='replace').splitlines()[0]
assert 'Sah_int' in head and 'Sah_int_lag1' in head
assert ',Sah,' not in ','+head+',' and not head.startswith('Sah,')
print('Inf header OK')
PY

export FEATURE_CONFIGS=F5
export FEATURE_CONFIG=F5
export ZONE=2
export SEQ_LEN=48
export SKIP_EXISTING=0
export PRESET=paper
export TRAIN_EPOCHS=10
export PATIENCE=3
export MAX_EPOCHS=50
export N_TRIALS=30

chmod +x scripts/hpc/slurm_*_cpu.sh scripts/hpc/slurm_xgboost.sh || true

echo "Submitting AF F5 CPU..."
sbatch --job-name=af_f5_intmem --export=ALL,FEATURE_CONFIGS,ZONE,SEQ_LEN,SKIP_EXISTING,MAX_EPOCHS \
  scripts/hpc/slurm_airformer_cpu.sh

echo "Submitting GAT F5 CPU..."
sbatch --job-name=gat_f5_intmem --export=ALL,FEATURE_CONFIGS,ZONE,SEQ_LEN,SKIP_EXISTING,MAX_EPOCHS \
  scripts/hpc/slurm_gat_informer_mf_pm10graph_cpu.sh

echo "Submitting Inf F5 CPU..."
sbatch --job-name=inf_f5_intmem --export=ALL,FEATURE_CONFIGS,ZONE,SEQ_LEN,SKIP_EXISTING,PRESET,TRAIN_EPOCHS,PATIENCE \
  scripts/hpc/slurm_informer_cpu.sh

echo "Submitting XGB F5 CPU..."
sbatch --job-name=xgb_f5_intmem --export=ALL,FEATURE_CONFIGS,ZONE,SEQ_LEN,SKIP_EXISTING,N_TRIALS \
  scripts/hpc/slurm_xgboost.sh

squeue -u "$USER"
