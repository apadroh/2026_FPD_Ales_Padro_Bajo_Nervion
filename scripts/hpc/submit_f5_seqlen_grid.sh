#!/bin/bash
# Submit F5 intensity+memory trains for SEQ_LEN in {24,72,96} (all models).
# L=48 is handled by submit_f5_intmem.sh / jobs 223916-223919.
set -euo pipefail
cd "$HOME/MASTER"
mkdir -p logs/slurm dist

# Expect packs already under:
#   data/training/zone_2/airformer/ZONE2_PM10/seq_len_{L}/F5
#   data/training/zone_2/gat_informer_mf_pm10graph/F5/data{L}.npz
#   data/training/zone_2/informer2020/*/F5/data.csv  (shared across L)

python - <<'PY'
from pathlib import Path
root = Path('.')
ok = True
for L in (24, 72, 96):
    af = root / f'data/training/zone_2/airformer/ZONE2_PM10/seq_len_{L}/F5/train.npz'
    gat = root / f'data/training/zone_2/gat_informer_mf_pm10graph/F5/data{L}.npz'
    print('AF', L, af.exists(), af)
    print('GAT', L, gat.exists(), gat)
    if not af.exists() or not gat.exists():
        ok = False
inf = list((root / 'data/training/zone_2/informer2020').glob('*/F5/data.csv'))
print('Inf F5 packs', len(inf))
if len(inf) < 18:
    ok = False
if not ok:
    raise SystemExit('Missing packs — unpack/export first')
print('packs OK')
PY

export FEATURE_CONFIGS=F5
export FEATURE_CONFIG=F5
export ZONE=2
export SKIP_EXISTING=0
export PRESET=paper
export TRAIN_EPOCHS=10
export PATIENCE=3
export MAX_EPOCHS=50
export N_TRIALS=30

chmod +x scripts/hpc/slurm_*_cpu.sh scripts/hpc/slurm_xgboost.sh || true

for L in 24 72 96; do
  export SEQ_LEN=$L
  export LABEL_LEN=$((L / 2))
  echo "===== SEQ_LEN=$L ====="
  sbatch --job-name=af_f5_s${L} --export=ALL,FEATURE_CONFIGS,ZONE,SEQ_LEN,SKIP_EXISTING,MAX_EPOCHS \
    scripts/hpc/slurm_airformer_cpu.sh
  sbatch --job-name=gat_f5_s${L} --export=ALL,FEATURE_CONFIGS,ZONE,SEQ_LEN,SKIP_EXISTING,MAX_EPOCHS \
    scripts/hpc/slurm_gat_informer_mf_pm10graph_cpu.sh
  sbatch --job-name=inf_f5_s${L} --export=ALL,FEATURE_CONFIGS,ZONE,SEQ_LEN,LABEL_LEN,SKIP_EXISTING,PRESET,TRAIN_EPOCHS,PATIENCE \
    scripts/hpc/slurm_informer_cpu.sh
  sbatch --job-name=xgb_f5_s${L} --export=ALL,FEATURE_CONFIGS,ZONE,SEQ_LEN,LABEL_LEN,SKIP_EXISTING,N_TRIALS \
    scripts/hpc/slurm_xgboost.sh
done

squeue -u "$USER"
