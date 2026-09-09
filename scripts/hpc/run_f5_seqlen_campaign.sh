#!/bin/bash
set -euo pipefail
cd "$HOME/MASTER"
mkdir -p data/training/zone_2/airformer/ZONE2_PM10
mkdir -p data/training/zone_2/gat_informer_mf_pm10graph/F5
tar xzf dist/f5_af_seqlen.tgz -C data/training/zone_2/airformer/ZONE2_PM10
tar xzf dist/f5_gat_seqlen.tgz -C data/training/zone_2/gat_informer_mf_pm10graph/F5
echo UNPACK_OK
ls data/training/zone_2/airformer/ZONE2_PM10/seq_len_24/F5/train.npz
ls data/training/zone_2/gat_informer_mf_pm10graph/F5/data24.npz
bash scripts/hpc/archive_f5_seqlen_binary.sh "$HOME/MASTER"
bash scripts/hpc/submit_f5_seqlen_grid.sh
