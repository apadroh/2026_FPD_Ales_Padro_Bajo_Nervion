#!/bin/bash
# Archive old binary-Sah F5 results under seq_len_* (keep L=48 archive separate).
set -euo pipefail
cd "${1:-$HOME/MASTER}"
mkdir -p results/_archive_F5_Sah_binary
stamp=$(date +%Y%m%d_%H%M%S)

archive_dir() {
  local src="$1"
  local name="$2"
  if [[ -d "$src" ]]; then
    local dest="results/_archive_F5_Sah_binary/${name}_${stamp}"
    rm -rf "$dest"
    mv "$src" "$dest"
    echo "archived $src -> $dest"
  else
    echo "skip missing $src"
  fi
}

for L in 24 72 96; do
  archive_dir "results/airformer/zone_2/ZONE2_PM10/seq_len_${L}/F5" "airformer_s${L}_F5"
  archive_dir "results/gat_informer_mf_pm10graph/zone_2/seq_len_${L}/F5" "gat_s${L}_F5"
  shopt -s nullglob
  for st in results/informer2020/zone_2/seq_len_${L}/*/F5; do
    base=$(basename "$(dirname "$st")")
    archive_dir "$st" "informer_s${L}_${base}_F5"
  done
  for st in results/xgboost/zone_2/seq_len_${L}/*/F5; do
    base=$(basename "$(dirname "$st")")
    archive_dir "$st" "xgboost_s${L}_${base}_F5"
  done
done

echo "done archive seq_len F5"
