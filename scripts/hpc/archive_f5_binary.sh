#!/bin/bash
# Archive previous F5 results (binary Sah) before intensity+memory retrain.
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

archive_dir results/airformer/zone_2/ZONE2_PM10/F5 airformer_F5
archive_dir results/gat_informer_mf_pm10graph/zone_2/F5 gat_F5

shopt -s nullglob
for st in results/informer2020/zone_2/*/F5; do
  base=$(basename "$(dirname "$st")")
  archive_dir "$st" "informer_${base}_F5"
done
for st in results/xgboost/zone_2/*/F5; do
  base=$(basename "$(dirname "$st")")
  archive_dir "$st" "xgboost_${base}_F5"
done

echo "archive count: $(ls results/_archive_F5_Sah_binary | wc -l)"
squeue -u "$USER" || true
