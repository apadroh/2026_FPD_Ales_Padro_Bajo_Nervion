# Notes for TFM evaluators

**Package:** `2026_FPD_Ales_Padro_Bajo_Nervion` — reviewer snapshot (manuscript v02, September 2026)

## Start here

1. **Tables backing Results:** `results/comparison/zone_2/all_models/tables/`
2. **Figures:** `results/comparison/zone_2/all_models/figures/` (regenerate with scripts below)
3. **Protocol:** `configs/bajo_nervion_pm10.yaml`

## Key supplementary analyses

| Topic | Where |
|-------|-------|
| Loss-function check (AirFormer MAE vs MSE) | `results/airformer/zone_2/ZONE2_PM10/F*_mse/` |
| Supplementary horizon H=48 | `results/comparison/zone_2/horizon_48/`, `configs/horizon_48.yaml` |
| Look-back sensitivity L∈{24,48,72,96} | `results/comparison/zone_2/seq_len_sensitivity/` |
| Lead-time RMSE / hourly POD (≥45 µg/m³) | `tables/lead_*`, `scripts/analysis/plot_headline_lead_rmse_hit.py` |
| MA24 ≥45 µg/m³ exceedance (per station) | `tables/exceedance_ma24_thr45_*` |

## Reproduce without GPU (≈10 min)

```bash
python analysis/comparison/plot_skill_with_peak_extras.py
python analysis/comparison/error_by_pm10_amplitude_packs.py
python scripts/analysis/plot_headline_lead_rmse_hit.py
python src/experiments/compare_horizon_24_48.py
```

## Data

Unzip `data/training/zone_2/dataset_zone_2_PM10.csv.zip` before any training.
Full tensor caches and checkpoints are **not** shipped (rebuild from CSV).

**Repository:** https://github.com/apadroh/2026_FPD_Ales_Padro_Bajo_Nervion
