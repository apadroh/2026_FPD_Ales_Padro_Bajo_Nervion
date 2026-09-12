# 2026_FPD_Ales_Padro_Bajo_Nervion

PM₁₀ forecasting in **Bajo Nervión** — MSc Final Project reproduction package (Alejandro Padro).

Minimal package to reproduce the thesis experiments:

- Informer, GAT-Informer, AirFormer, XGBoost
- Feature packs F1–F5 (+ peak extras F3IL / F5I)
- Look-back sensitivity \(L \in \{24,48,72,96\}\) (official \(L=48\) from ACF)
- Supplementary horizon H=48 and AirFormer MSE loss sensitivity
- Lead-time error and MA24 ≥45 µg/m³ exceedance diagnostics
- Equal station-mean MA24 ranking, skill vs persistence, error by PM₁₀ amplitude

This is a **reviewer-facing** extract of the full research workspace (not the
entire lab dump). See **`REVIEWER.md`** for a one-page evaluator guide.

**Repository:** https://github.com/apadroh/2026_FPD_Ales_Padro_Bajo_Nervion

## Quick start

```bash
python -m venv venv
# Windows: .\venv\Scripts\Activate.ps1
source venv/bin/activate
pip install -r requirements.txt

cd data/training/zone_2
python -c "import zipfile; zipfile.ZipFile('dataset_zone_2_PM10.csv.zip').extractall('.')"
cd ../../..
```

## What is included

| Path | Role |
|------|------|
| `data/training/zone_2/dataset_zone_2_PM10.csv.zip` | Definitive public dataset (~20 MB) |
| `analysis/eda` | EDA scripts / notebooks |
| `analysis/feature_selection/zone_2/PM10` | ACF, lag correlations, F1–F5 readiness |
| `analysis/acf/zone_2/PM10` | Shared ACF / `seq_len` recommendation |
| `configs/bajo_nervion_pm10.yaml` | Official protocol |
| `configs/seq_len_{24,48,72,96}.yaml` | Look-back sensitivity |
| `configs/horizon_48.yaml` | Supplementary H=48 protocol |
| `src/` | Training, models, comparison metrics |
| `analysis/comparison/` | Paper figures |
| `results/comparison/zone_2/**/tables` | Metric tables (incl. lead-time & exceedance) |
| `analysis/comparison/` + `results/comparison/zone_2/**/figures` | Paper figures (regenerable) |
| `results/comparison/zone_2/horizon_48/` | H=24 vs H=48 comparison tables |
| `results/airformer/zone_2/ZONE2_PM10/F*_mse/` | MSE loss sensitivity (AirFormer) |

**Not included:** full raw dump (~500 MB) — public and rebuildable; see `data/README.md`.
Model tensor caches (tens of GB) are also omitted.

## Reproduction levels

1. **Tables & figures (CPU, minutes)** — regenerate plots from shipped CSVs:
   ```bash
   python analysis/comparison/plot_skill_with_peak_extras.py
   python analysis/comparison/error_by_pm10_amplitude_packs.py
   python scripts/analysis/plot_headline_lead_rmse_hit.py
   python src/experiments/compare_horizon_24_48.py
   ```
2. **EDA / ACF** — run `analysis/eda/eda_zone_contaminant.py` (PM10, Bajo Nervión)
   and inspect `analysis/feature_selection/zone_2/PM10/`.
3. **Full retrain (GPU, hours–days)** — prepare model tensors from the definitive
   CSV, then train with `seq_len` overrides from `configs/seq_len_*.yaml`.
   See model READMEs under `models/*/README.md` and `scripts/hpc/`.

## Protocol (summary)

- Bajo Nervión, 18 PM₁₀ stations; official horizon H=24 h (+ supplementary H=48)
- Official look-back L=48 h (ACF); sensitivity L∈{24,48,72,96}
- AirFormer trained with official MAE; MSE variant reported as sensitivity check
- Chronological 70/10/20 split
- Primary metric: equal-station-mean MA24 RMSE

## Upstream code

- AirFormer: https://github.com/yoshall/AirFormer
- GAT-Informer: https://github.com/ChengqingYu/GAT-Informer
- Informer: Zhou et al. (AAAI 2021)

## License / data

Observational data are from public Basque air-quality / meteorology sources.
Please cite the thesis and the upstream model papers when reusing this package.
