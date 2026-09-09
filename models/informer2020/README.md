# Informer2020 (PM10, zone 2)

Official AAAI 2021 Informer. One notebook: export → train → **hourly + MA24** evaluation.

## Forecast design

- At hour `h`, predict `h+1 … h+24` in one forward pass (`pred_len=24`).
- **MA24** = mean of those 24 hourly predictions (not a separate model head).

## Notebooks

| Notebook | Use |
|----------|-----|
| [`informer2020_pm10_zone2.ipynb`](informer2020_pm10_zone2.ipynb) | Local single-station |

## Paths

| What | Where |
|------|--------|
| Definitive CSV | `data/training/zone_2/dataset_zone_2_PM10.csv` |
| Wide station CSV | `data/training/zone_2/informer2020/{STATION}/{F*}/` |
| Checkpoints / metrics | `results/informer2020/zone_2/{STATION}/{F*}/` |
| Model code | `src/models/informer2020/` |
| Experiment CLI | `src/experiments/experiment_informer2020.py` |
| Batch all stations | `src/experiments/run_informer2020_batch.py` |
| Compare F1–F5 | `src/experiments/compare_informer2020_configs.py` |
| Comparison outputs | `results/informer2020/zone_2/comparison/` |

## CLI

```bash
# single station
python src/experiments/experiment_informer2020.py --station BASAURI --feature-config F5 --seq-len 48 --pred-len 24

# all stations F1 (local smoke: --preset fast)
python src/experiments/run_informer2020_batch.py --feature-config F1 --preset paper

# compare feature configs (uses batch_summary_F*.csv; skips failed/missing)
python src/experiments/compare_informer2020_configs.py --zone 2
# when F1–F5 are all complete:
python src/experiments/compare_informer2020_configs.py --zone 2 --require-all

# persistence baselines vs Informer (same test windows)
python src/experiments/evaluate_persistence_baseline.py --zone 2

# multi-seed stability on selected stations (writes .../seed_{N}/)
python src/experiments/run_informer_seed_stability.py --preset fast --seeds 0 1
```

Artifacts after test:

- `test_preds_hourly.npy` / `test_trues_hourly.npy`
- `test_preds_ma24.npy` / `test_trues_ma24.npy`
- `results.json` → `test_metrics_raw.hourly` and `.ma24`

Integration plan: `docs/informer2020_integration.md`.
