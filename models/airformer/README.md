# AirFormer (AAAI-23)

Official [yoshall/AirFormer](https://github.com/yoshall/AirFormer). Multi-station spatiotemporal forecast (CT-MSA + DS-MSA).

## Notebooks

| Notebook | Use |
|----------|-----|
| [`airformer_pm10_zone2.ipynb`](airformer_pm10_zone2.ipynb) | Upstream smoke / notes |

## Paths

| What | Where |
|------|--------|
| Vendored code | `src/models/airformer/_upstream/` |
| Zone-2 packs | `data/training/zone_2/airformer/ZONE2_PM10/{F1,F3,F5}/` |
| Experiment CLI | `src/experiments/experiment_airformer.py` |
| Results | `results/airformer/zone_2/ZONE2_PM10/{F*}/` |

## Forecast design

- At hour `h`, predict `h+1…h+24` for **all 18 nodes** (`horizon=24`).
- **MA24** = mean of those 24 hourly predictions (overall + per station).

## CLI

```bash
# F1 (univariate PM10, input_dim=1)
python src/experiments/experiment_airformer.py --feature-config F1 --max-epochs 50
```

Defaults for ZONE2: `seq_len=48`, `stochastic_flag=False`, `dartboard=4` (trivial N=18 partition).

## vs Informer2020

| | Informer2020 | AirFormer |
|--|--------------|-----------|
| Runs | 1 per station | **1 per feature config** (18 nodes) |
| Data | CSV wide | `train/val/test.npz` + `adj_mx.pkl` |

Integration plan: `docs/airformer_integration.md`.
