# Data (public)

Air-quality and meteorology observations for Basque Country monitoring
stations are public (Euskadi open data / OpenData Euskadi). They may be
redistributed for research reproduction.

## What is shipped

**Definitive TFM table** for the Bajo Nervión study area (recommended entry point):

- `training/zone_2/dataset_zone_2_PM10.csv` (~146 MB uncompressed)
- `training/zone_2/dataset_zone_2_PM10.csv.zip` (~20 MB) — what git should track
- `training/zone_2/prepare_training_manifest_PM10.json` — station list / prep metadata

Unzip before training:

```bash
cd data/training/zone_2
python -c "import zipfile; zipfile.ZipFile('dataset_zone_2_PM10.csv.zip').extractall('.')"
```

## What is not shipped (and why)

| Asset | Reason |
|-------|--------|
| Full `data/raw/` dump (~500+ MB) | Too large for a light GitHub clone; identical public sources |
| Model tensors under `data/training/zone_2/<model>/` | Tens of GB; rebuild from the definitive CSV |

To rebuild tensors from the definitive CSV, use the export scripts under
`src/data/` (Informer / AirFormer / GAT-Informer / XGBoost feature packs).

Dartboard assignment for AirFormer DS-MSA:

- `graphs/zone2_18_dartboard/assignment.npy`
