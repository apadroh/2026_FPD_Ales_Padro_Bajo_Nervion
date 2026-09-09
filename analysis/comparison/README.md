# Comparación multi-modelo (zona 2 · PM10)

CLI principal:

```powershell
python src/experiments/compare_all_models.py --zone 2
python analysis/comparison/spatial_vs_temporal_zone2.py --zone 2
python src/experiments/lead_time_metrics.py --zone 2
python src/experiments/station_weight_sensitivity.py --zone 2
```

Modelos headline: **Informer2020**, **AirFormer**, **GAT-Informer (MF + grafo PM10)**, **XGBoost**, persistencia.

Salidas: `results/comparison/zone_2/all_models/`.

Figuras de grafo GAT: `plot_gat_graph_zone2.py`.

Atención espacial AirFormer vs GAT (slider hora del día):

```powershell
.\venv\Scripts\python.exe analysis/comparison/airformer_spatial_attention_by_hour.py
.\venv\Scripts\python.exe analysis/comparison/airformer_spatial_map_by_hour.py
# HTML heatmaps: results/comparison/zone_2/spatial_attention/spatial_attention_interactive.html
# HTML mapa Bajo Nervión: .../spatial_map_interactive.html
```

> Notebook / variantes antiguas (GAT-v1, GNN→Informer, Informer-vs-AirFormer solo): ver `archivo_2026-08-03/`.
