# GAT-Informer (adoptado: MF + grafo PM10)

Híbrido basado en [ChengqingYu/GAT-Informer](https://github.com/ChengqingYu/GAT-Informer):

- **GAT**: espacialidad **estática** (adyacencia)
- **Informer**: temporalidad
- **Cross-attention**: fusiona ramas

**Headline TFM:** multi-feature (F1–F5) + grafo de similitud **PM10** (top-5).  
Variantes descartadas (GAT-v1, Adj meteo solo, GNN→Informer): `archivo_2026-08-03/`.

## Entrenar (zona 2)

```powershell
# Packs MF (meteo Adj) + clon con grafo PM10
python src/data/export_gat_informer_mf_zone2.py --feature-configs F1 F2 F3 F4 F5
python src/graphs/build_pm10_graph.py --zone 2
python src/data/clone_gat_pack_alt_graph.py --src-config F2 `
  --graph-csv data/graphs/pm10_graph_zone_2_top5.csv --tag pm10graph

python src/experiments/experiment_gat_informer_mf.py --feature-config F2 `
  --data-dir data/training/zone_2/gat_informer_mf_pm10graph/F2 `
  --out-dir results/gat_informer_mf_pm10graph/zone_2/F2 --max-epochs 50

# HPC
python scripts/hpc/launch_from_pc.py --model gat_mf_pm10 --all-configs --upload-raw-data
```

## Rutas

| Qué | Dónde |
|-----|--------|
| Código oficial | `src/models/gat_informer/_upstream/` |
| Train MF | `src/models/gat_informer/zone2_train_mf.py` |
| Experiment CLI | `src/experiments/experiment_gat_informer_mf.py` |
| Datos (headline) | `data/training/zone_2/gat_informer_mf_pm10graph/{F*}/` |
| Resultados | `results/gat_informer_mf_pm10graph/zone_2/{F*}/` |
| Plan | `docs/gat_informer_integration.md` |
