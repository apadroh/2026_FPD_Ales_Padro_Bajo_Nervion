# Origen del código

- Repositorio: https://github.com/ChengqingYu/GAT-Informer
- Paper: *Attention mechanism is useful in spatio-temporal wind speed prediction: Evidence from China* (Applied Soft Computing, 2023)
- Demo pública sobre PEMS04 (el código comercial original no está abierto)

## Idea del híbrido

| Bloque | Rol |
|--------|-----|
| **GAT** (`GraphAttentionLayer`) | Componente **espacial estática** (grafo de estaciones fijo) |
| **Informer** (`Informer` / `InformerStack`) | Componente **temporal** (ProbSparse + decoder) |
| **Cross-attention** (`cross_att`) | Fusión de salidas GAT + Informer |
| **RevIN** | Normalización reversible por serie |

## Citar

```bibtex
@article{yu2023attention,
  title={Attention mechanism is useful in spatio-temporal wind speed prediction: Evidence from China},
  author={Yu, Chengqing and Yan, Guangxi and Yu, Chengming and Mi, Xiwei},
  journal={Applied Soft Computing},
  volume={148},
  pages={110864},
  year={2023},
  publisher={Elsevier}
}
```
