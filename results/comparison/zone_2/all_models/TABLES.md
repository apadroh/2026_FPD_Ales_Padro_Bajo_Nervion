# Comparison tables — zone 2 · PM10

Headline models: Informer2020, AirFormer, GAT-Informer (PM10 graph).
Primary metric: **MA24 RMSE** (lower better). Also hourly MAE/RMSE.

## Overall ranking (best F + baselines)

| feature_config   | description                  | model            | model_label        |   n_stations |   hourly_mae |   hourly_rmse |   ma24_mae |   ma24_rmse |
|:-----------------|:-----------------------------|:-----------------|:-------------------|-------------:|-------------:|--------------:|-----------:|------------:|
| F3               | F2 + meteorology             | airformer        | AirFormer          |           18 |        5.848 |         9.127 |      3.890 |       5.704 |
| F1               | Autoregressive (target only) | informer         | Informer2020       |           18 |        6.254 |         9.522 |      4.223 |       6.057 |
| F2               | Target + temporal features   | gat_informer     | GAT-Informer       |           18 |        6.059 |         9.546 |      4.092 |       6.097 |
| baseline         | baseline                     | persistence_last | Persistence (last) |           18 |        7.093 |        11.514 |      5.540 |       8.794 |
| baseline         | baseline                     | persistence_ma24 | Persistence (MA24) |           18 |        6.734 |        10.367 |      4.817 |       7.204 |

## Best config per model

| feature_config   | description                  | model        | model_label   |   n_stations |   hourly_mae |   hourly_rmse |   ma24_mae |   ma24_rmse |
|:-----------------|:-----------------------------|:-------------|:--------------|-------------:|-------------:|--------------:|-----------:|------------:|
| F3               | F2 + meteorology             | airformer    | AirFormer     |           18 |        5.848 |         9.127 |      3.890 |       5.704 |
| F1               | Autoregressive (target only) | informer     | Informer2020  |           18 |        6.254 |         9.522 |      4.223 |       6.057 |
| F2               | Target + temporal features   | gat_informer | GAT-Informer  |           18 |        6.059 |         9.546 |      4.092 |       6.097 |

## Global mean by F x model

| feature_config   | description                       | model        | model_label   |   n_stations |   hourly_mae |   hourly_rmse |   ma24_mae |   ma24_rmse |
|:-----------------|:----------------------------------|:-------------|:--------------|-------------:|-------------:|--------------:|-----------:|------------:|
| F1               | Autoregressive (target only)      | informer     | Informer2020  |           18 |        6.254 |         9.522 |      4.223 |       6.057 |
| F1               | Autoregressive (target only)      | airformer    | AirFormer     |           18 |        6.148 |         9.501 |      4.212 |       6.229 |
| F1               | Autoregressive (target only)      | gat_informer | GAT-Informer  |           18 |        6.054 |         9.551 |      4.091 |       6.103 |
| F2               | Target + temporal features        | informer     | Informer2020  |           18 |        6.363 |         9.572 |      4.363 |       6.156 |
| F2               | Target + temporal features        | airformer    | AirFormer     |           18 |        6.072 |         9.516 |      4.206 |       6.274 |
| F2               | Target + temporal features        | gat_informer | GAT-Informer  |           18 |        6.059 |         9.546 |      4.092 |       6.097 |
| F3               | F2 + meteorology                  | informer     | Informer2020  |           18 |        6.488 |         9.700 |      4.492 |       6.310 |
| F3               | F2 + meteorology                  | airformer    | AirFormer     |           18 |        5.848 |         9.127 |      3.890 |       5.704 |
| F3               | F2 + meteorology                  | gat_informer | GAT-Informer  |           18 |        6.065 |         9.557 |      4.089 |       6.111 |
| F4               | F3 + co-pollutants                | informer     | Informer2020  |           18 |        6.446 |         9.690 |      4.417 |       6.273 |
| F4               | F3 + co-pollutants                | airformer    | AirFormer     |           18 |        5.947 |         9.206 |      4.004 |       5.828 |
| F4               | F3 + co-pollutants                | gat_informer | GAT-Informer  |           18 |        6.293 |         9.701 |      4.264 |       6.311 |
| F5               | F4 + Saharan dust intrusion (Sah) | informer     | Informer2020  |           18 |        6.534 |         9.770 |      4.508 |       6.383 |
| F5               | F4 + Saharan dust intrusion (Sah) | airformer    | AirFormer     |           18 |        5.903 |         9.142 |      3.929 |       5.725 |
| F5               | F4 + Saharan dust intrusion (Sah) | gat_informer | GAT-Informer  |           18 |        6.068 |         9.564 |      4.133 |       6.123 |

## Station wins by F-config

| feature_config   |   n_stations |   informer_wins |   airformer_wins |   gat_informer_wins |   informer_mean |   airformer_mean |   gat_informer_mean |
|:-----------------|-------------:|----------------:|-----------------:|--------------------:|----------------:|-----------------:|--------------------:|
| F1               |           18 |               7 |                8 |                   3 |           6.057 |            6.229 |               6.103 |
| F2               |           18 |               4 |                8 |                   6 |           6.156 |            6.274 |               6.097 |
| F3               |           18 |               0 |               16 |                   2 |           6.310 |            5.704 |               6.111 |
| F4               |           18 |               1 |               15 |                   2 |           6.273 |            5.828 |               6.311 |
| F5               |           18 |               1 |               14 |                   3 |           6.383 |            5.725 |               6.123 |

## Skill vs persistence MA24

| feature_config   | description                       | model        | model_label   |   n_stations |   hourly_mae |   hourly_rmse |   ma24_mae |   ma24_rmse | baseline         |   baseline_rmse |   skill |   improvement_pct |
|:-----------------|:----------------------------------|:-------------|:--------------|-------------:|-------------:|--------------:|-----------:|------------:|:-----------------|----------------:|--------:|------------------:|
| F1               | Autoregressive (target only)      | informer     | Informer2020  |           18 |        6.254 |         9.522 |      4.223 |       6.057 | persistence_ma24 |           7.204 |   0.159 |            15.913 |
| F1               | Autoregressive (target only)      | airformer    | AirFormer     |           18 |        6.148 |         9.501 |      4.212 |       6.229 | persistence_ma24 |           7.204 |   0.135 |            13.532 |
| F1               | Autoregressive (target only)      | gat_informer | GAT-Informer  |           18 |        6.054 |         9.551 |      4.091 |       6.103 | persistence_ma24 |           7.204 |   0.153 |            15.273 |
| F2               | Target + temporal features        | informer     | Informer2020  |           18 |        6.363 |         9.572 |      4.363 |       6.156 | persistence_ma24 |           7.204 |   0.145 |            14.542 |
| F2               | Target + temporal features        | airformer    | AirFormer     |           18 |        6.072 |         9.516 |      4.206 |       6.274 | persistence_ma24 |           7.204 |   0.129 |            12.903 |
| F2               | Target + temporal features        | gat_informer | GAT-Informer  |           18 |        6.059 |         9.546 |      4.092 |       6.097 | persistence_ma24 |           7.204 |   0.154 |            15.364 |
| F3               | F2 + meteorology                  | informer     | Informer2020  |           18 |        6.488 |         9.700 |      4.492 |       6.310 | persistence_ma24 |           7.204 |   0.124 |            12.411 |
| F3               | F2 + meteorology                  | airformer    | AirFormer     |           18 |        5.848 |         9.127 |      3.890 |       5.704 | persistence_ma24 |           7.204 |   0.208 |            20.814 |
| F3               | F2 + meteorology                  | gat_informer | GAT-Informer  |           18 |        6.065 |         9.557 |      4.089 |       6.111 | persistence_ma24 |           7.204 |   0.152 |            15.170 |
| F4               | F3 + co-pollutants                | informer     | Informer2020  |           18 |        6.446 |         9.690 |      4.417 |       6.273 | persistence_ma24 |           7.204 |   0.129 |            12.921 |
| F4               | F3 + co-pollutants                | airformer    | AirFormer     |           18 |        5.947 |         9.206 |      4.004 |       5.828 | persistence_ma24 |           7.204 |   0.191 |            19.098 |
| F4               | F3 + co-pollutants                | gat_informer | GAT-Informer  |           18 |        6.293 |         9.701 |      4.264 |       6.311 | persistence_ma24 |           7.204 |   0.124 |            12.391 |
| F5               | F4 + Saharan dust intrusion (Sah) | informer     | Informer2020  |           18 |        6.534 |         9.770 |      4.508 |       6.383 | persistence_ma24 |           7.204 |   0.114 |            11.387 |
| F5               | F4 + Saharan dust intrusion (Sah) | airformer    | AirFormer     |           18 |        5.903 |         9.142 |      3.929 |       5.725 | persistence_ma24 |           7.204 |   0.205 |            20.526 |
| F5               | F4 + Saharan dust intrusion (Sah) | gat_informer | GAT-Informer  |           18 |        6.068 |         9.564 |      4.133 |       6.123 | persistence_ma24 |           7.204 |   0.150 |            14.997 |

## Best model×config per station (MA24 RMSE)

| station           | station_key     | model        | feature_config   |   hourly_mae |   hourly_rmse |   ma24_mae |   ma24_rmse |
|:------------------|:----------------|:-------------|:-----------------|-------------:|--------------:|-----------:|------------:|
| SAN MIGUEL        | SAN_MIGUEL      | gat_informer | F5               |        3.703 |         5.593 |      2.700 |       3.873 |
| ABANTO            | ABANTO          | airformer    | F3               |        4.561 |         7.032 |      2.977 |       4.265 |
| SANTURCE          | SANTURCE        | airformer    | F3               |        5.146 |         7.769 |      3.178 |       4.718 |
| MUSKIZ            | MUSKIZ          | airformer    | F3               |        4.066 |         7.362 |      2.922 |       4.919 |
| ALONSOTEGI        | ALONSOTEGI      | airformer    | F5               |        5.369 |         8.183 |      3.421 |       5.100 |
| MAZARREDO         | MAZARREDO       | airformer    | F5               |        5.027 |         7.896 |      3.490 |       5.262 |
| SANGRONIZ         | SANGRONIZ       | airformer    | F5               |        5.984 |         9.247 |      3.621 |       5.305 |
| CASTREJANA        | CASTREJANA      | airformer    | F3               |        6.313 |         9.350 |      3.720 |       5.343 |
| LAS CARRERAS      | LAS_CARRERAS    | airformer    | F5               |        5.775 |         8.806 |      3.852 |       5.427 |
| EUROPA            | EUROPA          | airformer    | F5               |        5.542 |         8.691 |      3.736 |       5.615 |
| Mª DIAZ HARO      | M_DIAZ_HARO     | airformer    | F3               |        5.280 |         8.341 |      3.810 |       5.660 |
| ARRAIZ (Monte)    | ARRAIZ_MONTE    | gat_informer | F1               |        5.134 |         8.168 |      3.901 |       5.928 |
| ERANDIO           | ERANDIO         | airformer    | F5               |        6.096 |         9.566 |      4.232 |       6.054 |
| SAN JULIAN        | SAN_JULIAN      | airformer    | F5               |        6.316 |         9.909 |      4.020 |       6.125 |
| BARAKALDO         | BARAKALDO       | airformer    | F3               |        6.831 |        10.347 |      4.712 |       6.608 |
| BASAURI           | BASAURI         | airformer    | F5               |        6.537 |        10.863 |      4.531 |       6.629 |
| ZIERBENA (Puerto) | ZIERBENA_PUERTO | airformer    | F5               |        7.713 |        13.042 |      4.549 |       6.766 |
| ALGORTA_BBIZI2    | ALGORTA_BBIZI2  | informer     | F1               |        8.871 |        13.041 |      5.521 |       7.582 |

## Oracle vs single model

| strategy                | model        | model_label               | feature_config   |   n_stations |   ma24_rmse |   vs_oracle_delta |   improvement_vs_oracle_pct |   skill_vs_pers_ma24 |   improvement_vs_pers_pct |
|:------------------------|:-------------|:--------------------------|:-----------------|-------------:|------------:|------------------:|----------------------------:|---------------------:|--------------------------:|
| oracle_best_per_station | oracle       | Oracle (best per station) | mixed            |           18 |       5.621 |             0.000 |                       0.000 |                0.220 |                    21.969 |
| single_model_best_F     | airformer    | AirFormer                 | F3               |           18 |       5.704 |             0.083 |                       1.458 |                0.208 |                    20.814 |
| single_model_best_F     | informer     | Informer2020              | F1               |           18 |       6.057 |             0.436 |                       7.202 |                0.159 |                    15.913 |
| single_model_best_F     | gat_informer | GAT-Informer              | F2               |           18 |       6.097 |             0.476 |                       7.804 |                0.154 |                    15.364 |

