# Multi-model comparison (zone 2 · PM10)

Protocol: `seq_len=48`, horizon=24, primary metric **MA24 RMSE** (lower better).
Fair global means use stations common to all *available learned models* within each F-config.

## Best configuration per model

 model_label feature_config  n_stations hourly_mae hourly_rmse ma24_mae ma24_rmse
     XGBoost             F5          18     6.0942      9.0203   3.9764    5.4478
   AirFormer             F5          18     6.0083      9.0842   3.9867    5.6477
Informer2020             F1          18     6.2539      9.5221   4.2229    6.0573
GAT-Informer             F2          18     6.0589      9.5457   4.0918    6.0968

## Global mean (all F × model)

feature_config  model_label  n_stations hourly_mae hourly_rmse ma24_mae ma24_rmse
            F1 Informer2020          18     6.2539      9.5221   4.2229    6.0573
            F1    AirFormer          18     6.1484      9.5007   4.2122    6.2288
            F1 GAT-Informer          18     6.0543      9.5513   4.0911    6.1033
            F1      XGBoost          18     6.2176      9.3898   4.1932    5.9738
            F2 Informer2020          18     6.3627      9.5720   4.3627    6.1560
            F2    AirFormer          18     6.0716      9.5157   4.2055    6.2740
            F2 GAT-Informer          18     6.0589      9.5457   4.0918    6.0968
            F2      XGBoost          18     6.1370      9.2576   4.1345    5.8490
            F3 Informer2020          18     6.4875      9.6997   4.4916    6.3095
            F3    AirFormer          18     5.8481      9.1269   3.8899    5.7042
            F3 GAT-Informer          18     6.0653      9.5568   4.0891    6.1108
            F3      XGBoost          18     6.2060      9.2725   4.1636    5.8188
            F4 Informer2020          18     6.4457      9.6900   4.4167    6.2727
            F4    AirFormer          18     5.9468      9.2057   4.0043    5.8278
            F4 GAT-Informer          18     6.2932      9.7013   4.2645    6.3110
            F4      XGBoost          18     6.1540      9.2447   4.1127    5.7937
            F5 Informer2020          18     6.5340      9.7704   4.5079    6.3832
            F5    AirFormer          18     6.0083      9.0842   3.9867    5.6477
            F5 GAT-Informer          18     6.0643      9.6015   4.1663    6.1740
            F5      XGBoost          18     6.0942      9.0203   3.9764    5.4478

## Skill vs persistence MA24

 model_label feature_config  ma24_rmse  skill improvement_pct
Informer2020             F1   6.057272 0.1591          15.91%
   AirFormer             F1   6.228765 0.1353          13.53%
GAT-Informer             F1   6.103338 0.1527          15.27%
     XGBoost             F1   5.973815 0.1707          17.07%
Informer2020             F2   6.155993 0.1454          14.54%
   AirFormer             F2   6.274050 0.1290          12.90%
GAT-Informer             F2   6.096809 0.1536          15.36%
     XGBoost             F2   5.848967 0.1880          18.80%
Informer2020             F3   6.309507 0.1241          12.41%
   AirFormer             F3   5.704182 0.2081          20.81%
GAT-Informer             F3   6.110792 0.1517          15.17%
     XGBoost             F3   5.818790 0.1922          19.22%
Informer2020             F4   6.272749 0.1292          12.92%
   AirFormer             F4   5.827796 0.1910          19.10%
GAT-Informer             F4   6.310967 0.1239          12.39%
     XGBoost             F4   5.793668 0.1957          19.57%
Informer2020             F5   6.383244 0.1139          11.39%
   AirFormer             F5   5.647662 0.2160          21.60%
GAT-Informer             F5   6.174014 0.1429          14.29%
     XGBoost             F5   5.447788 0.2437          24.37%

## Station wins by F-config

feature_config  n_stations  informer_wins  airformer_wins  gat_informer_wins  xgboost_wins  informer_mean  airformer_mean  gat_informer_mean  xgboost_mean
            F1          18            2.0               6                3.0             7       6.057272        6.228765           6.103338      5.973815
            F2          18            1.0               1                3.0            13       6.155993        6.274050           6.096809      5.848967
            F3          18            0.0               7                2.0             9       6.309507        5.704182           6.110792      5.818790
            F4          18            0.0               3                2.0            13       6.272749        5.827796           6.310967      5.793668
            F5          18            0.0               4                2.0            12       6.383244        5.647662           6.174014      5.447788
           F3S          18            NaN              10                NaN             8            NaN        5.735894                NaN      5.836550
           F3I          18            NaN              11                NaN             7            NaN        5.475510                NaN      5.544308
           F5I          18            NaN               5                NaN            13            NaN        5.724779                NaN      5.421615
          F3IL          18            NaN              10                NaN             8            NaN        5.419931                NaN      5.538853

## Best model×config per station (MA24 RMSE)

        station     station_key        model feature_config  hourly_mae  hourly_rmse  ma24_mae  ma24_rmse
     SAN MIGUEL      SAN_MIGUEL gat_informer             F2    3.735091     5.598509  2.716749   3.875714
         ABANTO          ABANTO    airformer            F5I    4.532980     6.850274  2.903872   3.994987
         MUSKIZ          MUSKIZ      xgboost            F3I    4.191690     6.860463  2.915359   4.186928
     ALONSOTEGI      ALONSOTEGI      xgboost             F5    5.492164     7.980580  3.309748   4.555856
       SANTURCE        SANTURCE    airformer           F3IL    5.275970     7.713797  3.337614   4.672279
      MAZARREDO       MAZARREDO      xgboost             F5    4.926532     7.408211  3.382283   4.730867
      SANGRONIZ       SANGRONIZ    airformer            F3I    5.896293     8.964822  3.515615   4.876307
     CASTREJANA      CASTREJANA    airformer           F3IL    6.362091     9.146402  3.698659   5.005275
   LAS_CARRERAS    LAS_CARRERAS      xgboost            F5I    5.877863     8.800444  3.797113   5.071289
         EUROPA          EUROPA      xgboost            F5I    5.542667     8.389712  3.693594   5.140656
   Mª DIAZ HARO     M_DIAZ_HARO    airformer            F3I    5.160326     7.956934  3.664424   5.148492
   ARRAIZ_Monte    ARRAIZ_MONTE      xgboost            F5I    5.376368     8.032718  3.978369   5.546354
        ERANDIO         ERANDIO    airformer            F3I    6.013597     9.319219  4.098724   5.670725
     SAN JULIAN      SAN_JULIAN    airformer            F3I    6.265815     9.632678  3.935436   5.772630
      BARAKALDO       BARAKALDO      xgboost             F5    6.613623     9.925120  4.230648   5.918472
        BASAURI         BASAURI      xgboost           F3IL    6.838390    10.551700  4.496298   6.061141
ZIERBENA_Puerto ZIERBENA_PUERTO      xgboost             F5    7.955780    12.665907  4.521083   6.183498
 ALGORTA_BBIZI2  ALGORTA_BBIZI2      xgboost            F3I    8.593569    12.459257  5.255263   7.051698

## Oracle routing (best model×F per station)

Upper bound: at each station pick the lowest test MA24 RMSE among learned models.
Not a deployable selector (uses test labels); use as a ceiling vs single global model.

              model_label feature_config  n_stations ma24_rmse vs_oracle_delta improvement_vs_oracle_pct  improvement_vs_pers_pct
Oracle (best per station)          mixed          18    5.1924          0.0000                    0.0000                27.918770
                AirFormer           F3IL          18    5.4199          0.2275                    4.1981                24.760146
                  XGBoost            F5I          18    5.4216          0.2292                    4.2278                24.736763
             Informer2020             F1          18    6.0573          0.8649                   14.2783                15.912529
             GAT-Informer             F2          18    6.0968          0.9044                   14.8342                15.363679

### Who wins how many stations

       model feature_config  n_stations_won  model_label
   airformer            F3I               4    AirFormer
     xgboost             F5               4      XGBoost
     xgboost            F5I               3      XGBoost
   airformer           F3IL               2    AirFormer
     xgboost            F3I               2      XGBoost
   airformer            F5I               1    AirFormer
gat_informer             F2               1 GAT-Informer
     xgboost           F3IL               1      XGBoost

## Realistic station routing (proxy: 1st half → 2nd half)

Proxy: no per-station validation MA24 saved; se elige con la 1ª mitad temporal del test y se mide en la 2ª.

Agreement with oracle on holdout: 11.1% of stations.

                                model_label feature_config  n_stations eval_rmse vs_best_single_delta vs_oracle_delta
            Oracle (chooses using 2nd half)          mixed          18    4.9322              -0.2959          0.0000
                                    XGBoost            F5I          18    5.2281               0.0000          0.2959
Selector realista (elige en 1ª, mide en 2ª)          mixed          18    5.2360               0.0079          0.3039
                                  AirFormer           F3IL          18    5.2725               0.0445          0.3404
                               GAT-Informer             F2          18    5.3361               0.1081          0.4040
                               Informer2020             F1          17    5.6375               0.4095          0.7054

### Models chosen by realistic selector

sel_model sel_config  n_stations_chosen model_label
  xgboost        F3I                  5     XGBoost
airformer        F3I                  4   AirFormer
airformer       F3IL                  4   AirFormer
  xgboost         F5                  3     XGBoost
  xgboost         F1                  1     XGBoost
airformer        F5I                  1   AirFormer
