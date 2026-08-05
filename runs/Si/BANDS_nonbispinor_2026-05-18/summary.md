# Si non-bispinor band-count sensitivity — summary table

| config | variant | band_chunk | r_chunk | n_chunks | gflat_cs | HWM_pred (GB/dev) | mem_stats peak | nvsmi peak | %-err |
|---|---|---|---|---|---|---|---|---|---|
| 3x3x3_nb100 | platform_false | 128 | 27000 | 1 | 100 | 15.63 | — | 2.93 | — |
| 3x3x3_nb100 | bfc_pre95 | 128 | 27000 | 1 | 100 | 15.63 | 15.71 | 39.17 | -0.5% |
| 3x3x3_nb200 | platform_false | 256 | 27000 | 1 | 100 | 15.7 | — | 3.03 | — |
| 3x3x3_nb200 | bfc_pre95 | 256 | 27000 | 1 | 100 | 15.7 | 18.24 | 39.17 | -13.9% |
| 4x4x4_nb100 | platform_false | 128 | 7932 | 4 | 100 | 22.39 | — | 4.06 | — |
| 4x4x4_nb100 | bfc_pre95 | 128 | 7932 | 4 | 100 | 22.39 | 22.66 | 39.17 | -1.2% |
| 4x4x4_nb200 | platform_false | 256 | 7812 | 4 | 100 | 22.4 | — | 4.5 | — |
| 4x4x4_nb200 | bfc_pre95 | 256 | 7812 | 4 | 100 | 22.4 | 23.13 | 39.17 | -3.2% |

# Per-peak components — bfc_pre95 (true peak detection)

| config | Peak A | Peak B | Peak C | Peak D | Peak E | bottleneck |
|---|---|---|---|---|---|---|
| 3x3x3_nb100 | 0.47 | 0.25 | 15.63 | 1.44 | 0.14 | C_fit_one_rchunk |
| 3x3x3_nb200 | 0.93 | 0.32 | 15.7 | 1.51 | 0.17 | C_fit_one_rchunk |
| 4x4x4_nb100 | 0.56 | 2.05 | 22.39 | 2.59 | 0.71 | C_fit_one_rchunk |
| 4x4x4_nb200 | 1.09 | 2.38 | 22.4 | 2.9 | 0.88 | C_fit_one_rchunk |

# Peak C component breakdown — bfc_pre95

| config | P_pair | zeta_out | centroids_persist | gflat_acc | L_q | sphere_idx |
|---|---|---|---|---|---|---|
| 3x3x3_nb100 | 14.277 | 1.19 | 0.071 | 0.071 | 0.018 | 0.003 |
| 3x3x3_nb200 | 14.277 | 1.19 | 0.141 | 0.071 | 0.018 | 0.003 |
| 4x4x4_nb100 | 19.884 | 1.657 | 0.334 | 0.338 | 0.17 | 0.007 |
| 4x4x4_nb200 | 19.583 | 1.632 | 0.668 | 0.338 | 0.17 | 0.007 |
