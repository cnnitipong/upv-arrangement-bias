# Revision analysis summary (seed 42; Semi-Indirect in data = semi-direct in paper)

- **Step0: mean vel D/S/I (km/s)**: 4.46/4.24/3.62  
  _raw mean pulse velocity by arrangement_
- **Step0: chi2 = 144.6 is the**: 3x4 arrangement x quality table (3x2 Doubtful-vs-other chi2 = 14.6)  
  _which table the submitted chi-square belongs to_
- **Step0: delta_indirect exact / delta_semi exact**: 0.2370 / -0.0549  
  _column-centred offsets to direct datum_
- **Step1: columns with Direct / Semi / Indirect**: 12 / 24 / 14  
  _of 26 columns_
- **Step1: columns with both Direct and Indirect**: 0  
  _expected 0_
- **Step1: Direct+Semi / Indirect+Semi / Indirect only**: 12 / 12 / 2  
  _column allocation classes_
- **Step2a: FE Indirect vs Direct (km/s), cluster-robust t CI**: -0.85 [-1.108, -0.592]  
  _within-column additive estimate; negative = indirect reads lower_
- **Step2a: FE Semi vs Direct**: -0.241 [-0.373, -0.108]
- **Step2a: FE Indirect vs Direct, normal CI**: [-1.096, -0.605]  
  _z-based cluster-robust CI_
- **Step2a: Mixed Indirect vs Direct**: -0.858 [-1.001, -0.716]  
  _random intercept by column, REML_
- **Step2a: Mixed Semi vs Direct**: -0.239 [-0.363, -0.116]
- **Step2a: Mixed group variance / residual variance**: 0.027 / 0.204  
  _REML variance components_
- **Step2b: mean(semi-direct) over 12 columns [boot CI]**: -0.235 [-0.361, -0.11]  
  _within-column_
- **Step2b: mean(semi-indirect) over 12 columns [boot CI]**: 0.547 [0.256, 0.787]  
  _within-column_
- **Step2b: chained indirect offset [boot CI]**: 0.782 [0.467, 1.054]  
  _(S-I)+(D-S)_
- **Step2c: raw_direct_minus_indirect**: 0.848; cluster CI [0.673, 1.009]; point CI [0.751, 0.941]  
  _10000 column-resamples vs 2000 point-resamples_
- **Step2c: epsilon_squared**: 0.459; cluster CI [0.325, 0.575]; point CI [0.384, 0.532]  
  _10000 column-resamples vs 2000 point-resamples_
- **Step2c: RR_doubtful_indirect_vs_direct_haldane**: 5.513; cluster CI [2.291, 30.851]; point CI [2.217, 31.333]  
  _10000 column-resamples vs 2000 point-resamples_
- **Step2c: colcentred_Direct**: 0.094; cluster CI [0.045, 0.144]; point CI [0.035, 0.15]  
  _10000 column-resamples vs 2000 point-resamples_
- **Step2c: colcentred_Semi**: 0.149; cluster CI [0.0, 0.305]; point CI [0.051, 0.239]  
  _10000 column-resamples vs 2000 point-resamples_
- **Step2c: colcentred_Indirect**: -0.143; cluster CI [-0.185, -0.084]; point CI [-0.207, -0.082]  
  _10000 column-resamples vs 2000 point-resamples_
- **Step2c: delta_indirect_colcentred**: 0.237; cluster CI [0.161, 0.304]; point CI [0.152, 0.321]  
  _10000 column-resamples vs 2000 point-resamples_
- **Step3: R2 column / arrangement / both**: 0.344 / 0.376 / 0.471
- **Step3: increment of arrangement beyond column / column beyond arrangement**: 0.127 / 0.096
- **Step3: ICC (between-column SS share)**: 0.344
- **Step3: within two-arrangement columns, arrangement increment beyond column**: 0.139 (partial eta2 0.207, F=54.398)  
  _24 columns_
- **Step4: RR Doubtful indirect vs direct**: 6.75; Wald [1.635, 27.859]; score [1.839, 25.488]; cluster boot (Haldane) [2.291, 30.851]  
  _27/216 vs 2/108_
- **Step4: Fisher exact p / exact OR CI**: 0.000802 / OR 7.54 [1.834, 66.693]
- **Step4: residual RR after correction (0.237)**: 4.25 (Doubtful {'Direct': 2, 'Semi-Indirect': 6, 'Indirect': 17})
- **Step4: residual RR after FE correction**: 0.0 (Doubtful {'Direct': 2, 'Semi-Indirect': 5, 'Indirect': 0})  
  _FE delta_ind=+0.85 lifts every indirect point above 3.0, so 0 Indirect Doubtful remain (RR 0)_
- **Step5: delta_ind=0.16**: Doubtful 35->25 (reclassified 10, newly Doubtful 0); by arr after {'Direct': 2, 'Semi-Indirect': 6, 'Indirect': 17}; Excellent 109->101  
  _delta {'Direct': 0.0, 'Semi-Indirect': -0.055, 'Indirect': 0.16}_
- **Step5: delta_ind=0.237**: Doubtful 35->25 (reclassified 10, newly Doubtful 0); by arr after {'Direct': 2, 'Semi-Indirect': 6, 'Indirect': 17}; Excellent 109->108  
  _delta {'Direct': 0.0, 'Semi-Indirect': -0.055, 'Indirect': 0.237}_
- **Step5: delta_ind=0.30**: Doubtful 35->23 (reclassified 12, newly Doubtful 0); by arr after {'Direct': 2, 'Semi-Indirect': 6, 'Indirect': 15}; Excellent 109->110  
  _delta {'Direct': 0.0, 'Semi-Indirect': -0.055, 'Indirect': 0.3}_
- **Step5: delta_ind=FE**: Doubtful 35->7 (reclassified 28, newly Doubtful 0); by arr after {'Direct': 2, 'Semi-Indirect': 5, 'Indirect': 0}; Excellent 109->268  
  _delta {'Direct': 0.0, 'Semi-Indirect': 0.241, 'Indirect': 0.85}_
- **Step5: Excellent explanation**: Excellent total changes at delta=+0.237/-0.055 because (i) semi-direct points are shifted down by 0.055 km/s and those just above 4.5 km/s fall to Good (Excellent->Good by arrangement: {'Direct': 0, 'Semi-Indirect': 12, 'Indirect': 0}), and (ii) indirect points are shifted up by 0.237 and those between 4.263 and 4.5 km/s rise to Excellent (Good->Excellent by arrangement: {'Direct': 1, 'Semi-Indirect': 0, 'Indirect': 10}). Direct points are unchanged. Stored quality differs from the threshold rule applied to raw velocity for 1 point(s): C1-1B : C1.F AB1 v=4.594 stored Good (so the Direct Good->Excellent transition above is this record, not a shift).
- **Step6 [delta_0.237] A_doubtful**: {"points_before": 35, "points_after": 25, "columns_before": 16, "columns_after": 15}  
  _before -> after_
- **Step6 [delta_0.237] B_doubtful_or_medium**: {"points_before": 83, "points_after": 57, "columns_before": 20, "columns_after": 20}  
  _before -> after_
- **Step6 [delta_0.237] C_column_ge_10pct_doubtful**: {"columns_before": 6, "columns_after": 6, "points_in_flagged_columns_before": 94, "points_in_flagged_columns_after": 94, "doubtful_points_before": 35, "doubtful_points_after": 25}  
  _before -> after_
- **Step6 [delta_0.237] C_column_ge_20pct_doubtful**: {"columns_before": 5, "columns_after": 3, "points_in_flagged_columns_before": 70, "points_in_flagged_columns_after": 33, "doubtful_points_before": 35, "doubtful_points_after": 25}  
  _before -> after_
- **Step6 [FE] A_doubtful**: {"points_before": 35, "points_after": 7, "columns_before": 16, "columns_after": 5}  
  _before -> after_
- **Step6 [FE] B_doubtful_or_medium**: {"points_before": 83, "points_after": 27, "columns_before": 20, "columns_after": 13}  
  _before -> after_
- **Step6 [FE] C_column_ge_10pct_doubtful**: {"columns_before": 6, "columns_after": 1, "points_in_flagged_columns_before": 94, "points_in_flagged_columns_after": 15, "doubtful_points_before": 35, "doubtful_points_after": 7}  
  _before -> after_
- **Step6 [FE] C_column_ge_20pct_doubtful**: {"columns_before": 5, "columns_after": 1, "points_in_flagged_columns_before": 70, "points_in_flagged_columns_after": 15, "doubtful_points_before": 35, "doubtful_points_after": 7}  
  _before -> after_
- **Step7: face pair/single raw D-I**: 0.551 / None  
  _n pair 264, single 198_
- **Step7: face confounding**: Face type is confounded with arrangement: all 198 single-face points are Indirect (columns: 12); pair-face points are Direct 108, Semi 138, Indirect 18. An arrangement contrast within single-face points is impossible; within pair-face the Indirect arm has only 18 points (columns: 2), so the pair-face Indirect-vs-Direct contrast is not reliable.
- **Step7: r(vel,rebN) overall / by arrangement (D,S,I)**: 0.226 / [0.097, 0.097, 0.16]
- **Step7: r(corrected vel,rebN) 0.237 / FE**: 0.214 / 0.125
- **Step7: corr(CoV, column mean) raw / 0.237 / FE**: -0.857 / -0.799 / -0.708
- **Step8: Mahalanobis raw flagged / thr**: 12 / 2.663  
  _97.5th pct of d_M_
- **Step8: corrected_0.237 Mahalanobis flagged; overlap with original 12**: 12; 9  
  _newly flagged 3_
- **Step8: corrected_0.237 DBSCAN noise; overlap with original DBSCAN noise / original 12**: 13; 9 / 6
- **Step8: corrected_FE Mahalanobis flagged; overlap with original 12**: 12; 5  
  _newly flagged 7_
- **Step8: corrected_FE DBSCAN noise; overlap with original DBSCAN noise / original 12**: 16; 5 / 4
- **Step8: original DBSCAN noise; overlap with original 12**: 9; 6
- **Step8: concern original**: rho 0.472, p 0.017, top5 overlap 5/5, rank of orig worst 1, worst now C1-1A : C1.C  
  _rho vs priority (priority 1 = worst)_
- **Step8: concern V1_corrected_velocity_recomputed_doubtful**: rho 0.545, p 0.005, top5 overlap 4/5, rank of orig worst 1, worst now C1-1A : C1.C  
  _rho vs priority (priority 1 = worst)_
- **Step8: concern V2_no_doubtful_fraction_corrected_mean_CoV**: rho 0.528, p 0.007, top5 overlap 4/5, rank of orig worst 1, worst now C1-1A : C1.C  
  _rho vs priority (priority 1 = worst)_
- **Step8: concern weights (0.5, 0.25, 0.25)**: rho 0.491, p 0.012, top5 overlap 5/5, rank of orig worst 1
- **Step8: concern weights (0.25, 0.5, 0.25)**: rho 0.48, p 0.015, top5 overlap 5/5, rank of orig worst 1
- **Step8: concern weights (0.25, 0.25, 0.5)**: rho 0.484, p 0.014, top5 overlap 5/5, rank of orig worst 1
- **Step8: concern weights (0.5, 0.5, 0.0)**: rho 0.501, p 0.011, top5 overlap 5/5, rank of orig worst 1
- **Step9: logistic per-fold AUC (original, ungrouped)**: [0.705, 0.765, 0.722, 0.734, 0.662] mean 0.718 SD 0.034  
  _submitted 0.704 +/- 0.048; folds NOT grouped by column_
- **Step9: logistic GroupKFold by column**: [0.677, 0.539, 0.648, 0.618, 0.755] mean 0.647 SD 0.071
- **Step9: repeated random column-grouped 5-fold (20x)**: mean 0.63, SD across repeats 0.035
- **Step9: k-means silhouette k=2..6**: {2: 0.314, 3: 0.299, 4: 0.306, 5: 0.31, 6: 0.313}  
  _submitted 0.314/0.299/0.302/0.310/0.310_
- **Step9: Moran I / p / null 2.5-97.5**: 0.102 / 0.263 / [-0.258, 0.254]  
  _binary weights d<=1.5, n=25_
- **Step10: MC M / seed / sigma_delta (I, S)**: 20000 / 20 / 0.034, 0.030 (hard-coded)  
  _semi sigma_delta is a fixed assumption_
- **Step10: % points not confident (maxP<0.8)**: 18.398  
  _submitted JSON 18.4; spec said 18.8_
- **Step10: % points fragile (flip>0.10) at 1/2/3% noise, simulated**: [15.6, 29.0, 42.2]  
  _submitted 15.6/29.0/42.2_
- **Step10: same, analytic exact**: [15.6, 29.2, 42.9]
- **Step9: script-03 pooled out-of-fold AUC (rank-based) vs submitted 0.636**: 0.637  
  _stream replicated approximately_

## Notes

- step2d: No column was measured with both direct and indirect arrangements (0 of 26). The column-centred indirect mean is computed against a column mean that already contains the arrangement mix of that column (Indirect points sit with Semi-direct points or alone; Direct points sit with Semi-direct points), so the column-centred estimate (delta_indirect = 0.237) is attenuated relative to the within-column contrast. The column fixed-effects estimate uses only within-column differences and is identified through the semi-direct arrangement as a bridge (additivity assumption: the arrangement offset is the same in every column): indirect vs semi within 12 columns, semi vs direct within 12 other columns, chained.
- step5_excellent: Excellent total changes at delta=+0.237/-0.055 because (i) semi-direct points are shifted down by 0.055 km/s and those just above 4.5 km/s fall to Good (Excellent->Good by arrangement: {'Direct': 0, 'Semi-Indirect': 12, 'Indirect': 0}), and (ii) indirect points are shifted up by 0.237 and those between 4.263 and 4.5 km/s rise to Excellent (Good->Excellent by arrangement: {'Direct': 1, 'Semi-Indirect': 0, 'Indirect': 10}). Direct points are unchanged.
- step5_data_quirk: Stored quality differs from the threshold rule applied to raw velocity for 1 point(s): C1-1B : C1.F AB1 v=4.594 stored Good (so the Direct Good->Excellent transition above is this record, not a shift).
- g9_original_labels: original g9 y-labels were mangled by n.replace('C1','').replace(' ','') -> e.g. '-1A:.C' (column prefix lost); replaced by full ids
