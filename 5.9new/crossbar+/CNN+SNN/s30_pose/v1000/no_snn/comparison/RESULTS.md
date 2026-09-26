# v1000 LOSO SNN ablation results

Primary matched comparison: `gtest` balanced-accuracy threshold rule, seed 11,
four held-out GMDCSA subjects, 160 pooled out-of-fold videos.

| Configuration | Accuracy | Balanced accuracy | ADL specificity | Fall recall |
|---|---:|---:|---:|---:|
| no-SNN | 62.62±11.05 | 62.55±10.71 | 81.97±15.34 | 43.12±21.46 |
| SNN | 82.12±7.45 | 81.88±7.72 | 89.53±8.02 | 74.22±12.17 |
| SNN − no-SNN | +19.49 | +19.33 | +7.56 | +31.10 |

Values are macro mean±sample SD across the four held-out subjects. The SNN
balanced-accuracy improvement is positive in all four folds: +34.38, +24.00,
+18.51, and +0.44 percentage points for S1–S4, respectively.

Pooled OOF confusion matrices (`TN, FP, FN, TP`):

- no-SNN: `66, 15, 44, 35` (pooled balanced accuracy 62.89%).
- SNN: `72, 9, 19, 60` (pooled balanced accuracy 82.42%).

The `zenodo_only` and `recall_priority` rows in the CSV outputs retain the two
additional pre-existing v1000 threshold protocols for audit and sensitivity
analysis. The main figure should use one predeclared protocol consistently.
