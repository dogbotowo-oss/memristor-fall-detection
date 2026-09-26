# v1000 no-SNN LOSO ablation

This directory contains the matched no-SNN ablation for the four-fold v1000
LOSO experiment. The training data, warm start, seed, device non-idealities,
and evaluation rules match the existing SNN runs; only the SNN branch and its
branch-specific gate/loss options are removed.

Directory layout:

- `models/`: one no-SNN checkpoint per held-out subject.
- `reports/`: training logs and per-fold validation/test metrics.
- `image/`: training curves.
- `comparison/`: fold-level and pooled SNN vs no-SNN tables and figures.
- `scripts/`: reproducible training/evaluation and comparison scripts.

Run the four folds:

```powershell
& 'D:\software\anaconde\envs\my_yizu_3.10\python.exe' `
  '.\scripts\run_loso_no_snn.py'
```

Rebuild comparison outputs after all folds finish:

```powershell
& 'D:\software\anaconde\envs\my_yizu_3.10\python.exe' `
  '.\scripts\build_snn_comparison.py'
```

The primary comparison uses the existing `gtest` balanced-accuracy threshold
rule. `test` (Zenodo-validation-only) and `gtestR` (recall-priority) results are
also retained so the threshold protocol remains auditable.
