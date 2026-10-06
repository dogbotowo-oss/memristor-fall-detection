# 7.5 model-mined hard-negative optimization

## Version purpose

This folder replaces the earlier hard_negative_score_opt experiment name. The old 4x over-boost experiment is archived in:

_archive_hard_negative_score_opt_4x_overboost_20260518

## Code optimization

7.5 changes hard-negative training from manual score amplification to model-mined hard negatives:

1. Load the 7.3 best checkpoint before fine-tuning.
2. Run the initialized model over normal ADL training windows.
3. Treat normal windows with high pre_fall/fall probability as mined hard negatives.
4. Use only model-mined hard-negative score for sampler boost in this run.
5. Cap the hard-negative sampler multiplier at 2x to avoid the 4x over-boost problem seen in hard_negative_score_opt.
6. Keep the teacher-student consistency branch, but preserve normal_positive_penalty instead of overwriting it.

## Main parameters

- Python/PyTorch environment: my_yizu_3.10
- Initial model: 7.3 best model
- hard_negative_normal_weight: 1.0
- hard_negative_max_multiplier: 2.0
- mined_hard_negative_prob_threshold: 0.45
- mined_hard_negative_source: model_only
- normal_positive_penalty: 0.12
- crossbar_readout_noise_scale: 0.25
- device dynamics: EPSC, PPF, LTP, LTD enabled

## Strict evaluation flow

1. Train/fine-tune with Subject1-2 + Zenodo train.
2. Calibrate threshold on Subject3 + Zenodo val.
3. Test Subject4 once as independent final test.

## Progress note

The previous hard_negative_score_opt run proved that aggressive score amplification can raise Fall recall but damage ADL specificity. 7.5 is designed to test a milder, model-error-driven approach.

## Run result on 2026-05-18

Environment confirmed:

- Python: F:\ANACONDA\envs\my_yizu_3.10\python.exe
- PyTorch: 2.5.1+cu121
- Runtime device used by training/testing: cuda

Model-mining summary:

- normal ADL training windows: 873
- mined hard-negative windows: 173
- mean mined score: 0.055
- top10 mined score: 0.778
- final mean sampler multiplier: 1.05
- top10 sampler multiplier: 1.78

Selected strict mixed-validation threshold:

- fall_prob_threshold: 0.85
- fall_margin_threshold: 0.05
- min_fall_duration_sec: 0.25s

Subject4 final test:

- video_accuracy: 70.27%
- ADL specificity: 70.00% (14/20)
- Fall recall: 70.59% (12/17)
- precision: 66.67%
- balanced_accuracy: 70.29%
- false alarms: 6
- missed falls: 5

Comparison with 7.4:

- 7.4: Acc 72.97%, ADL 80.00%, Fall 64.71%, Balanced 72.35%, false alarms 4, missed falls 6.
- 7.5: Acc 70.27%, ADL 70.00%, Fall 70.59%, Balanced 70.29%, false alarms 6, missed falls 5.

Interpretation:

7.5 is better than the previous 4x hard-negative over-boost run because it avoids severe ADL collapse. It improves Fall recall compared with 7.4, but loses ADL specificity and overall balanced accuracy. Current best strict version remains 7.4, while 7.5 is useful as an ablation showing that model-mined hard negatives trade ADL specificity for Fall recall.
