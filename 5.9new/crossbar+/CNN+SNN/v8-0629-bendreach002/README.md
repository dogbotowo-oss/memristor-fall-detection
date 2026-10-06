# v8-0629-bendreach002：bend_reach 单通道更轻 penalty

## 版本目的

本版本继续沿用 `v8-0626` 的单 subtype 路线，只保留 `bend_reach` 通道约束，但将 targeted penalty 权重从 `0.03` 降到 `0.02`。

目的：

- 测试更轻的 `bend_reach` penalty 是否能减轻模型保守性；
- 尝试在保持 ADL specificity 的同时恢复 Fall recall；
- 保持 `score_threshold=0.30` 不变，避免同时改动多个因素。

## 工程位置

- 工程根目录：`F:\12.18-2`
- 当前版本目录：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8-0629-bendreach002`
- 主代码：`code\automatic_fall_detector.py`
- 严格评估脚本：`reports\strict_eval_v8_0629_bendreach002.py`

## 数据与约束

本版本继续遵守严格划分：

- 训练：`Zenodo train`
- 训练过程验证：`Zenodo val`
- 阈值校准：`Subject3 + Zenodo val`
- 最终独立测试：`Subject4`

`Subject4` 没有参与训练、模型选择、阈值扫描、sampler 权重选择或 penalty 权重选择。

## 关键参数

与 `v8-0626` 对齐：

- `image-size = 64`
- `external-labeled-stride = 16`
- `snn-adl-gate-penalty = 0.06`
- `hard-negative-normal-weight = 1.10`
- `hard-negative-min-percentile = 0.88`
- `hard-negative-score-gamma = 2.0`
- `adl-context-min-percentile = 0.95`
- `adl-context-extra-copies = 1`
- `adl-subtype-loss-weight = 0.10`
- `adl-subtype-hard-only = on`
- `adl-subtype-hard-score-threshold = 0.24`
- `targeted-adl-penalty-hard-only = on`
- `targeted-adl-penalty-score-threshold = 0.30`

本版本唯一核心差异：

- `targeted-adl-subtypes = bend_reach`
- `targeted-adl-subtype-weights = bend_reach=0.02`

## 数据规模核对

最终训练口径与 `v8-0626` / `v-8-final-new-0625` 对齐：

- `train = 2508`
- `val = 564`
- 训练 ADL / Fall：`1031 / 1477`
- 验证 ADL / Fall：`197 / 367`

训练日志确认：

- 已加载 conductance data：`HRS=9164, LRS=10001`
- 已加载 device dynamics：`EPSC / PPF / LTP / LTD`
- ADL subtype 统计：
  - train：`other=432, sit_descent=227, bend_reach=167, squat_kneel=129, lie_ground=76`
  - val：`other=74, sit_descent=44, bend_reach=26, squat_kneel=38, lie_ground=15`

## 训练过程

训练从 `v8 final` p006 模型初始化：

`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8\models\automatic_fall_event_detector_cnn_snn_v8_final_p006_penalty006.pth`

训练现象：

- Early stopping 出现在 `epoch 8`
- `Epoch 01`：`val_acc=73.9%`，`val_bal=70.2%`，`val_normal=57.9%`，`val_fall=82.6%`
- `Epoch 03`：`val_acc=73.0%`，`val_bal=70.4%`，`val_normal=61.4%`，`val_fall=79.3%`
- 后续 epoch 中 ADL specificity 与 Fall recall 仍有明显摆动

## 严格验证阈值扫描

阈值只在 `Subject3 + Zenodo val` 上选择。

推荐阈值：

- `0.70`

推荐阈值对应验证集结果：

- Accuracy：`73.77%`
- Balanced Accuracy：`73.71%`
- ADL specificity：`70.00%`
- Fall recall：`77.42%`
- Precision：`72.73%`
- TP / TN / FP / FN：`24 / 21 / 9 / 7`

值得注意：

- `threshold=0.35-0.50` 的验证集 Fall recall 达到 `90.32%`，但 ADL specificity 只有 `53.33%`；
- 按既定阈值选择策略，`threshold=0.70` 的 balanced accuracy 更高，因此被选中；
- 这说明本版本的最终谨慎性主要来自验证集阈值校准结果，而不仅仅是 `bend_reach=0.02` 本身。

## Subject4 最终独立测试

文件：

`reports\v8_0629_bendreach002_subject4_final_metrics.json`

结果：

- Threshold：`0.70`
- Accuracy：`78.38%`
- Balanced Accuracy：`76.47%`
- ADL specificity：`100.00%`
- Fall recall：`52.94%`
- Precision：`100.00%`
- TP / TN / FP / FN：`9 / 20 / 0 / 8`

误报 ADL：

- 无

漏检 Fall：

- `01, 02, 05, 08, 09, 11, 13, 17`

## 与关键版本对比

### v8 final p006

- Accuracy：`78.38%`
- Balanced Accuracy：`78.24%`
- ADL specificity：`80.00%`
- Fall recall：`76.47%`
- Precision：`76.47%`
- TP / TN / FP / FN：`13 / 16 / 4 / 4`

### v8-0626：bend_reach=0.03

- Threshold：`0.50`
- Accuracy：`75.68%`
- Balanced Accuracy：`74.41%`
- ADL specificity：`90.00%`
- Fall recall：`58.82%`
- Precision：`83.33%`
- TP / TN / FP / FN：`10 / 18 / 2 / 7`

### v8-0629-bendreach002：bend_reach=0.02

- Threshold：`0.70`
- Accuracy：`78.38%`
- Balanced Accuracy：`76.47%`
- ADL specificity：`100.00%`
- Fall recall：`52.94%`
- Precision：`100.00%`
- TP / TN / FP / FN：`9 / 20 / 0 / 8`

## 结论

本版本没有超过 `v8 final` p006，也没有优于 `v8-0626` 的综合平衡。

主要现象：

- `bend_reach=0.02` 并没有让最终 Subject4 Fall recall 恢复；
- 验证集选择阈值从 `v8-0626` 的 `0.50` 推高到 `0.70`；
- Subject4 ADL specificity 达到 `100.00%`，说明模型非常谨慎；
- Fall recall 降到 `52.94%`，漏检增加到 8 个。

因此，这版更像是“极低误报、强谨慎”的版本，不适合作为主线结果。

## 下一步建议

如果继续围绕 `bend_reach` 通道微调，单纯降低权重并不一定能恢复 Fall recall，因为阈值校准会把决策点推向更高阈值。

更值得尝试的方向：

1. 保持 `bend_reach=0.03` 或 `0.02`，但提高 `targeted-adl-penalty-score-threshold` 到 `0.32 / 0.35`，让 penalty 只作用于更 hard 的 bend_reach；
2. 尝试 `bend_reach=0.01`，观察验证集是否仍把阈值推到 `0.70`；
3. 引入 Fall rescue / temporal event positive mining，而不是继续只调 ADL penalty；
4. 对阈值选择策略做论文口径内的保守改造，例如在 balanced accuracy 接近时加入 Fall recall 下限，但仍必须只基于 `Subject3 + external val`。

## 文件记录

- 模型：`models\automatic_fall_event_detector_cnn_snn_v8_0629_bendreach002.pth`
- 训练日志：`reports\train_v8_0629_bendreach002.stdout.log`
- 训练曲线：`image\training_curve_v8_0629_bendreach002.png`
- 阈值扫描：`reports\v8_0629_bendreach002_strict_validation_threshold_scan.csv`
- 推荐阈值：`reports\v8_0629_bendreach002_recommended_threshold.json`
- Subject4 最终结果：`reports\v8_0629_bendreach002_subject4_final_metrics.json`
- Subject4 视频级结果：`reports\v8_0629_bendreach002_subject4_final_video_metrics.csv`
