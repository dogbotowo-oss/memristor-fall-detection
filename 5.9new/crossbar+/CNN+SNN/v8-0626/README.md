# v8-0626：只对 bend_reach 单一 ADL subtype 做轻量约束

## 版本目的

本版本是在 `v-8-final-new-0625` 路线 A 的基础上继续收缩实验范围：不再同时压制 `sit_descent / bend_reach / squat_kneel` 三个 ADL subtype，而是只对 `bend_reach` 一个通道做轻量 targeted penalty。

设计动机：

- `bend_reach` 对应弯腰、伸手、接近地面但未跌倒动作；
- 相比 `sit_descent` 和 `squat_kneel`，它与真实跌倒瞬间的快速下降/低姿态重叠更小；
- 因此它更适合作为第一轮“单通道 ADL 误报抑制”实验；
- 目标是观察能否减少 ADL 误报，同时避免继续牺牲 Fall recall。

## 工程位置

- 工程根目录：`F:\12.18-2`
- 当前版本目录：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8-0626`
- 主代码：`code\automatic_fall_detector.py`
- 严格评估脚本：`reports\strict_eval_v8_0626.py`

## 数据划分规则

本版本继续遵守严格划分：

- 训练：`Zenodo train`
- 训练过程验证：`Zenodo val`
- 阈值校准：`Subject3 + Zenodo val`
- 最终独立测试：`Subject4`

`Subject4` 没有参与训练、模型选择、阈值扫描、sampler 权重选择或 penalty 权重选择。

## 关键参数

与 `v-8-final-new-0625` 对齐的设置：

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

本版本唯一核心实验差异：

- `targeted-adl-subtypes = bend_reach`
- `targeted-adl-subtype-weights = bend_reach=0.03`

## 数据规模核对

本版本最终采用与上一版路线 A 对齐的数据口径：

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

说明：一开始曾用过更低的 `adl-context-min-percentile=0.88`，会导致训练窗口变为 `2572`，与上一版不可比；最终已改回 `0.95` 并重跑。

## 训练过程

训练从 `v8 final` p006 模型初始化：

`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8\models\automatic_fall_event_detector_cnn_snn_v8_final_p006_penalty006.pth`

训练现象：

- Early stopping 出现在 `epoch 8`
- 最好训练过程验证点出现在较早 epoch
- `Epoch 03`：`val_acc=73.8%`，`val_bal=72.2%`，`val_normal=67.0%`，`val_fall=77.4%`
- 后续 epoch 中 ADL specificity 与 Fall recall 仍有明显摆动

## 严格验证阈值扫描

阈值只在 `Subject3 + Zenodo val` 上选择。

推荐阈值：

- `0.50`

推荐阈值对应验证集结果：

- Accuracy：`72.13%`
- Balanced Accuracy：`71.99%`
- ADL specificity：`63.33%`
- Fall recall：`80.65%`
- Precision：`69.44%`
- TP / TN / FP / FN：`25 / 19 / 11 / 6`

验证集中 `threshold=0.35 / 0.40 / 0.45` 的 Fall recall 更高，为 `83.87%`，但 ADL specificity 只有 `60.00%`；按既定规则，`threshold=0.50` 因 balanced accuracy 略高且 ADL specificity 更好而被选中。

## Subject4 最终独立测试

文件：

`reports\v8_0626_subject4_final_metrics.json`

结果：

- Threshold：`0.50`
- Accuracy：`75.68%`
- Balanced Accuracy：`74.41%`
- ADL specificity：`90.00%`
- Fall recall：`58.82%`
- Precision：`83.33%`
- TP / TN / FP / FN：`10 / 18 / 2 / 7`

误报 ADL：

- `05, 06`

漏检 Fall：

- `01, 02, 05, 08, 09, 13, 17`

## 与关键版本对比

### v8 final p006

- Accuracy：`78.38%`
- Balanced Accuracy：`78.24%`
- ADL specificity：`80.00%`
- Fall recall：`76.47%`
- Precision：`76.47%`
- TP / TN / FP / FN：`13 / 16 / 4 / 4`

### v-8-final-new-0625（三通道 targeted subtype penalty）

- Accuracy：`67.57%`
- Balanced Accuracy：`66.91%`
- ADL specificity：`75.00%`
- Fall recall：`58.82%`
- Precision：`66.67%`
- TP / TN / FP / FN：`10 / 15 / 5 / 7`

### v8-0626（bend_reach only）

- Accuracy：`75.68%`
- Balanced Accuracy：`74.41%`
- ADL specificity：`90.00%`
- Fall recall：`58.82%`
- Precision：`83.33%`
- TP / TN / FP / FN：`10 / 18 / 2 / 7`

## 结论

本版本没有超过 `v8 final` p006，因此不能作为当前主线论文结果。

但它相对 `v-8-final-new-0625` 有一个明确改善：

- ADL 误报从 `5` 个降到 `2` 个；
- ADL specificity 从 `75.00%` 提升到 `90.00%`；
- Precision 从 `66.67%` 提升到 `83.33%`；
- Accuracy 从 `67.57%` 提升到 `75.68%`。

问题也很明确：

- Fall recall 仍然只有 `58.82%`；
- 漏检 Fall 与三通道版本完全一致，仍为 `01, 02, 05, 08, 09, 13, 17`；
- 说明单压 `bend_reach` 能减少误报，但没有解决真实跌倒漏检。

## 下一步建议

如果继续沿单通道路线走，不建议加大 `bend_reach` penalty，因为当前误报已经明显减少，主要瓶颈变成 Fall recall。

更合理的下一步是：

1. 保留 `bend_reach` 单通道轻约束；
2. 加入一个 Fall rescue / temporal event positive mining 分支，用动态下坠特征恢复真实跌倒；
3. 或者把 subtype 信息用于 post-score 校正，而不是继续压低主分类 Fall 概率；
4. 如果还想扫 penalty，只建议轻扫 `bend_reach=0.01 / 0.02 / 0.03`，观察能否在保留 ADL specificity 的同时恢复 Fall recall。

## 文件记录

- 模型：`models\automatic_fall_event_detector_cnn_snn_v8_0626_bendreach_only.pth`
- 训练日志：`reports\train_v8_0626.stdout.log`
- 训练曲线：`image\training_curve_v8_0626.png`
- 阈值扫描：`reports\v8_0626_strict_validation_threshold_scan.csv`
- 推荐阈值：`reports\v8_0626_recommended_threshold.json`
- Subject4 最终结果：`reports\v8_0626_subject4_final_metrics.json`
- Subject4 视频级结果：`reports\v8_0626_subject4_final_video_metrics.csv`
