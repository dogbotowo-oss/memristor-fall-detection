# v-8-final-new-0625：基于 v8 final 的细粒度 ADL subtype 定向抑制

## 版本目的

本版本沿用 `v8 final` 的 CNN+SNN 主体结构，尝试按路线 A 继续优化：

- 不再使用一个统一的 ADL 抑制强度；
- 改为对 `sit_descent`、`bend_reach`、`squat_kneel` 分别设置 penalty；
- 并且只在 `hard-negative` 条件下启用该 penalty；
- 目标是尽量保留 `v8 final` 的 ADL 误报控制能力，同时减少最像 Fall 的 ADL 误报。

## 本次代码改动

主代码：

- `code\automatic_fall_detector.py`

新增开关：

- `--targeted-adl-subtype-weights`
- `--targeted-adl-penalty-hard-only`
- `--targeted-adl-penalty-score-threshold`

本次核心逻辑：

1. 保留 `v8 final` 的 SNN gate penalty、teacher-student gaussian noise、Crossbar device dynamics。
2. 新增 ADL subtype head，用于辅助学习 ADL 内部分型。
3. 对目标 subtype 不再统一使用一个总 penalty，而是：
   - `sit_descent=0.04`
   - `bend_reach=0.05`
   - `squat_kneel=0.07`
4. 该 penalty 只在 `hard_negative_score >= 0.30` 的 ADL 窗口上生效，避免把普通 ADL 一起压掉。

## 运行环境

- 项目根目录：`F:\12.18-2`
- Python 环境：`F:\ANACONDA\envs\my_yizu_3.10\python.exe`
- Crossbar 数据目录：`F:\12.18-2\data`

## 数据划分

严格保持：

- 训练：`Subject 1 + Subject 2 + Zenodo train`
- 验证与阈值校准：`Subject 3 + Zenodo val`
- 最终独立测试：`Subject 4`

`Subject 4` 没有参与训练、阈值扫描或参数选择。

## 本次训练配置

初始化模型：

- `F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8\models\automatic_fall_event_detector_cnn_snn_v8_final_p006_penalty006.pth`

关键参数：

- `external-labeled-stride = 16`
- `image-size = 64`
- `snn-adl-gate-penalty = 0.06`
- `hard-negative-normal-weight = 1.10`
- `adl-subtype-loss-weight = 0.10`
- `adl-subtype-hard-only = on`
- `adl-subtype-hard-score-threshold = 0.24`
- `targeted-adl-subtypes = sit_descent bend_reach squat_kneel`
- `targeted-adl-subtype-weights = 0.04 / 0.05 / 0.07`
- `targeted-adl-penalty-hard-only = on`
- `targeted-adl-penalty-score-threshold = 0.30`

训练日志中确认：

- 已加载 `conductance data`
- 已加载 `EPSC / PPF / LTP / LTD`
- 训练窗口统计为：
  - `train = 2508`
  - `val = 564`

## 训练现象

训练过程中，最好的验证点出现在较早阶段：

- `Epoch 03`
  - `val_acc = 74.5%`
  - `val_bal = 73.8%`
  - `val_normal = 71.6%`
  - `val_fall = 76.0%`
  - `val_subtype = 41.1%`

后续 epoch 出现明显摆动：

- 一部分轮次 `val_normal` 很高，但 `val_fall` 明显下降；
- 另一部分轮次 `val_fall` 回升，但 `val_normal` 又被压低；
- 最终 `epoch 8` early stop。

这说明：

- subtype 辅助监督确实学到了东西；
- 但该 subtype 信息并没有稳定转化成更好的最终 Fall/ADL 判别边界。

## 严格验证阈值结果

阈值只在 `Subject 3 + Zenodo val` 上扫描。

推荐阈值：

- `0.55`

对应验证集最佳点：

- `Accuracy = 72.13%`
- `Balanced Accuracy = 71.99%`
- `ADL specificity = 63.33%`
- `Fall recall = 80.65%`
- `Precision = 69.44%`
- `TP / TN / FP / FN = 25 / 19 / 11 / 6`

## Subject4 最终结果

- `Threshold = 0.55`
- `Accuracy = 67.57%`
- `Balanced Accuracy = 66.91%`
- `ADL specificity = 75.00%`
- `Fall recall = 58.82%`
- `Precision = 66.67%`
- `TP / TN / FP / FN = 10 / 15 / 5 / 7`

误报 ADL：

- `05, 06, 11, 15, 16`

漏检 Fall：

- `01, 02, 05, 08, 09, 13, 17`

## 与 v8 final 对照

`v8 final`：

- `Accuracy = 78.38%`
- `Balanced Accuracy = 78.24%`
- `ADL specificity = 80.00%`
- `Fall recall = 76.47%`
- `Precision = 76.47%`
- `TP / TN / FP / FN = 13 / 16 / 4 / 4`

`v-8-final-new-0625`：

- `Accuracy = 67.57%`
- `Balanced Accuracy = 66.91%`
- `ADL specificity = 75.00%`
- `Fall recall = 58.82%`
- `Precision = 66.67%`
- `TP / TN / FP / FN = 10 / 15 / 5 / 7`

## 本次结论

这次路线 A 的尝试没有超过 `v8 final`。

主要现象是：

1. ADL 误报相比部分高召回版本有所控制；
2. 但真实 Fall recall 下降明显；
3. 说明“细粒度 subtype penalty + hard-only 条件”虽然更精细，但当前强度和作用形式仍然偏保守。

更具体地说：

- 模型已经学会了部分 ADL subtype 区别；
- 但这些 subtype 信息更多体现在辅助头上；
- 主分类头仍然倾向于把边界往保守方向收紧；
- 最终导致 Fall 漏检增加。

## 对下一步的启发

这版的价值主要在于验证：

- “按 subtype 分权重”是可实现的；
- “hard-only subtype penalty”这条逻辑是稳定可跑的；
- 但如果直接把它叠加到 `v8 final` 上，当前更容易牺牲 Fall recall。

因此下一步更适合：

1. 继续保留这套可分 subtype 的代码框架；
2. 但不要继续加大 penalty；
3. 改为更轻量的 subtype 抑制，或者只对最关键的单个 subtype 做约束；
4. 同时优先观察是否能把 subtype 信息用于 gate 或 post-score 校正，而不是直接压主分类正类概率。

## 文件记录

- 代码：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v-8-final-new-0625\code\automatic_fall_detector.py`
- 模型：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v-8-final-new-0625\models\automatic_fall_event_detector_cnn_snn_v8final_new_0625.pth`
- 训练日志：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v-8-final-new-0625\reports\train_v8final_new_0625.stdout.log`
- 训练曲线：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v-8-final-new-0625\image\training_curve_v8final_new_0625.png`
- 验证阈值扫描：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v-8-final-new-0625\reports\v8final_new_0625_strict_validation_threshold_scan.csv`
- 推荐阈值：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v-8-final-new-0625\reports\v8final_new_0625_recommended_threshold.json`
- Subject4 结果：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v-8-final-new-0625\reports\v8final_new_0625_subject4_final_metrics.json`
