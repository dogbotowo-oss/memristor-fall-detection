# v10：Fall event positive mining

## 版本目的

v10 在 v8 final 的基础上只新增一项改动：

- 对训练集中的 Fall 窗口做 `Fall event positive mining`

目标是让模型更关注真正跌倒过程中“快速下落、重心突变、运动突增、姿态快速变化”的窗口，从而增强对真实 Fall 瞬间的敏感性。

本版本没有引入新的数据集，也没有使用 Subject4 做现象分析或调参。

## 数据划分

- 训练：GMDCSA Subject1-2 + `F:\12.18-2\zenodo_falldb_video_split\train`
- 验证与阈值校准：GMDCSA Subject3 + `F:\12.18-2\zenodo_falldb_video_split\val`
- 最终独立测试：`F:\12.18-2\gmdcsa_subject4_test\Subject 4`

严格约束：

- Subject4 只用于最终一次独立测试
- 阈值选择只使用 `Subject3 + Zenodo val`
- 训练与评估均使用 `F:\ANACONDA\envs\my_yizu_3.10\python.exe`
- 显式传入 `--data-dir F:\12.18-2\data`

## 核心改动

在 `code\automatic_fall_detector.py` 中新增：

- `compute_fall_event_positive_score(window_frames)`
- `--fall-event-positive-weight`
- `--fall-event-positive-min-percentile`
- `--fall-event-positive-score-gamma`
- `--fall-event-positive-max-multiplier`

逻辑：

1. 对 Fall 窗口计算事件分数，重点看快速下落、中心速度、运动峰值、运动均值、高度变化、宽高比变化等动态特征。
2. 只对高分 Fall 窗口提高采样概率。
3. 不改 ADL hard-negative、SNN gate、teacher-student noise、device dynamics 的原有逻辑。

这一步本质上是在训练阶段告诉模型：不是所有 Fall 窗口同等重要，真正接近“跌倒瞬间”的窗口更值得反复学习。

## 训练参数

v10 训练从 v8 final 初始化：

- 初始化模型：`models\automatic_fall_event_detector_cnn_snn_v8_final_source.pth`
- 输出模型：`models\automatic_fall_event_detector_cnn_snn_v10_fall_event_positive_w050.pth`

主要参数：

```text
--use-snn-temporal-branch
--use-snn-fusion-gate
--snn-fusion-gate-mode group_channel
--learnable-snn-dynamics
--freeze-cnn-gru-epochs 2
--teacher-student-noise
--student-noise-mode gaussian
--use-device-dynamics
--crossbar-readout-noise-scale 0.25
--hard-negative-normal-weight 1.10
--hard-negative-min-percentile 0.88
--hard-negative-score-gamma 2.00
--normal-positive-penalty 0.00
--snn-adl-gate-penalty 0.06
--snn-event-score-threshold 0.55
--snn-event-score-sharpness 10.0
--fall-event-positive-weight 0.50
--fall-event-positive-min-percentile 0.85
--fall-event-positive-score-gamma 1.50
--fall-event-positive-max-multiplier 3.00
--external-labeled-stride 16
--adl-context-path-keyword zenodo_falldb_video_split/train/adl
--adl-context-min-percentile 0.95
--adl-context-extra-copies 1
--image-size 64
--skip-visual-outputs
```

训练日志确认：

```text
Loaded conductance data: HRS=9164, LRS=10001
Loaded device dynamics: epsc=EPSC-0.001-60-60.xls, ppf=PPF-0.001-60-60.xls, ltp=LTP-0.001-1MS.xls, ltd=LTD-0.02-1MS.xls
```

训练窗口统计：

```text
train windows: 3539, normal 1758, fall 1781
val windows: 1076, normal 573, fall 503
fall positive boosted_windows: 267
```

## 训练结果

训练未显式传 `--epochs`，因此使用代码默认 `14` 轮；但在第 7 轮触发 early stopping。

关键验证结果：

| Epoch | Val Acc | Val Balanced Acc | Val ADL Specificity | Val Fall Recall |
|---|---:|---:|---:|---:|
| 1 | 72.90% | 73.20% | 68.10% | 78.30% |
| 2 | 72.80% | 73.20% | 66.00% | 80.50% |
| 3 | 69.90% | 70.70% | 58.30% | 83.10% |
| 4 | 73.00% | 72.80% | 77.10% | 68.40% |
| 5 | 69.10% | 69.80% | 58.30% | 81.30% |
| 6 | 71.10% | 71.20% | 69.30% | 73.20% |
| 7 | 66.50% | 68.10% | 44.30% | 91.80% |

可以看到，加入 Fall positive mining 后，模型明显更偏向保住 Fall recall，但 ADL specificity 波动较大。

## 严格阈值校准

脚本：

- `reports\strict_eval_v10.py`

输出：

- `reports\v10_strict_validation_threshold_scan.csv`
- `reports\v10_recommended_threshold.json`

阈值扫描只使用 `Subject3 + Zenodo val`。

推荐阈值：

```text
threshold = 0.70
```

验证集最优结果：

| Threshold | Accuracy | Balanced Accuracy | ADL Specificity | Fall Recall | Precision |
|---:|---:|---:|---:|---:|---:|
| 0.70 | 72.13% | 71.99% | 63.33% | 80.65% | 69.44% |

说明：

- 这是扫描区间 `0.35-0.70` 内的最佳点
- 选择策略为：先看 balanced accuracy，再看 ADL specificity，最后取更高阈值

## Subject4 最终独立测试

输出：

- `reports\v10_subject4_final_video_metrics.csv`
- `reports\v10_subject4_final_metrics.json`

最终结果：

| 版本 | Threshold | Accuracy | Balanced Accuracy | ADL Specificity | Fall Recall | Precision | TP/TN/FP/FN |
|---|---:|---:|---:|---:|---:|---:|---|
| v10 fall positive mining | 0.70 | 67.57% | 68.68% | 55.00% | 82.35% | 60.87% | 14 / 11 / 9 / 3 |

误报 ADL：

```text
ADL 05, 06, 07, 08, 11, 12, 15, 16, 17
```

漏检 Fall：

```text
Fall 01, 02, 13
```

## 与 v8 final 对照

这里的对照基线采用最终保存版本：

- `F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8\models\automatic_fall_event_detector_cnn_snn_v8_final_p006_penalty006.pth`
- 配置记录：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8\models\v8_final_config_p006.json`

v8 final（p006）：

- Threshold：0.70
- Accuracy：78.38%
- Balanced Accuracy：78.24%
- ADL Specificity：80.00%
- Fall Recall：76.47%
- Precision：76.47%

v10：

- Threshold：0.70
- Accuracy：67.57%
- Balanced Accuracy：68.68%
- ADL Specificity：55.00%
- Fall Recall：82.35%
- Precision：60.87%

对照结论：

- v10 的 Fall recall 比 v8 final 高 1 个视频
- 但 ADL 误报明显更多，整体准确率、balanced accuracy、precision 都低于 v8 final
- 所以 `Fall event positive mining` 单独加入后，确实把模型推向了“更激进保 Fall”的方向，但没有超过当前最佳主线

如果目标优先级是：

- 更少漏检 Fall：v10 说明这条思路有效
- 论文主结果更稳：v8 final 仍然更适合作为主基线

## 当前判断

`Fall event positive mining` 这个方向不是无效，而是它把模型进一步推向了 “宁可多报，也尽量不要漏掉跌倒”。

它说明：

- 模型确实能利用跌倒瞬间的动态信号
- 但仅靠正样本强化，还不足以同时把 ADL 误报压住

因此如果继续沿这条线做，下一步更合理的方向不是继续单独加大 Fall positive mining，而是：

- 把 Fall positive mining 和更有针对性的 ADL 抑制策略配合起来
- 或者只对最明确的 Fall 事件窗口做更窄范围强化，避免把“接近跌倒的 ADL”也一起推高
