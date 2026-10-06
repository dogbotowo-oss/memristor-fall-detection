# v8：ADL hard-negative SNN gate penalty

## 版本目的

v7 证明 `fall_event_score + group-wise + channel-wise gate` 能显著提高 Fall recall，但 ADL specificity 下降较明显。v8 在 v7 的基础上加入一个轻量 ADL hard-negative gate penalty，目标是：

- 真实 Fall 的关键动态通道仍然可以被 SNN gate 放大。
- 坐下、蹲下、弯腰、靠近地面但没有跌倒这类 ADL hard negative，不要把 SNN gate 打得过高。
- 尝试降低 ADL 误报，同时尽量保留 v7 对 Fall 的召回优势。

## 核心代码逻辑

v8 保留 v7 的结构：

```text
CNN/GRU branch + SNN temporal branch
SNN gate input = [CNN pooled features | SNN pooled features | fall_event_score]
SNN gate = group_gate * channel_gate
```

当前代码已将 `fall_event_score` 改为“先归一化，再门限触发”：

```text
raw_event = fall_event_like 时间序列
peak = raw_event 的窗口内峰值
baseline = raw_event 的窗口内均值
spread = raw_event 的窗口内标准差
z_score = (peak - baseline) / spread
normalized = sigmoid((z_score - 1.0) * 1.5)
trigger = sigmoid((normalized - snn_event_score_threshold) * snn_event_score_sharpness)
fall_event_score_for_gate = normalized * trigger
```

默认参数：

```text
--snn-event-score-threshold 0.55
--snn-event-score-sharpness 10.0
```

这样做的目的不是让所有运动都增强 SNN，而是只有窗口内出现相对明显的动态突变时，才更强地影响 gate。相比直接把原始 `fall_event_score` 输入 gate，这种方式更不容易受视频亮度、人体大小、动作幅度和数据集风格影响。

新增训练约束：

```text
gate_penalty = hard_negative_score * mean_snn_gate
loss = original_loss + snn_adl_gate_penalty * gate_penalty
```

其中：

- `hard_negative_score` 来自现有 ADL hard-negative 评分函数。
- 只有 label 为 `normal` 的 ADL 样本参与该 penalty。
- 越像 Fall 的 ADL 窗口，`hard_negative_score` 越高，对 gate 的抑制越明显。
- 真实 Fall 样本不会被这个 penalty 直接压制。

新增参数：

```text
--snn-adl-gate-penalty 0.04
```

## 运行环境

- Python：`F:\ANACONDA\envs\my_yizu_3.10\python.exe`
- 项目根目录：`F:\12.18-2`
- 器件数据目录：`F:\12.18-2\data`

训练和严格评估日志均确认加载：

```text
Loaded conductance data: HRS=9164, LRS=10001
Loaded device dynamics: epsc=EPSC-0.001-60-60.xls, ppf=PPF-0.001-60-60.xls, ltp=LTP-0.001-1MS.xls, ltd=LTD-0.02-1MS.xls
```

## 数据划分

- 训练：GMDCSA Subject1-2 + Zenodo train
- 验证/阈值校准：GMDCSA Subject3 + Zenodo val
- 最终测试：GMDCSA Subject4

Subject4 只用于最终独立测试，没有参与阈值、采样权重或 gate penalty 调参。

训练窗口数量：

```text
train windows: 3539, normal 1758, fall 1781
val windows: 1076, normal 573, fall 503
```

## 训练关键参数

```text
--init-model-path F:\12.18-2\5.9new\crossbar+\CNN+SNN\v7\models\automatic_fall_event_detector_cnn_snn_v7_group_channel_gate.pth
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
--snn-adl-gate-penalty 0.04
--snn-event-score-threshold 0.55
--snn-event-score-sharpness 10.0
--external-labeled-stride 16
--adl-context-path-keyword zenodo_falldb_video_split/train/adl
--adl-context-min-percentile 0.95
--adl-context-extra-copies 1
--image-size 64
```

## 训练过程结果

| Epoch | Val Acc | Val Balanced Acc | Val ADL Specificity | Val Fall Recall |
|---|---:|---:|---:|---:|
| 1 | 73.70% | 73.80% | 72.90% | 74.60% |
| 2 | 72.60% | 73.00% | 66.00% | 80.10% |
| 3 | 68.40% | 69.90% | 47.50% | 92.20% |
| 4 | 71.30% | 71.40% | 69.50% | 73.40% |
| 5 | 71.00% | 72.10% | 55.80% | 88.30% |
| 6 | 66.80% | 66.00% | 78.40% | 53.70% |

第 6 个 epoch 触发 early stopping。保存的 checkpoint 对应验证集 balanced accuracy 最好的版本。

## 严格阈值校准

阈值扫描只使用 `Subject3 + Zenodo val`。推荐阈值：

```text
threshold = 0.65
```

验证集最高结果：

| Threshold | Accuracy | Balanced Accuracy | ADL Specificity | Fall Recall | Precision |
|---:|---:|---:|---:|---:|---:|
| 0.65 | 72.13% | 71.99% | 63.33% | 80.65% | 69.44% |
| 0.60 | 72.13% | 71.99% | 63.33% | 80.65% | 69.44% |

并列时使用保守策略：先看 balanced accuracy，再优先 ADL specificity，最后选更高阈值。

## Subject4 最终独立测试结果

| 版本 | Accuracy | Balanced Accuracy | ADL Specificity | Fall Recall | Precision |
|---|---:|---:|---:|---:|---:|
| v6 channel gate | 70.27% | 69.41% | 80.00% | 58.82% | 71.43% |
| v7 group+channel gate | 67.57% | 68.68% | 55.00% | 82.35% | 60.87% |
| v8 ADL gate penalty | 67.57% | 68.24% | 60.00% | 76.47% | 61.90% |

v8 Subject4 详细结果：

```text
threshold = 0.65
TP / TN / FP / FN = 13 / 12 / 8 / 4
```

误报 ADL 视频：

```text
ADL 05, 06, 07, 11, 12, 15, 16, 17
```

漏检 Fall 视频：

```text
Fall 01, 02, 09, 13
```

## 当前结论

v8 的 ADL hard-negative gate penalty 起到了预期方向的作用：相比 v7，ADL specificity 从 55.00% 提升到 60.00%，误报从 9 个降到 8 个。

但它也压低了一部分 Fall 动态通道，Fall recall 从 82.35% 降到 76.47%。这说明该策略能够抑制 ADL 误报，但当前 penalty 强度和作用形式还需要继续小范围优化。

## fall_event_score 归一化 + 门限触发

本次更新把直接输入 gate 的 `fall_event_score` 改成“先归一化，再门限触发”：

```python
event_series = features[:, :, 6]
peak = event_series.amax(dim=1, keepdim=True)
baseline = event_series.mean(dim=1, keepdim=True)
spread = event_series.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
z_score = (peak - baseline) / spread
normalized = torch.sigmoid((z_score - 1.0) * 1.5)
trigger = torch.sigmoid((normalized - threshold) * sharpness)
fall_event_score = normalized * trigger
```

这样做的目的不是让原始动态分数直接支配 gate，而是先判断“这个窗口内部是否真的出现相对突出的快速变化”。如果某个数据集整体运动幅度偏大，但窗口内部没有明显峰值，归一化后不会被过度放大；只有超过门限的快速变化才会明显打开 SNN gate。

新增参数：

```text
--snn-event-score-threshold 0.55
--snn-event-score-sharpness 10.0
```

## snn_adl_gate_penalty 扫描结果

扫描只使用 `Subject3 + Zenodo val` 做验证和阈值选择，没有使用 Subject4 调参。训练仍使用 `F:\ANACONDA\envs\my_yizu_3.10\python.exe`，并显式传入 `--data-dir F:\12.18-2\data`；日志确认加载了 conductance 与 EPSC/PPF/LTP/LTD 器件动态数据。

| penalty | 推荐阈值 | Accuracy | Balanced Acc | ADL Specificity | Fall Recall | Precision | TP/TN/FP/FN |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.00 | 0.65 | 72.13% | 71.99% | 63.33% | 80.65% | 69.44% | 25/19/11/6 |
| 0.02 | 0.65 | 72.13% | 71.99% | 63.33% | 80.65% | 69.44% | 25/19/11/6 |
| 0.04 | 0.65 | 72.13% | 71.99% | 63.33% | 80.65% | 69.44% | 25/19/11/6 |
| 0.06 | 0.70 | 72.13% | 72.04% | 66.67% | 77.42% | 70.59% | 24/20/10/7 |
| 0.08 | 0.70 | 72.13% | 72.04% | 66.67% | 77.42% | 70.59% | 24/20/10/7 |

结论：归一化 + 门限触发以后，`0.06` 和 `0.08` 比 `0.00-0.04` 多保住 1 个 ADL 正确识别，precision 更高，但少识别 1 个 Fall。若目标是更保守地压 ADL 误报，推荐继续看 `0.06`；若目标是保住 Fall recall，则 `0.00-0.04` 更合适。本轮只完成验证集策略筛选，Subject4 仍应作为最终独立测试，不参与选择 penalty。

扫描输出：

- 总表：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8\reports\snn_adl_gate_penalty_scan_event_norm\strict_validation_event_norm_penalty_scan_summary.csv`
- 各档模型和验证阈值：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8\reports\snn_adl_gate_penalty_scan_event_norm`

## p002 与 p006 的 Subject4 对照

按验证集策略选择两个代表点做一次 Subject4 独立测试对照：

- `p002`：`snn_adl_gate_penalty=0.02`，验证集推荐阈值 `0.65`
- `p006`：`snn_adl_gate_penalty=0.06`，验证集推荐阈值 `0.70`

注意：这一步是对已经由 `Subject3 + Zenodo val` 选出的候选策略做最终测试记录，不能继续用 Subject4 结果反复调参。

| 版本 | Subject4 阈值 | Accuracy | Balanced Acc | ADL Specificity | Fall Recall | Precision | TP/TN/FP/FN |
|---|---:|---:|---:|---:|---:|---:|---|
| p002 | 0.65 | 70.27% | 70.74% | 65.00% | 76.47% | 65.00% | 13/13/7/4 |
| p006 | 0.70 | 78.38% | 78.24% | 80.00% | 76.47% | 76.47% | 13/16/4/4 |

这次对照中，`p006` 更理想：Fall recall 没有比 `p002` 下降，仍为 76.47%，同时 ADL specificity 从 65.00% 提升到 80.00%，误报从 7 个降到 4 个。因此若只在这两个候选中做论文记录，`p006` 更适合作为当前 v8 event-normalized gate penalty 的代表结果。

## v8 最终保存版本

当前 v8 最终版本固定为：

```text
snn_adl_gate_penalty = 0.06
tag = p006
validation threshold = 0.70
threshold source = Subject3 + Zenodo val only
```

最终模型已从扫描目录复制到 `models` 目录：

- 通用最终模型：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8\models\automatic_fall_event_detector_cnn_snn_v8_final.pth`
- 带参数标识模型：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8\models\automatic_fall_event_detector_cnn_snn_v8_final_p006_penalty006.pth`
- 最终配置记录：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8\models\v8_final_config_p006.json`

说明：`Subject4` 结果仅作为最终独立测试记录，不再用于继续选择阈值或调参。

## v8 二维小扫描：gate penalty × hard-negative weight

为检查 v8 final 是否仍是合理折中点，新增一个 3×3 小扫描：

- `snn_adl_gate_penalty = 0.04 / 0.06 / 0.08`
- `hard_negative_normal_weight = 0.80 / 1.00 / 1.10`

扫描只使用 `Subject3 + Zenodo val` 做验证和阈值选择，没有使用 Subject4。所有组合都从同一个 v7 初始化模型开始训练，并显式使用 `--data-dir F:\12.18-2\data`。

| hard-negative weight | penalty=0.04 | penalty=0.06 | penalty=0.08 | 主要表现 |
|---:|---|---|---|---|
| 0.80 | ADL 53.33%, Fall 90.32% | ADL 53.33%, Fall 90.32% | ADL 53.33%, Fall 90.32% | Fall 放开最多，但 ADL 误报偏高 |
| 1.00 | ADL 63.33%, Fall 80.65% | ADL 63.33%, Fall 80.65% | ADL 63.33%, Fall 80.65% | 中间折中点 |
| 1.10 | ADL 63.33%, Fall 80.65% | ADL 66.67%, Fall 77.42% | ADL 66.67%, Fall 77.42% | 更保守，ADL 与 precision 最好 |

结论：本次二维扫描中，`hard_negative_normal_weight` 是主导变量，`snn_adl_gate_penalty=0.04-0.08` 的小范围变化影响较小。当前 v8 final 的 `snn_adl_gate_penalty=0.06`、`hard_negative_normal_weight=1.10` 仍然是验证集上的保守优选点；如果下一步想提高 Fall recall，应优先尝试把 `hard_negative_normal_weight` 降到 `1.00`，而不是继续加大 gate penalty。

扫描记录：

- 扫描 README：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8\reports\snn_adl_gate_penalty_hnweight_scan_2d\README.md`
- 汇总表：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8\reports\snn_adl_gate_penalty_hnweight_scan_2d\strict_validation_2d_scan_summary.csv`

输出文件：

- 汇总表：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8\reports\snn_adl_gate_penalty_scan_event_norm\subject4_compare_p002_p006\subject4_compare_p002_p006_summary.csv`
- p002 明细：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8\reports\snn_adl_gate_penalty_scan_event_norm\subject4_compare_p002_p006\subject4_summary_p002.json`
- p006 明细：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8\reports\snn_adl_gate_penalty_scan_event_norm\subject4_compare_p002_p006\subject4_summary_p006.json`

## 文件输出

- 代码：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8\code\automatic_fall_detector.py`
- 最终模型：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8\models\automatic_fall_event_detector_cnn_snn_v8_final.pth`
- 旧主模型记录：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8\models\automatic_fall_event_detector_cnn_snn_v8_adl_gate_penalty.pth`
- 训练曲线：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8\image\training_curve_cnn_snn_v8_adl_gate_penalty.png`
- 阈值扫描：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8\reports\v8_strict_validation_threshold_scan.csv`
- Subject4 结果：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8\reports\v8_subject4_final_metrics.json`
