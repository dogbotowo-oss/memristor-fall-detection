# v7：Fall-event aware group + channel SNN fusion gate

## 版本目标

v7 在 v6 的基础上继续优化 SNN 融合方式，核心是让 gate 不仅看 CNN / SNN 自身特征，还显式接收 `fall_event_score`，使其更容易识别“快速下落、重心突变、运动能量突增”的窗口。

同时，v7 把 SNN 输出进一步拆成“组级 + 通道级”两层权重：

- 组级：对应 `mean spike / max spike / last spike` 三个子块
- 通道级：对应每个子块里的单独通道

这样可以兼顾“哪些时间统计更重要”和“哪些动态通道更重要”。

## 关键变化

1. `fall_event_score` 显式输入 gate。
2. 新增 `group_channel` 融合模式。
3. gate 逻辑变成：
   - 先学三组权重
   - 再学每个通道权重
   - 最后两者相乘
4. 保留 v6 的 learnable SNN dynamics、Crossbar 器件噪声、teacher-student noise、严格阈值策略。

## 融合逻辑

SNN 输出仍然是 3 组拼接：

- `spike_train.mean(dim=1)`
- `spike_train.max(dim=1).values`
- `spike_train[:, -1, :]`

即总长度为 `snn_hidden_dim * 3`。

v7 的 gate 输入改为：

```text
[ CNN pooled features | SNN pooled features | fall_event_score ]
```

然后：

- `group_gate` 输出 3 个值
- `channel_gate` 输出 `snn_hidden_dim * 3` 个值
- 最终门控为 `group_gate_repeat * channel_gate`

## 推荐用法

训练时建议：

```text
--use-snn-temporal-branch
--use-snn-fusion-gate
--snn-fusion-gate-mode group_channel
--learnable-snn-dynamics
--freeze-cnn-gru-epochs 2
```

其余 Crossbar / noise / hard-negative 参数先沿用 v6 的稳定配置，再做小范围验证。

## 环境约束

- Python：`F:\ANACONDA\envs\my_yizu_3.10\python.exe`
- 数据目录：`F:\12.18-2\data`
- Subject4 只做最终测试，不参与调参

## 本次训练设置

本次训练使用 v6 checkpoint 初始化，新增的 `group_gate` 和 `channel_gate` 为 v7 新参数，其余可兼容权重从 v6 继承。

训练命令核心开关：

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
--external-labeled-stride 16
--adl-context-path-keyword zenodo_falldb_video_split/train/adl
--adl-context-min-percentile 0.95
--adl-context-extra-copies 1
--image-size 64
```

数据划分：

- 训练：GMDCSA Subject1-2 + Zenodo train
- 验证/阈值校准：GMDCSA Subject3 + Zenodo val
- 最终测试：GMDCSA Subject4

本次训练窗口数量：

```text
train windows: 3539, normal 1758, fall 1781
val windows: 1076, normal 573, fall 503
```

日志确认已加载：

```text
Loaded conductance data: HRS=9164, LRS=10001
Loaded device dynamics: epsc=EPSC-0.001-60-60.xls, ppf=PPF-0.001-60-60.xls, ltp=LTP-0.001-1MS.xls, ltd=LTD-0.02-1MS.xls
```

## 训练过程结果

| Epoch | Val Acc | Val Balanced Acc | Val ADL Specificity | Val Fall Recall |
|---|---:|---:|---:|---:|
| 1 | 71.70% | 70.60% | 86.40% | 54.90% |
| 2 | 73.20% | 73.50% | 70.00% | 76.90% |
| 3 | 71.00% | 72.10% | 54.80% | 89.50% |
| 4 | 70.30% | 71.30% | 55.10% | 87.50% |
| 5 | 67.90% | 69.60% | 44.50% | 94.60% |
| 6 | 69.50% | 69.90% | 64.00% | 75.70% |
| 7 | 73.10% | 73.70% | 65.60% | 81.70% |
| 8 | 66.90% | 67.20% | 63.40% | 71.00% |
| 9 | 69.40% | 69.40% | 69.30% | 69.60% |
| 10 | 69.90% | 70.30% | 63.40% | 77.30% |
| 11 | 70.10% | 69.90% | 73.10% | 66.60% |
| 12 | 69.00% | 69.20% | 65.10% | 73.40% |

第 12 个 epoch 触发 early stopping。验证集中最均衡的是第 7 轮附近，说明 v7 确实把模型往 Fall recall 方向拉动，但 ADL specificity 比 v6 更低。

## 严格阈值校准

阈值扫描只使用 `Subject3 + Zenodo val`，没有使用 Subject4。推荐阈值：

```text
threshold = 0.65
```

验证集最高 balanced accuracy：

| Threshold | Accuracy | Balanced Accuracy | ADL Specificity | Fall Recall | Precision |
|---:|---:|---:|---:|---:|---:|
| 0.65 | 67.21% | 67.04% | 56.67% | 77.42% | 64.86% |

## Subject4 最终独立测试结果

| 指标 | 结果 |
|---|---:|
| Accuracy | 67.57% |
| Balanced Accuracy | 68.68% |
| ADL Specificity | 55.00% |
| Fall Recall | 82.35% |
| Precision | 60.87% |
| TP / TN / FP / FN | 14 / 11 / 9 / 3 |

误报 ADL 视频：

```text
ADL 02, 05, 06, 07, 11, 12, 15, 16, 17
```

漏检 Fall 视频：

```text
Fall 01, 02, 13
```

## 文件输出

- 模型：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v7\models\automatic_fall_event_detector_cnn_snn_v7_group_channel_gate.pth`
- 训练曲线：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v7\image\training_curve_cnn_snn_v7_group_channel.png`
- 阈值扫描：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v7\reports\v7_strict_validation_threshold_scan.csv`
- 推荐阈值：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v7\reports\v7_recommended_threshold.json`
- Subject4 结果：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v7\reports\v7_subject4_final_metrics.json`

## 当前结论

v7 的 `fall_event_score + group-wise + channel-wise gate` 明显增强了 Fall 召回能力。Subject4 的 Fall recall 从 v6 的 58.82% 提升到 82.35%，漏检从 7 个降到 3 个。

但代价是 ADL specificity 从 v6 的 80.00% 降到 55.00%，误报从 4 个增加到 9 个。因此 v7 证明了该门控方向能够辅助 Fall，但目前偏激进，下一步应重点控制 ADL 误报。

建议下一版优先考虑：

1. 给 `fall_event_score` 加更保守的归一化或阈值门槛，避免普通 ADL 动作也放大 SNN。
2. 对 group gate 加轻微正则，让 `max spike` 或 `last spike` 不能长期过强。
3. 在不使用 Subject4 调参的前提下，只在 `Subject3 + Zenodo val` 上扫描 `gate bias / gate temperature / min gate`。
