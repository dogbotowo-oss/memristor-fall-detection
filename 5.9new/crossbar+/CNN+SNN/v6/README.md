# v6：Channel-wise SNN Fusion Gate + 保守阈值并列策略

## 版本目的

v6 在 CNN+SNN v5 的基础上继续优化融合方式。v5 使用的是 scalar fusion gate，即对整个 SNN 分支给一个总权重；v6 改为 channel-wise gate，让 SNN 动态特征的每个通道分别学习融合权重。

本版本同时在严格评估脚本中加入更保守的阈值并列选择策略，目的是在验证集 balanced accuracy 不下降的前提下，优先减少 ADL 被误报为 Fall 的风险。

## 关键代码变化

1. 开启 SNN temporal branch。
2. 开启 learnable SNN dynamics，即 SNN 分支中的 decay / threshold 可学习。
3. 开启 SNN fusion gate。
4. fusion gate 模式从 v5 的 scalar 改为 channel。
5. 严格评估时使用保守阈值并列策略：
   - 优先选择验证集 balanced accuracy 最高的阈值。
   - 如果并列，选择 ADL specificity 更高的阈值。
   - 如果仍并列，选择更高阈值，降低 ADL 误报倾向。

## Channel-wise Gate 逻辑

v5：

```text
gate shape = [batch, 1]
snn_pooled = snn_pooled * gate
```

v6：

```text
gate shape = [batch, snn_hidden_dim * 3]
snn_pooled = snn_pooled * gate
```

含义是：不再用一个总开关控制整个 SNN 分支，而是让每个 SNN 动态通道都有自己的融合权重。如果某些动态通道有利于识别 Fall，可以被增强；如果某些通道容易把 ADL 误报成 Fall，可以被压低。

## 运行环境

- Python 环境：`F:\ANACONDA\envs\my_yizu_3.10\python.exe`
- 项目根目录：`F:\12.18-2`
- 器件数据目录：`F:\12.18-2\data`

注意：本项目涉及 Crossbar / 器件动态数据，训练命令必须显式传入：

```text
--data-dir F:\12.18-2\data
```

本次日志已确认读取到 conductance data 和 device dynamics，没有使用 fallback。

## 数据划分

- 训练集：GMDCSA Subject1-2 + Zenodo train
- 验证/阈值校准：GMDCSA Subject3 + Zenodo val
- 最终测试：GMDCSA Subject4

Subject4 只用于最终一次独立测试，没有参与阈值扫描、采样权重选择或模型参数调整。

## 训练关键参数

```text
--use-snn-temporal-branch
--use-snn-fusion-gate
--snn-fusion-gate-mode channel
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

训练窗口数量：

```text
train windows: 2508, normal 1031, fall 1477
val windows: 564, normal 197, fall 367
```

## 训练过程结果

| Epoch | Val Acc | Val Balanced Acc | Val ADL Specificity | Val Fall Recall |
|---|---:|---:|---:|---:|
| 1 | 70.20% | 66.10% | 52.30% | 79.80% |
| 2 | 72.20% | 67.60% | 52.30% | 82.80% |
| 3 | 60.30% | 64.50% | 78.70% | 50.40% |
| 4 | 53.20% | 62.40% | 92.90% | 31.90% |
| 5 | 57.60% | 64.60% | 87.80% | 41.40% |
| 6 | 59.90% | 62.20% | 69.50% | 54.80% |
| 7 | 54.40% | 59.70% | 77.20% | 42.20% |

第 7 个 epoch 触发 early stopping。

## 严格阈值校准结果

阈值扫描只使用 `Subject3 + Zenodo val`。推荐阈值：

```text
threshold = 0.50
```

验证集扫描中，0.50 的 balanced accuracy 最高，因此没有使用 Subject4 参与阈值选择。

## Subject4 最终独立测试结果

| 指标 | 结果 |
|---|---:|
| Accuracy | 70.27% |
| Balanced Accuracy | 69.41% |
| ADL Specificity | 80.00% |
| Fall Recall | 58.82% |
| Precision | 71.43% |
| TP / TN / FP / FN | 10 / 16 / 4 / 7 |

误报 ADL 视频：

```text
ADL 05, 06, 07, 11
```

漏检 Fall 视频：

```text
Fall 01, 02, 05, 08, 09, 13, 17
```

## 文件输出

- 模型：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v6\models\automatic_fall_event_detector_cnn_snn_v6_channel_gate.pth`
- 训练曲线：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v6\image\training_curve_cnn_snn_v6_channel_gate.png`
- 阈值扫描：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v6\reports\v6_strict_validation_threshold_scan.csv`
- 推荐阈值：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v6\reports\v6_recommended_threshold.json`
- Subject4 结果：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v6\reports\v6_subject4_final_metrics.json`

## 当前结论

v6 的 channel-wise gate 保持了较高 ADL specificity，Subject4 ADL specificity 为 80.00%，但 Fall recall 为 58.82%，仍未超过此前无 SNN 的强基线。说明当前 SNN 动态分支虽然能提供时序动态信息，但融合后对 GMDCSA Subject4 的 Fall 召回提升还不稳定。

下一步如果继续沿 CNN+SNN 方向优化，应优先检查：

1. SNN 动态特征是否真正覆盖 Fall 落下瞬间，而不是只增强了整体运动幅度。
2. gate 是否过度压制了 SNN 分支，导致动态信息贡献不足。
3. 是否需要把 Fall event positive mining 和 SNN temporal branch 更紧密结合，让 SNN 分支重点学习跌倒瞬间的快速状态变化。
