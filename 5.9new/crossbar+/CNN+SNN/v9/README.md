# v9：hard_negative_normal_weight 降到 1.00 的对照版本

## 版本目的

v9 基于 v8 的二维小扫描结果建立，核心变化是：

```text
snn_adl_gate_penalty = 0.06
hard_negative_normal_weight = 1.00
validation threshold = 0.70
```

其他关键设置保持 v8 一致：

- SNN temporal branch 开启
- group+channel fusion gate 开启
- fall_event_score 采用归一化 + 门限触发
- Crossbar 开启
- `--use-device-dynamics` 开启
- `--crossbar-readout-noise-scale 0.25`
- teacher-student gaussian noise 开启
- `--data-dir F:\12.18-2\data`
- Python 环境：`F:\ANACONDA\envs\my_yizu_3.10\python.exe`

## 数据划分

- 训练集：GMDCSA Subject1-2 + Zenodo train
- 验证/阈值校准：GMDCSA Subject3 + Zenodo val
- 最终独立测试：GMDCSA Subject4

Subject4 只用于最终测试，没有参与阈值或参数选择。

## 来源说明

v9 使用 v8 二维扫描中的 `p006_w100` 模型归档而来：

```text
F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8\reports\snn_adl_gate_penalty_hnweight_scan_2d\p006_w100
```

该模型与 v9 final 模型哈希一致，说明复制无误。

## 验证集结果

验证集来源：`Subject3 + Zenodo val`

| Threshold | Accuracy | Balanced Acc | ADL Specificity | Fall Recall | Precision | TP/TN/FP/FN |
|---:|---:|---:|---:|---:|---:|---|
| 0.70 | 72.13% | 71.99% | 63.33% | 80.65% | 69.44% | 25/19/11/6 |

相比 v8 final 的验证集结果，v9 更偏向保留 Fall recall，但 ADL specificity 较低。

## Subject4 最终独立测试

阈值固定使用验证集选出的 `0.70`。

| Version | Accuracy | Balanced Acc | ADL Specificity | Fall Recall | Precision | TP/TN/FP/FN |
|---|---:|---:|---:|---:|---:|---|
| v8 final | 78.38% | 78.24% | 80.00% | 76.47% | 76.47% | 13/16/4/4 |
| v9 hn_weight=1.00 | 67.57% | 68.24% | 60.00% | 76.47% | 61.90% | 13/12/8/4 |

Subject4 上，v9 没有提高 Fall recall，仍为 76.47%；但 ADL specificity 从 80.00% 降到 60.00%，误报从 4 个增加到 8 个。因此 v9 不应替代 v8 final。

## 错误样本

误报 ADL：

```text
ADL 05, 06, 07, 11, 12, 15, 16, 17
```

漏检 Fall：

```text
Fall 01, 02, 09, 13
```

## 当前结论

`hard_negative_normal_weight=1.00` 在验证集上看起来能保留更高 Fall recall，但在 Subject4 独立测试上没有带来 Fall 提升，反而显著增加 ADL 误报。当前更稳的最终版本仍然是 v8 final：

```text
snn_adl_gate_penalty = 0.06
hard_negative_normal_weight = 1.10
threshold = 0.70
```

## 输出文件

- 代码：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v9\code\automatic_fall_detector.py`
- 最终模型：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v9\models\automatic_fall_event_detector_cnn_snn_v9_final.pth`
- 配置记录：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v9\models\v9_final_config.json`
- 训练曲线：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v9\image\training_curve_v9_hnweight100.png`
- 验证集结果：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v9\reports\v9_strict_validation_summary.json`
- Subject4 结果：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v9\reports\v9_subject4_final_metrics.json`
- Subject4 视频明细：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v9\reports\v9_subject4_final_video_metrics.csv`
