# v8 二维小扫描记录

## 扫描目的

本次扫描验证两个训练侧参数的交互关系：

- 横轴：`snn_adl_gate_penalty`
- 纵轴：`hard_negative_normal_weight`

扫描只使用 `Subject3 + Zenodo val` 做验证和阈值选择，没有使用 Subject4 调参。

## 固定设置

- Python 环境：`F:\ANACONDA\envs\my_yizu_3.10\python.exe`
- 主代码：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8\code\automatic_fall_detector.py`
- 初始化模型：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v7\models\automatic_fall_event_detector_cnn_snn_v7_group_channel_gate.pth`
- 器件数据：`F:\12.18-2\data`
- 训练集：GMDCSA Subject1-2 + Zenodo train
- 验证集：GMDCSA Subject3 + Zenodo val
- `external-labeled-stride=16`
- `image-size=64`
- `snn_event_score_threshold=0.55`
- `snn_event_score_sharpness=10.0`
- `hard_negative_min_percentile=0.88`
- `hard_negative_score_gamma=2.00`

训练日志均确认：

```text
external_labeled_windows=3539
external_val_labeled_windows=1076
Loaded conductance data: HRS=9164, LRS=10001
Loaded device dynamics: epsc=EPSC-0.001-60-60.xls, ppf=PPF-0.001-60-60.xls, ltp=LTP-0.001-1MS.xls, ltd=LTD-0.02-1MS.xls
```

## 扫描结果

| tag | penalty | hard-negative weight | 推荐阈值 | Accuracy | Balanced Acc | ADL Specificity | Fall Recall | Precision | TP/TN/FP/FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| p004_w080 | 0.04 | 0.80 | 0.50 | 72.13% | 71.83% | 53.33% | 90.32% | 66.67% | 28/16/14/3 |
| p006_w080 | 0.06 | 0.80 | 0.45 | 72.13% | 71.83% | 53.33% | 90.32% | 66.67% | 28/16/14/3 |
| p008_w080 | 0.08 | 0.80 | 0.45 | 72.13% | 71.83% | 53.33% | 90.32% | 66.67% | 28/16/14/3 |
| p004_w100 | 0.04 | 1.00 | 0.70 | 72.13% | 71.99% | 63.33% | 80.65% | 69.44% | 25/19/11/6 |
| p006_w100 | 0.06 | 1.00 | 0.70 | 72.13% | 71.99% | 63.33% | 80.65% | 69.44% | 25/19/11/6 |
| p008_w100 | 0.08 | 1.00 | 0.70 | 72.13% | 71.99% | 63.33% | 80.65% | 69.44% | 25/19/11/6 |
| p004_w110 | 0.04 | 1.10 | 0.65 | 72.13% | 71.99% | 63.33% | 80.65% | 69.44% | 25/19/11/6 |
| p006_w110 | 0.06 | 1.10 | 0.70 | 72.13% | 72.04% | 66.67% | 77.42% | 70.59% | 24/20/10/7 |
| p008_w110 | 0.08 | 1.10 | 0.70 | 72.13% | 72.04% | 66.67% | 77.42% | 70.59% | 24/20/10/7 |

## 结论

1. `hard_negative_normal_weight` 是本次二维扫描的主导变量。
2. `weight=0.80` 会明显放开 Fall，Fall recall 达到 90.32%，但 ADL specificity 只有 53.33%，误报偏多。
3. `weight=1.00` 是中间折中点，ADL specificity 回到 63.33%，Fall recall 为 80.65%。
4. `weight=1.10` 更保守，其中 `penalty=0.06/0.08` 把 ADL specificity 提到 66.67%，precision 提到 70.59%，但 Fall recall 降到 77.42%。
5. 在 `penalty=0.04-0.08` 范围内，同一 weight 行内结果高度一致，说明 gate penalty 在这个小范围内不是主要瓶颈。

因此，当前 v8 final 的 `snn_adl_gate_penalty=0.06`、`hard_negative_normal_weight=1.10` 仍然是验证集上的保守优选点。如果后续目标是提高 Fall recall，可以优先尝试 `hard_negative_normal_weight=1.00`，而不是继续扩大 gate penalty。

## 输出文件

- 扫描脚本：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8\reports\snn_adl_gate_penalty_hnweight_scan_2d\run_2d_scan.py`
- 汇总表：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8\reports\snn_adl_gate_penalty_hnweight_scan_2d\strict_validation_2d_scan_summary.csv`
- 各组合模型、训练曲线和验证明细：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8\reports\snn_adl_gate_penalty_hnweight_scan_2d`
