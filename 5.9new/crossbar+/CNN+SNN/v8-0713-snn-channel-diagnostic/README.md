# v8-0713 SNN 通道诊断

## 目的

本版本承接 `v8-0707` 的 SNN gate / ADL penalty 消融结果。现有结果已经证明：

- SNN temporal branch 本身能够提升 Fall recall。
- `snn_adl_gate_penalty=0.06` 与关闭 penalty 的结果近似，penalty 不是当前主要瓶颈。
- `group_channel` 的 96 个通道 gate 更可能造成部分 Fall 动态通道被过度压制。

本目录暂不训练新模型，先执行一次低成本通道诊断。

## 严格数据口径

- 通道排序：GMDCSA Subject1-2 + Zenodo train。
- 掩码比较：GMDCSA Subject3 + Zenodo val。
- Subject4：完全不参与通道排序、掩码选择或结构选择。

## 输入模型

使用 `v8-0707` 的 `group_channel + penalty=0` checkpoint：

`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8-0707\models\snn_group_channel_gate_p000\automatic_fall_event_detector_snn_group_channel_gate_p000.pth`

## 输出

- `reports/channel_importance.csv`：96 个通道的 gate、贡献和区分度。
- `reports/group_summary.csv`：mean/max/last 三组统计。
- `reports/mask_validation_scan.csv`：Top-24/48/72 与各组组合的免训练验证结果。
- `reports/recommended_masks.json`：Top-K 通道索引。
- `reports/diagnostic_summary.json`：诊断摘要及推荐掩码。

掩码扫描只用于决定下一步训练结构，不作为最终模型结果。
