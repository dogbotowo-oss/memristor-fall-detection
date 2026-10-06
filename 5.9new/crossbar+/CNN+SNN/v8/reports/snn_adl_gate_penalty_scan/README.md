# snn_adl_gate_penalty 扫描记录

## 扫描目的

本轮只扫描 `--snn-adl-gate-penalty`，用于判断 ADL hard-negative gate 抑制项是否能提高 ADL specificity，同时尽量保留 Fall recall。

## 数据纪律

- 训练：GMDCSA Subject1-2 + Zenodo train
- 验证/阈值校准：GMDCSA Subject3 + Zenodo val
- 本轮扫描没有使用 Subject4
- Subject4 仍保留为最终独立测试，不能用于选择 penalty

## group_channel 与 channel 的区别

`channel`：只对 SNN 输出的每个通道学习一个权重。它能判断某个通道该不该增强，但不知道这个通道属于哪一类时间统计。

`group_channel`：SNN 输出先分成三组：`mean spike`、`max spike`、`last spike`。模型先学习 3 个组权重，再学习每个通道权重，最后两者相乘。因此它同时具备组级控制和通道级控制，更适合解释“哪些时序统计对 Fall/ADL 更重要”。

## 严格验证集扫描结果

以下结果只来自 `Subject3 + Zenodo val` 视频级阈值扫描。

| snn_adl_gate_penalty | 推荐阈值 | Accuracy | Balanced Accuracy | ADL Specificity | Fall Recall | Precision | TP/TN/FP/FN |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.00 | 0.70 | 72.13% | 72.04% | 66.67% | 77.42% | 70.59% | 24/20/10/7 |
| 0.02 | 0.65 | 72.13% | 71.99% | 63.33% | 80.65% | 69.44% | 25/19/11/6 |
| 0.04 | 0.65 | 72.13% | 71.99% | 63.33% | 80.65% | 69.44% | 25/19/11/6 |
| 0.06 | 0.70 | 72.13% | 72.04% | 66.67% | 77.42% | 70.59% | 24/20/10/7 |
| 0.08 | 0.70 | 72.13% | 72.04% | 66.67% | 77.42% | 70.59% | 24/20/10/7 |

## 当前判断

从严格验证集看，`0.00 / 0.06 / 0.08` 的 balanced accuracy 略高，并且 ADL specificity 更好；`0.02 / 0.04` 的 Fall recall 更高，但 ADL specificity 略低。

如果目标是整体均衡和控制 ADL 误报，当前更推荐 `0.00` 或 `0.06`。如果目标是优先保 Fall recall，可考虑 `0.02` 或 `0.04`。由于 `0.00` 不引入额外 gate penalty，而且验证表现与 `0.06 / 0.08` 持平，下一步更建议不要继续加大这个 penalty，而是改 penalty 形式，例如只惩罚最高分 ADL hard negative 或只惩罚特定 gate group。

## 输出文件

- 窗口级训练汇总：`snn_adl_gate_penalty_scan_summary.csv`
- 视频级严格验证汇总：`strict_validation_penalty_scan_summary.csv`
- 每组训练日志和模型：`p000`、`p002`、`p004`、`p006`、`p008`
