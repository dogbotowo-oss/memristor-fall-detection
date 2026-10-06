# Ordered Residual 三组消融结果

## 统一口径

- 训练集：GMDCSA Subject 1-2 + Zenodo train，共 3539 个窗口
- 验证集：GMDCSA Subject 3 + Zenodo val，共 1076 个窗口
- Crossbar 与 device dynamics：启用
- CNN/GRU：按本轮统一训练设置冻结
- 学习率：`2e-4`
- ADL penalty：`0.0`
- Subject4：未使用

## 结果

| 结构 | 最佳 epoch | Balanced Accuracy | ADL specificity | Fall recall |
|---|---:|---:|---:|---:|
| group-only | 1 | 74.1% | 72.9% | 75.3% |
| group-only + ordered residual | 5 | 74.0% | 67.9% | 80.1% |
| group-channel + ordered residual | 5 | 74.0% | 67.9% | 80.1% |

## 结论

两种 ordered residual 都成功参与了训练，checkpoint 中 residual 权重已从零发生更新，但没有改善综合指标。相对于 group-only，Fall recall 提高 4.8 个百分点，却使 ADL specificity 降低 5.0 个百分点；相对于 group-only + ordered residual，增加 channel gate 没有带来任何额外收益。

因此当前瓶颈不是“是否增加 channel 数量”，而是 ordered temporal evidence 与 ADL 的边界混淆。下一步应停止继续增加 channel gate，转向更明确的 ADL hard-negative temporal supervision 或基于视频级事件的 gate 校准。
