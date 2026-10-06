# Centered Channel Gate 结果

验证集：GMDCSA Subject3 + Zenodo val。Subject4 未参与调参。

## 最佳结果

- Best epoch：1
- Accuracy：74.1%
- Balanced Accuracy：74.1%
- ADL specificity：72.9%
- Fall recall：75.3%

该结果与 group-only 持平，没有产生额外指标提升。

## Gate 学习情况

- 24 个高权重通道：1.25 -> 平均 1.2518，范围 1.2442-1.2607。
- 12 个低权重通道：0.75 -> 平均 0.7537，范围 0.7486-0.7613。
- 60 个中性通道：1.00 -> 平均 0.9986，范围 0.9849-1.0125。

梯度放大 5 倍后，channel prior 确实发生了更新，但更新幅度仍小，并且没有改变最佳验证指标。这说明当前分类头主要依赖 CNN/GRU 与 group-level SNN 信息，静态 channel prior 的边际贡献有限。

当前继续保留 group-only 作为最简单、综合最优结构。不建议继续扫描 channel 权重或通道数量。
