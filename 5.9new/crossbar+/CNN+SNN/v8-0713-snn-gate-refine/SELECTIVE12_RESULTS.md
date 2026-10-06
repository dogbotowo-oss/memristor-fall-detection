# Selective-negative-12 结果

验证集：GMDCSA Subject3 + Zenodo val。Subject4 未参与调参。

## 最佳结果

- Best epoch：1
- Accuracy：74.0%
- Balanced Accuracy：74.0%
- ADL specificity：72.9%
- Fall recall：75.1%

## 对比

- group-only：74.1 / 74.1 / 72.9 / 75.3%
- selective-negative-12：74.0 / 74.0 / 72.9 / 75.1%

顺序均为 Accuracy / Balanced Accuracy / ADL specificity / Fall recall。

Selective-12 没有超过 group-only。12 个通道的 scale 从初始 0.75 最终仅变化到约 0.750-0.753，说明分类损失对这 12 个全局 gate 参数提供的梯度很弱。继续更换 12/16/24 个通道意义有限。

当前保留 group-only 为验证集综合最优结构。如果继续研究 channel gate，应改用输入相关的 centered residual channel gate，并为 gate 设置独立学习率，而不是继续扫描通道数量。
