# Group + Centered Channel Prior

本实验让 group gate 继续根据输入动态判断 mean/max/last 三组的重要性，同时让 channel gate 表示稳定的通道可靠性。

## 权重初始化

- 24 个稳定强正通道：初始 scale=1.25。
- 12 个稳定强负通道：初始 scale=0.75。
- 其余 60 个通道：初始 scale=1.00。

通道公式：

`channel_scale = 1 + 0.5 * tanh(channel_prior)`

范围为 0.5-1.5，因此 channel gate 既能抑制，也能增强，不再像 sigmoid gate 一样只能压低。

最终融合：

`effective_gate = dynamic_group_gate * channel_scale`

为解决 Selective-12 中 gate 几乎不更新的问题，channel prior 的反向梯度放大 5 倍。ADL gate penalty 保持为 0。

通道选择来自 Subject1-2 + Zenodo train，并由 Subject3 + Zenodo val 验证；Subject4 不参与调参。
