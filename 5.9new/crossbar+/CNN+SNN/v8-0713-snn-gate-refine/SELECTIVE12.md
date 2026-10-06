# Selective-negative-12 gate

本实验关闭 ADL gate penalty，只处理通道 gate 本身。

稳定强负贡献通道（0-based）：

`56, 46, 62, 45, 44, 47, 33, 53, 41, 38, 61, 15`

其中 11 个来自 max-spike 组，1 个来自 mean-spike 组。

未选中的 84 个通道保持 `channel_scale=1.0`。选中的 12 个通道使用：

`channel_scale = 0.5 + 0.5 * sigmoid(z)`

因此这些通道可以在 0.5-1.0 范围内被学习性衰减，但不会被直接清零。最终 gate 为：

`effective_gate = group_gate * channel_scale`

训练与结构选择只使用 Subject1-3 和 Zenodo train/val，Subject4 不参与调参。
