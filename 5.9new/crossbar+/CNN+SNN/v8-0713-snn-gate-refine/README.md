# v8-0713 SNN gate 精简与残差实验

## 诊断依据

前置诊断位于 `v8-0713-snn-channel-diagnostic`。

- 训练集有 52 个、验证集有 58 个通道的 Fall-hard ADL 区分贡献为负。
- 直接使用 Top-24、Top-48、Top-72 掩码没有超过完整 96 通道。
- group-channel 的验证集平均 gate 很低，mean/max/last 三组约为 0.05/0.05/0.09。
- 当前主要问题是双层 gate 整体过度压制，而不是简单删除若干通道即可解决。

## 两个候选结构

### group_only_p000

- 只学习 mean/max/last 三个组 gate。
- 移除 96 个 channel gate。
- ADL gate penalty 为 0。

### group_channel_residual_p000

- 保留 group gate 与 channel gate。
- 最低通过比例为 0.20。
- `effective_gate = 0.20 + 0.80 * group_gate * channel_gate`。
- ADL gate penalty 为 0。

## 数据纪律

- 训练：GMDCSA Subject1-2 + Zenodo train。
- 验证和结构选择：GMDCSA Subject3 + Zenodo val。
- Subject4 不参与结构选择、阈值选择和调参。
