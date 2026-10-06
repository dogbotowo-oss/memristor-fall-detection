# Posture-aware Group Gate

本实验只优化 group gate，关闭 ADL gate penalty，不再使用 channel gate。

Group gate 的输入增加六类可解释证据：

- impact：跌倒样动态峰值。
- peak motion：快速移动峰值。
- center drop：人体中心向下变化。
- posture change：高度降低或宽高比增大。
- ground hold：低位/姿势变化后尾段保持静止。
- confirmed fall：impact 与 ground hold 同时成立。

三个 group 的引导含义：

- mean-spike：更关注持续性和倒地后保持。
- max-spike：更关注冲击与快速运动。
- last-spike：更关注尾段姿势、静止和地面接触。

最终 group gate 由 50% 学习 gate 与 50% 可解释引导 gate 融合。训练和选择只使用 Subject1-3 与 Zenodo train/val；Subject4 不参与调参。
