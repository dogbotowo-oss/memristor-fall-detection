# Ordered Event Group Gate 结果

## 实验口径

- 结构：`group_ordered_event_p000`
- 初始化：`group_only_p000`
- 时间顺序：快速冲击 -> 中心下降 -> 姿势/触地变化 -> 尾段持续静止
- 训练集：GMDCSA Subject 1-2 + Zenodo train，共 3539 个窗口
- 验证集：GMDCSA Subject 3 + Zenodo val，共 1076 个窗口
- ADL penalty：0.0
- Crossbar 与 device dynamics：启用且加载正常
- Subject4：未参与训练、阈值选择或结构选择

## 最佳验证结果

最佳 checkpoint 来自 epoch 1，随后在 epoch 6 触发早停。

| 结构 | Accuracy | Balanced Accuracy | ADL specificity | Fall recall |
|---|---:|---:|---:|---:|
| group-only | 74.1% | 74.1% | 72.9% | 75.3% |
| ordered event | 73.9% | 74.0% | 71.6% | 76.5% |
| 变化 | -0.2 pp | -0.1 pp | -1.3 pp | +1.2 pp |

## 结论

时间顺序约束将 Fall recall 从 75.3% 提高到 76.5%，满足 Fall recall 不低于 75% 的要求，但 ADL specificity 从 72.9% 降到 71.6%，Balanced Accuracy 也未超过 group-only。

因此，本次 ordered confirmation 证明时间顺序信息能够增强跌倒敏感性，但当前单标量注入方式仍会放大部分具有相似动作顺序的 ADL。`group_ordered_event_p000` 不替代 `group_only_p000`，下一轮应保留时间顺序特征，但降低其 gate 增益或只在“冲击后持续触地”证据充分时启用。

