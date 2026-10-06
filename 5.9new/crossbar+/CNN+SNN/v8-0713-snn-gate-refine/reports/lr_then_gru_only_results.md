# 学习率与 GRU-only 对比

## 统一口径

- 训练集：GMDCSA Subject 1-2 + Zenodo train，共 3539 个窗口
- 验证集：GMDCSA Subject 3 + Zenodo val，共 1076 个窗口
- Crossbar 与 device dynamics：启用
- ADL penalty：0.0
- Subject4：未使用

## 结果

| 方案 | 最佳 epoch | Balanced Accuracy | ADL specificity | Fall recall |
|---|---:|---:|---:|---:|
| group-only | 1 | 74.1% | 72.9% | 75.3% |
| 统一 lr=1e-4 | 4 | 74.2% | 68.1% | 80.3% |
| CNN/proj 冻结 + GRU-only | 5 | 73.9% | 68.8% | 79.1% |

## 结论

将统一学习率从 8e-4 降至 1e-4 后，解冻 CNN/GRU 不再出现原先的明显崩溃，说明学习率过高确实是主要原因之一。GRU-only 比统一低学习率方案略提高 ADL specificity，但 Balanced Accuracy 和 Fall recall 均略低；它没有超过 group-only 的 ADL-Fall 平衡。

本轮应保留统一 lr=1e-4 作为后续基线。GRU-only 方案可作为稳定性备选，但暂不替代 group-only。第一次错误的 GRU-only 运行使用了 train=2508、val=564，已停止并标记为无效，不纳入比较。
