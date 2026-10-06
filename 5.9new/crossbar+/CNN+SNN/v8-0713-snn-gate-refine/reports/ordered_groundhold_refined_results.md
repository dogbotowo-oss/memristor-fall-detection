# Ground-hold 强化版结果

## 设置

- 训练集：GMDCSA Subject 1-2 + Zenodo train，3539 窗口
- 验证集：GMDCSA Subject 3 + Zenodo val，1076 窗口
- CNN/proj：全程冻结
- 学习率：GRU、SNN、gate、head 统一 `2e-4`
- 调整：静止判断从峰值后 `+2` 帧改为 `+3` 帧；静止指数 `4 -> 5`；低位阈值 `0.55 -> 0.60`；ordered factor `0.25~1.0 -> 0.20~0.80`
- Subject4：未使用

## 结果

| 方案 | 最佳 epoch | Balanced Accuracy | ADL specificity | Fall recall |
|---|---:|---:|---:|---:|
| group-only | 1 | 74.1% | 72.9% | 75.3% |
| CNN 冻结 + lr=2e-4 | 5 | 74.4% | 68.6% | 80.1% |
| ground-hold 强化版 | 5 | 74.4% | 68.8% | 80.1% |

## 结论

ground-hold 强化仅使 ADL specificity 提高 0.2 个百分点，Balanced Accuracy 和 Fall recall 不变，仍明显低于 group-only 的 ADL specificity。说明当前 gate 网络对 ordered score 的依赖较弱，单纯调整输入 score 的阈值和系数无法解决主要的 ADL-Fall 混淆。

下一步应停止继续微调同一组系数，转向 gate 输出后校正或最小噪声消融，确认问题来自 gate 学习能力还是器件/teacher-student 噪声叠加。
