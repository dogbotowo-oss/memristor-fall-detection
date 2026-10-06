# CNN 冻结与分层学习率结果

## 口径

- 训练集：GMDCSA Subject 1-2 + Zenodo train，3539 窗口
- 验证集：GMDCSA Subject 3 + Zenodo val，1076 窗口
- CNN encoder/proj：全程冻结
- 学习率：CNN `1e-5`、GRU `5e-5`、SNN/gate/head `2e-4`
- Crossbar 与 device dynamics：启用
- Subject4：未使用

## 结果

| 方案 | 最佳 epoch | Balanced Accuracy | ADL specificity | Fall recall |
|---|---:|---:|---:|---:|
| group-only | 1 | 74.1% | 72.9% | 75.3% |
| 统一 lr=1e-4 | 4 | 74.2% | 68.1% | 80.3% |
| CNN 冻结 + 分层学习率 | 5 | 74.2% | 68.6% | 79.7% |

## 结论

CNN 全程冻结并使用分层学习率后，训练保持稳定，说明冻结空间特征可以避免 CNN 被继续微调破坏。与统一 `lr=1e-4` 相比，ADL specificity 提高 0.5 个百分点，但 Balanced Accuracy 相同且 Fall recall 略低 0.6 个百分点；仍未超过 group-only 的 ADL-Fall 平衡。因此该设置适合作为稳定训练方案，但暂不替代 group-only。
