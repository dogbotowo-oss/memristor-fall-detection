# v8-0713 SNN gate 验证结果

以下结果均来自 `GMDCSA Subject3 + Zenodo val`，以最高 balanced accuracy 的 epoch 为准。Subject4 未参与结构选择或调参。

| 版本 | Accuracy | Balanced Accuracy | ADL specificity | Fall recall |
|---|---:|---:|---:|---:|
| no SNN | 69.5% | 69.6% | 68.1% | 71.2% |
| SNN no gate | 73.2% | 73.6% | 68.6% | 78.5% |
| scalar gate, penalty=0.06 | 73.1% | 73.7% | 64.9% | 82.5% |
| scalar gate, penalty=0 | 73.0% | 73.5% | 64.6% | 82.5% |
| group-channel, penalty=0.06 | 73.7% | 73.8% | 72.9% | 74.6% |
| group-channel, penalty=0 | 73.7% | 73.7% | 73.1% | 74.4% |
| group-only, penalty=0 | 74.1% | 74.1% | 72.9% | 75.3% |
| group-channel residual, penalty=0 | 73.8% | 73.9% | 72.9% | 74.8% |

## 结论

- ADL gate penalty 从 0.06 改为 0 后，指标变化仅约 0.1-0.3 个百分点，不是主要瓶颈。
- residual floor=0.20 只带来轻微提升，没有超过 group-only。
- group-only 是当前验证集综合最优结构：相对原 group-channel penalty=0.06，balanced accuracy 提升 0.3 个百分点，Fall recall 提升 0.7 个百分点，ADL specificity 保持 72.9%。
- 下一步优先对 group-only 补随机种子稳定性验证，不继续扫描 residual floor。
