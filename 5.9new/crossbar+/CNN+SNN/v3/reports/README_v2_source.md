# CNN+SNN v2

## 版本目的

本版本在 `CNN+SNN` 第一版基础上继续优化 SNN temporal branch。核心变化是加入“冻结 CNN/GRU 前几轮”的训练策略：

- 前 2 个 epoch 冻结 CNN encoder、projection、GRU。
- 只训练新增 SNN temporal branch 和融合分类 head。
- 第 3 个 epoch 起解冻全部模型继续联合训练。

这样做的目的，是避免随机初始化的 SNN 分支在训练初期破坏 v4 已经学好的 CNN/GRU 表征。

## 路径

- 当前工程：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v2`
- 代码：`code\automatic_fall_detector.py`
- 扫描结果：`reports\scan_freeze_snn`
- 汇总表：`reports\scan_freeze_snn\scan_freeze_snn_summary.csv`
- 当前最佳模型：`models\best_d82_t55_model.pth`
- 当前最佳曲线：`image\best_d82_t55_curve.png`

## 数据说明

本轮仍然是结构验证，不是最终论文测试流程。

使用：

- train：`F:\12.18-2\zenodo_falldb_video_split\train`
- val：`F:\12.18-2\zenodo_falldb_video_split\val`
- `--external-labeled-stride 24`

未使用：

- 未使用 GMDCSA Subject1-2。
- 未使用 GMDCSA Subject3。
- 未使用 Subject4。
- 未使用 ADL-add。

回答用户疑问：

`external-labeled-stride=24` 这一轮只是为了让 Zenodo video split 的窗口数量接近 v4 当时的训练规模，避免 stride=6 时窗口数量过大导致内存错误。它不代表用了 GMDCSA Subject1-2，也不代表 val 是 Subject3。本轮 train/val 都来自 Zenodo video split。

如果后续要进入严格论文流程，需要重新组织：

- Subject1-2 + 选定外部 train 用于训练。
- Subject3 + 外部 val 用于阈值校准。
- Subject4 只做 final test。

## 新增代码开关

| 参数 | 本轮设置 | 含义 |
|---|---:|---|
| `--use-snn-temporal-branch` | 开启 | 开启 CNN+SNN 融合 |
| `--freeze-cnn-gru-epochs` | 2 | 前 2 轮冻结 CNN/GRU |
| `--snn-hidden-dim` | 32 | SNN 分支隐藏维度 |
| `--snn-surrogate-scale` | 8.0 | surrogate spike sigmoid 斜率 |

冻结时参数量：

| 状态 | trainable | frozen |
|---|---:|---:|
| epoch 1-2 | 63,460 | 153,664 |
| epoch 3+ | 217,124 | 0 |

## 扫描范围

| 参数 | 扫描值 |
|---|---|
| `snn-decay` | 0.75 / 0.82 / 0.90 |
| `snn-threshold` | 0.45 / 0.55 / 0.65 |

固定参数：

- v4 checkpoint 初始化。
- teacher-student gaussian noise 开启。
- device dynamics 开启。
- Crossbar readout noise scale = 0.25。
- hard-negative-normal-weight = 1.10。
- hard-negative-min-percentile = 0.88。
- hard-negative-score-gamma = 2.00。
- normal-positive-penalty = 0.00。
- epochs = 10。
- patience = 4。

## 扫描结果

| 组合 | snn-decay | snn-threshold | best epoch | val acc | val balanced | val normal | val fall |
|---|---:|---:|---:|---:|---:|---:|---:|
| `d82_t55` | 0.82 | 0.55 | 3 | 0.746 | 0.742 | 0.727 | 0.756 |
| `d90_t55` | 0.90 | 0.55 | 2 | 0.706 | 0.731 | 0.811 | 0.650 |
| `d75_t65` | 0.75 | 0.65 | 1 | 0.704 | 0.730 | 0.818 | 0.642 |
| `d82_t65` | 0.82 | 0.65 | 2 | 0.709 | 0.729 | 0.795 | 0.663 |
| `d75_t55` | 0.75 | 0.55 | 2 | 0.698 | 0.726 | 0.818 | 0.634 |
| `d75_t45` | 0.75 | 0.45 | 2 | 0.698 | 0.724 | 0.811 | 0.638 |
| `d82_t45` | 0.82 | 0.45 | 2 | 0.698 | 0.724 | 0.811 | 0.638 |
| `d90_t45` | 0.90 | 0.45 | 2 | 0.696 | 0.722 | 0.811 | 0.634 |
| `d90_t65` | 0.90 | 0.65 | 2 | 0.709 | 0.720 | 0.758 | 0.683 |

## 阶段结论

当前最佳组合是：

```text
snn-decay = 0.82
snn-threshold = 0.55
freeze-cnn-gru-epochs = 2
```

最佳验证结果：

| 指标 | 数值 |
|---|---:|
| val accuracy | 74.6% |
| val balanced accuracy | 74.2% |
| val normal accuracy | 72.7% |
| val fall accuracy | 75.6% |

这比 CNN+SNN v1 的最高 `val balanced accuracy = 69.4%` 明显更好，也高于 v4 训练日志中约 `71.0%` 的验证 balanced accuracy。

初步说明：

1. SNN temporal branch 本身有潜力，但不能直接全模型一起训练。
2. 先冻结 CNN/GRU，让 SNN 分支和融合 head 先适配，再联合微调，效果更稳定。
3. `snn-decay=0.82`、`snn-threshold=0.55` 是当前最平衡组合，normal 与 fall 都没有明显偏科。

注意：

这个结论只来自 Zenodo video split 验证，不是 Subject4 独立测试结论。不能用它直接写最终泛化性能。

## 下一步建议

1. 固定 `d82_t55` 作为 CNN+SNN 当前候选。
2. 进入严格验证流程：Subject3 + 外部 val 做阈值校准。
3. 如果验证集仍然稳定，再用 Subject4 做唯一一次 final test。
4. 如果后续要把 GMDCSA Subject1-2 加入训练，必须解决窗口全量拼接导致的内存问题，例如视频级采样、最大窗口数限制或 streaming dataset。

## 严格流程测试

执行时间：2026-05-26

严格流程：

1. 固定模型：`models\best_d82_t55_model.pth`。
2. 只用 `Subject3 + Zenodo val` 做阈值校准。
3. 根据验证集选出唯一阈值。
4. 使用唯一阈值对 Subject4 做一次 final test。

结果文件：

- 验证集阈值扫描：`reports\strict_validation_v2\strict_validation_threshold_summary.csv`
- 推荐阈值：`reports\strict_validation_v2\recommended_threshold.json`
- Subject4 final：`reports\strict_subject4_final_t60\subject4_final_metrics.json`

验证集阈值扫描：

| fall_prob_threshold | Accuracy | Balanced Acc | ADL specificity | Fall recall | Precision | TP/TN/FP/FN |
|---:|---:|---:|---:|---:|---:|---|
| 0.35 | 62.30% | 61.83% | 33.33% | 90.32% | 58.33% | 28/10/20/3 |
| 0.40 | 62.30% | 61.83% | 33.33% | 90.32% | 58.33% | 28/10/20/3 |
| 0.45 | 62.30% | 61.83% | 33.33% | 90.32% | 58.33% | 28/10/20/3 |
| 0.50 | 62.30% | 61.83% | 33.33% | 90.32% | 58.33% | 28/10/20/3 |
| 0.55 | 67.21% | 66.83% | 43.33% | 90.32% | 62.22% | 28/13/17/3 |
| 0.60 | 70.49% | 70.22% | 53.33% | 87.10% | 65.85% | 27/16/14/4 |
| 0.65 | 67.21% | 66.99% | 53.33% | 80.65% | 64.10% | 25/16/14/6 |

验证集推荐：

```text
fall_prob_threshold = 0.60
```

Subject4 final，严格独立测试：

| Accuracy | Balanced Acc | ADL specificity | Fall recall | Precision | TP/TN/FP/FN |
|---:|---:|---:|---:|---:|---|
| 43.24% | 44.41% | 30.00% | 58.82% | 41.67% | 10/6/14/7 |

阶段判断：

CNN+SNN v2 在验证集上表现为更敏感，验证集 Fall recall 很高，但 ADL specificity 不够。跨到 Subject4 后也延续了这个倾向：Fall recall 保留到 58.82%，但 ADL specificity 只有 30.00%，误报 ADL 明显偏多。

与 v4 penalty0 对比：

| 版本 | Subject4 Accuracy | ADL specificity | Fall recall | Precision |
|---|---:|---:|---:|---:|
| v4 penalty0 | 75.68% | 80.00% | 70.59% | 75.00% |
| CNN+SNN v2 strict | 43.24% | 30.00% | 58.82% | 41.67% |

结论：

当前 CNN+SNN v2 不能替代 v4 主线。它的价值在于证明 SNN temporal branch 能提高模型对 Fall 动态的敏感性，但目前缺少对 ADL hard negative 的约束，导致大量 ADL 被误报为 Fall。后续如果继续这条线，应优先加入验证集约束或双目标选择策略，而不是只看 Fall recall。
