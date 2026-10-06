# CNN+SNN v3

## 版本目的

本版本在 CNN+SNN v2 基础上继续向“本域 + 外域”混合训练推进，核心变化是把 GMDCSA Subject1-2 的 ADL/Fall 加入训练集，用来补足本域 hard negative 和 fall dynamic 的覆盖。

训练策略保持 v2 的 SNN 冻结思路：

- 前 2 个 epoch 冻结 CNN encoder / projection / GRU。
- 先训练 SNN temporal branch 和融合 head。
- 第 3 个 epoch 起解冻联合训练。

## 路径

- 当前工程：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v3`
- 代码：`code\automatic_fall_detector.py`
- 模型：`models\automatic_fall_event_detector_cnn_snn_v3_subject12_zenodo.pth`
- 训练曲线：`image\training_curve_cnn_snn_v3_subject12_zenodo.png`
- 日志：`reports\train.log`

## 数据划分

本轮训练：

- GMDCSA Subject1-2：训练
- Zenodo `train`：训练
- GMDCSA Subject3：验证和阈值校准
- Zenodo `val`：验证和阈值校准
- Subject4：只做最终独立测试

说明：

- Subject1-2 里同时包含 ADL 和 Fall，所以不是只补 ADL，而是把本域两类都加进来，避免模型只会适应 Zenodo 的轻量域。
- 为避免窗口量爆炸，仍然使用 `--external-labeled-stride 24`。

## 训练配置

固定参数：

| 参数 | 数值 |
|---|---:|
| `--use-snn-temporal-branch` | 开启 |
| `--freeze-cnn-gru-epochs` | 2 |
| `snn-hidden-dim` | 32 |
| `snn-decay` | 0.82 |
| `snn-threshold` | 0.55 |
| `snn-surrogate-scale` | 8.0 |
| `teacher-student-noise` | 开启 |
| `student-noise-mode` | gaussian |
| `use-device-dynamics` | 开启 |
| `crossbar-readout-noise-scale` | 0.25 |
| `hard-negative-normal-weight` | 1.10 |
| `hard-negative-min-percentile` | 0.88 |
| `hard-negative-score-gamma` | 2.00 |
| `normal-positive-penalty` | 0.00 |
| `external-labeled-stride` | 24 |

初始化：

- 使用 `CNN+SNN v2` 的最佳模型 `best_d82_t55_model.pth` 继续训练。

## 本轮训练数据统计

| 项目 | normal | fall | total |
|---|---:|---:|---:|
| train | 1187 | 1189 | 2376 |
| val | 391 | 335 | 726 |

## 训练过程摘要

| Epoch | train acc | val acc | val balanced | val normal | val fall |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.708 | 0.683 | 0.662 | 0.934 | 0.391 |
| 2 | 0.723 | 0.734 | 0.732 | 0.762 | 0.701 |
| 3 | 0.693 | 0.727 | 0.730 | 0.693 | 0.767 |
| 4 | 0.746 | 0.704 | 0.720 | 0.517 | 0.922 |
| 5 | 0.766 | 0.675 | 0.675 | 0.675 | 0.675 |
| 6 | 0.767 | 0.734 | 0.733 | 0.744 | 0.722 |
| 7 | 0.792 | 0.573 | 0.562 | 0.701 | 0.424 |
| 8 | 0.806 | 0.696 | 0.706 | 0.570 | 0.842 |
| 9 | 0.815 | 0.649 | 0.635 | 0.818 | 0.451 |
| 10 | 0.822 | 0.638 | 0.623 | 0.808 | 0.439 |
| 11 | 0.830 | 0.629 | 0.619 | 0.752 | 0.487 |

早停在 epoch 11。

## 严格验证

验证集校准使用：

- `Subject3 + Zenodo val`

阈值扫描结果：

| fall_prob_threshold | Acc | Balanced Acc | ADL specificity | Fall recall | Precision |
|---:|---:|---:|---:|---:|---:|
| 0.35 | 59.02% | 58.92% | 53.33% | 64.52% | 58.82% |
| 0.40 | 59.02% | 58.92% | 53.33% | 64.52% | 58.82% |
| 0.45 | 59.02% | 58.92% | 53.33% | 64.52% | 58.82% |
| 0.50 | 57.38% | 57.31% | 53.33% | 61.29% | 57.58% |
| 0.55 | 63.93% | 63.98% | 66.67% | 61.29% | 65.52% |
| 0.60 | 67.21% | 67.37% | 76.67% | 58.06% | 72.00% |
| 0.65 | 68.85% | 69.19% | 90.00% | 48.39% | 83.33% |

推荐阈值：

```text
fall_prob_threshold = 0.65
```

Subject4 final，严格独立测试：

| Accuracy | Balanced Acc | ADL specificity | Fall recall | Precision | TP/TN/FP/FN |
|---:|---:|---:|---:|---:|---|
| 67.57% | 66.03% | 85.00% | 47.06% | 72.73% | 8/17/3/9 |

## 结论

和 v2 相比，v3 更能压住 ADL 误报：

| 版本 | Subject4 Accuracy | ADL specificity | Fall recall | Precision |
|---|---:|---:|---:|---:|
| CNN+SNN v2 strict | 43.24% | 30.00% | 58.82% | 41.67% |
| CNN+SNN v3 strict | 67.57% | 85.00% | 47.06% | 72.73% |

这说明加入 Subject1-2 后，模型确实学到了更多本域 ADL 形态，ADL specificity 明显上升。但是 Fall recall 下降，说明现在模型变得更保守了。

当前判断：

- `v3` 比 `v2` 更像一个可用的“压 ADL”版本。
- 但它还没有超过 v4 penalty0 的 `75.68% / 80.00% / 70.59% / 75.00%`。
- 下一步如果继续这条线，应该优先在保持 ADL specificity 的前提下，把 Fall recall 往回拉，而不是再盲目扩 train 数据。
