# v13：放松 subtype 监督

## 版本目的

v13 基于 `v12` 做严格对照，只改一件核心事情：

- 放松 `ADL subtype` 辅助监督的保守性

目标是验证：

- `v12` 的问题是否主要来自 subtype 监督过强，导致模型过度保守；
- 如果把 subtype 监督放轻，是否可以在不污染测试集的前提下，把 `Fall recall` 拉回来。

## 相对 v12 的改动

1. `--adl-subtype-loss-weight`
   - 从 `0.30` 降到 `0.15`

2. 新增更温和的 subtype 监督策略
   - `--adl-subtype-hard-only`
   - `--adl-subtype-hard-score-threshold 0.22`

含义：

- 不再让所有 normal ADL 都参与 subtype 辅助分类；
- 只让更像 hard negative 的 ADL 窗口参与 subtype loss；
- 这样可以减少 subtype 分支对普通 ADL 的整体压制，避免主分类器被一起拖向过保守。

## 数据划分

严格保持：

- 训练：`Subject 1 + Subject 2 + Zenodo train`
- 验证与阈值校准：`Subject 3 + Zenodo val`
- 最终独立测试：`Subject 4`

`Subject 4` 没有参与训练、阈值扫描或参数选择。

## 运行环境

- 项目根目录：`F:\12.18-2`
- Python 环境：`F:\ANACONDA\envs\my_yizu_3.10\python.exe`
- Crossbar 数据目录：`F:\12.18-2\data`

## 训练设置

其余主干设置与 `v12` 保持一致：

- `group_channel` gate
- `teacher-student gaussian noise`
- `--use-device-dynamics`
- `--crossbar-readout-noise-scale 0.25`
- `hard_negative_normal_weight = 1.00`
- `snn_adl_gate_penalty = 0.06`
- `image_size = 64`
- `external_labeled_stride = 16`
- `freeze_cnn_gru_epochs = 2`

并显式加入：

- `--disable-auxiliary-data`

说明：

- 这一步不是方法改动，而是为了和 `v12` 的 `aux_records=0` 实际训练条件保持一致；
- 避免因为新建工程目录没有自标注 `image+xml` 辅助数据而在启动阶段报错。

## 训练过程结果

训练在 `epoch 6` 提前 early stop。

关键日志趋势：

- `epoch 1-2` 时模型比 `v12` 更放开，`val_fall` 更高；
- `epoch 3` 解冻 CNN/GRU 后，验证集开始重新向保守端回摆；
- 最终说明：只靠降低 subtype loss 强度，可以把 recall 拉回一些，但不足以同时稳住 ADL specificity。

## 严格评估结果

阈值选择：

- 仅在 `Subject3 + Zenodo val` 上扫描
- 推荐阈值：`0.45`

`Subject4` 最终结果：

- `Accuracy = 54.05%`
- `Balanced Accuracy = 56.18%`
- `ADL specificity = 30.00%`
- `Fall recall = 82.35%`
- `Precision = 50.00%`
- `TP / TN / FP / FN = 14 / 6 / 14 / 3`

## 结果解释

v13 的意义很明确：

- 它不是更优最终版；
- 但它证明了 `v12` 的 100% specificity 不是测试泄露，而是过度保守造成的极端工作点。

换句话说，v12 和 v13 共同夹出了当前强分类路线的两端：

- `v12`：ADL 压得很干净，但漏报太多；
- `v13`：Fall 抓得更全，但 ADL 误报太多。

这说明下一步不能只继续“加大/减小 subtype loss”，而要做更细的结构性约束，让模型学会：

- 对真正 Fall 的动态突变放大；
- 对坐下、蹲下、弯腰这类 ADL hard negative 做定向抑制；
- 而不是把所有 ADL 一起压成保守决策。

## 本版结论

v13 不是最终候选，但它完成了一个很重要的验证：

- `v12` 的主要问题确实是过度保守；
- 单纯放松 subtype 辅助监督，会把模型直接推向高 recall / 高误报的另一端；
- 所以下一步应该做“有选择地抑制 ADL hard negative”，而不是继续全局放松。

## 追加小扫描：`adl_subtype_loss_weight = 0.25 / 0.20`

后续又在 `v13` 目录下追加了两组权重扫描：

- `v13_w025`
- `v13_w020`

注意：

- 这两组和彼此之间是可直接对照的；
- 但它们使用的当前数据装载口径下，训练/验证窗口数为：
  - train `3539`
  - val `1076`
- 因此它们和最早那版 `v13_relaxed_subtype`（train `2508` / val `564`）不是完全同口径对照。

### `v13_w025`

- 推荐阈值：`0.70`
- Subject4：
  - `Accuracy = 64.86%`
  - `Balanced Accuracy = 66.18%`
  - `ADL specificity = 50.00%`
  - `Fall recall = 82.35%`
  - `Precision = 58.33%`
  - `TP / TN / FP / FN = 14 / 10 / 10 / 3`

### `v13_w020`

- 推荐阈值：`0.70`
- Subject4：
  - `Accuracy = 64.86%`
  - `Balanced Accuracy = 66.18%`
  - `ADL specificity = 50.00%`
  - `Fall recall = 82.35%`
  - `Precision = 58.33%`
  - `TP / TN / FP / FN = 14 / 10 / 10 / 3`

### 追加扫描结论

- 在当前这套 full split 口径下，`0.20` 和 `0.25` 最终几乎没有区别；
- 说明当前主矛盾已经不是 subtype loss weight 再微调 `0.05` 能解决的；
- 更大的问题仍然是：
  - 真实 Fall recall 已经较高；
  - 但 ADL hard negative 的误报压得还不够干净。
