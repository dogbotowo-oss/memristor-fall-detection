# CNN+SNN

## 版本目的

本工程是在 `v4_penalty0` 最好工程代码基础上建立的新方案，用于验证“CNN 主干 + SNN/LIF-inspired 时序分支”的神经形态增强路线。

核心想法：

- 保留 v4 的 CNN + GRU 主干、Crossbar 映射、器件动态、teacher-student noise、hard negative、event filter 等已有开关。
- 新增一个可开关的 `SNN temporal branch`。
- SNN 分支不直接吃完整 RGB/灰度图，而是从每个视频窗口中提取运动能量、人体重心变化、软姿态高宽比、fall-event-like 动态分数等时序特征。
- 使用 LIF-inspired 膜电位累积与脉冲响应建模快速状态变化。
- 最后将 SNN 分支输出与 CNN-GRU 时序特征融合，再输出 4 类 logits。

这个版本的目的不是立即替换 v4，而是作为论文中“忆阻器动态响应/神经形态时序增强”的算法分支，为后续消融实验提供代码基础。

## 路径

- 当前工程：`F:\12.18-2\5.9new\crossbar+\CNN+SNN`
- 代码：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\code\automatic_fall_detector.py`
- v4 来源工程：`F:\12.18-2\5.9new\crossbar+\优化7.3\reports\7.4\7.4-adl-2\v4_penalty0`
- 7.4 best no-filter 对照：`F:\12.18-2\5.9new\crossbar+\优化7.3\reports\7.4\7.4best-no-filter`
- 当前 v12 对照：`F:\12.18-2\5.9new\crossbar+\优化7.3\reports\7.4\7.4-adl-2\v12_screened_adl_hardneg`

## 运行环境

- conda 环境：`my_yizu_3.10`
- Python：`F:\ANACONDA\envs\my_yizu_3.10\python.exe`
- 工程根目录：`F:\12.18-2`

## 数据纪律

严格保持：

- GMDCSA Subject1-2：训练。
- GMDCSA Subject3：验证和阈值校准。
- GMDCSA Subject4：最终独立测试，只能在唯一策略确定后测试一次。
- 外部 Zenodo/ADL 数据必须按 subject/person/video/folder 划分，不能窗口随机混合。

本版本目前只完成代码结构，不使用 Subject4 做任何参数选择。

## 新增开关

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--use-snn-temporal-branch` | 关闭 | 开启 CNN+SNN 融合模型 |
| `--snn-hidden-dim` | 32 | SNN 分支隐藏维度 |
| `--snn-decay` | 0.82 | LIF 膜电位衰减系数 |
| `--snn-threshold` | 0.55 | LIF 脉冲阈值 |
| `--snn-surrogate-scale` | 8.0 | sigmoid surrogate spike 的斜率 |

默认关闭 SNN 分支时，模型行为保持接近 v4 原结构。

## 代码核心逻辑

新增位置：

- `FallEventDetector.__init__`
- `FallEventDetector.extract_temporal_event_features`
- `FallEventDetector.run_lif_temporal_branch`
- `FallEventDetector.forward`
- `load_compatible_model_state`

SNN 分支输入的 7 个时序特征：

| 特征 | 含义 |
|---|---|
| `mass` | 前景/人体响应整体强度 |
| `motion` | 相邻帧变化强度 |
| `center_y` | 软重心纵向位置 |
| `center_drop` | 重心向下快速变化 |
| `soft_height` | 软 bbox 高度估计 |
| `aspect` | 软宽高比 |
| `fall_event_like` | motion、center_drop、高度变化、姿态变化组合出的跌倒瞬间动态分数 |

LIF-inspired 过程：

```text
动态特征序列
→ Linear + LayerNorm + ReLU
→ 膜电位随时间累积：membrane = decay * membrane + input
→ surrogate spike：sigmoid((membrane - threshold) * scale)
→ spike 后膜电位软复位
→ mean / max / last spike pooling
→ 与 CNN-GRU pooled feature 拼接
→ 分类 head 输出 logits
```

## 兼容 v4 初始化

如果使用 v4 checkpoint 初始化，本版本会兼容加载同形状参数：

- 可加载：CNN encoder、projection、GRU 等相同结构参数。
- 自动跳过：新增 SNN 分支参数、融合后 head 中形状变化的参数。

这样可以在 v4 已学习到的视觉/时序基础上训练 SNN 增强分支，而不是完全从零开始。

## 当前状态

已完成：

- 建立 `code / image / models / reports` 工程结构。
- 从 `v4_penalty0` 复制主代码。
- 新增 SNN/LIF-inspired temporal branch。
- 新增 CLI 开关。
- 新增 checkpoint config 保存和加载逻辑。
- 新增 v4 checkpoint 兼容加载逻辑。
- 已通过 `py_compile`。
- 已通过随机张量前向测试，输出形状为 `(batch, 4)`。
- 已完成第一轮 CNN+SNN 训练。

尚未完成：

- 尚未做 Subject3 + 外部 val 阈值校准。
- 尚未做 Subject4 final test。

## 第一轮训练记录

训练时间：2026-05-26

输出文件：

- 模型：`models/automatic_fall_event_detector_cnn_snn_v1.pth`
- 训练曲线：`image/training_curve_cnn_snn_v1.png`
- 日志：`reports/train.log`

本轮训练使用 v4 可比规模：

- 训练数据：`F:\12.18-2\zenodo_falldb_video_split\train`
- 验证数据：`F:\12.18-2\zenodo_falldb_video_split\val`
- `--external-labeled-stride 24`
- 未加入 Subject4。
- 未加入 ADL-add。
- 未加入原始 GMDCSA Subject1/2 全量目录。

说明：

最开始尝试将原始 GMDCSA Subject1/2 与 Zenodo train 同时作为 `external-labeled-dir`，但窗口数量达到 9291，拼接数组需要约 9.07 GiB 内存，不符合 v4 基线规模，也导致内存错误。随后只使用 Zenodo train 且 stride=6，窗口数仍达到 6616，需要约 6.46 GiB。根据 v4 日志中的训练规模 `external_labeled_windows=1808`、验证规模 `external_val_labeled_windows=551`，本轮改用 `--external-labeled-stride 24` 来恢复 v4 量级，作为结构验证的第一轮。

本轮实际数据统计：

| 项目 | normal | fall | total |
|---|---:|---:|---:|
| train | 693 | 989 | 1682 |
| val | 132 | 246 | 378 |

训练参数：

| 参数 | 数值 |
|---|---:|
| `--use-snn-temporal-branch` | 开启 |
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
| `init-model-path` | v4 penalty0 checkpoint |

v4 checkpoint 兼容加载结果：

| 项目 | 数量 |
|---|---:|
| missing keys | 7 |
| unexpected keys | 0 |
| skipped shape keys | 1 |

这表示原 CNN/GRU 主干大部分参数已从 v4 加载，新增 SNN 分支和融合 head 中不兼容的参数重新初始化。

训练结果：

| Epoch | train loss | train acc | val loss | val acc | val balanced | val normal | val fall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9150 | 0.604 | 0.8074 | 0.709 | 0.661 | 0.500 | 0.821 |
| 2 | 0.8589 | 0.669 | 0.9039 | 0.630 | 0.607 | 0.530 | 0.683 |
| 3 | 0.8348 | 0.699 | 0.8434 | 0.690 | 0.694 | 0.705 | 0.683 |
| 4 | 0.8202 | 0.709 | 0.8710 | 0.688 | 0.662 | 0.576 | 0.748 |
| 5 | 0.8097 | 0.721 | 1.0057 | 0.495 | 0.575 | 0.841 | 0.309 |
| 6 | 0.7721 | 0.741 | 1.2890 | 0.421 | 0.507 | 0.795 | 0.220 |
| 7 | 0.7639 | 0.739 | 1.1299 | 0.474 | 0.585 | 0.955 | 0.215 |
| 8 | 0.7303 | 0.763 | 1.1288 | 0.508 | 0.545 | 0.667 | 0.423 |

早停：

- Early stopping at epoch 8。
- 最佳 checkpoint 来自验证指标最高的中间 epoch。

阶段判断：

CNN+SNN 分支可以正常训练，但第一轮验证表现没有超过 v4 原训练表现。v4 日志中最高验证 balanced accuracy 约为 `0.710`，本轮 CNN+SNN 最高约为 `0.694`。目前不能说明 SNN 分支带来性能增益，只能说明结构可运行，且没有明显训练崩溃。

可能原因：

1. SNN 分支新增参数随机初始化，虽然 CNN/GRU 从 v4 加载，但融合 head 有部分参数需要重新学习。
2. 当前 LIF 特征和阈值未经验证集调参，`snn-threshold=0.55`、`snn-decay=0.82` 只是第一轮保守默认值。
3. 本轮只恢复 v4 量级的 Zenodo split 训练，还没有加入严格的 Subject3 校准流程。
4. SNN 分支可能需要更小学习率或冻结 CNN 主干短训，否则会扰动 v4 已有表征。

## 后续建议

第一轮建议只做小改动训练：

1. 以 v4 checkpoint 初始化。
2. 开启 `--use-snn-temporal-branch`。
3. 不加入 ADL-add 新增数据强权重。
4. 先用 Subject3 + 外部 val 选模型和阈值。
5. 确定唯一策略后再跑 Subject4 final。

目标不是单纯提高 ADL specificity，而是观察 SNN 分支是否能在不明显增加 ADL 误报的前提下提升 Fall recall，特别是对跌倒瞬间快速动态变化的捕捉能力。

下一步更合理的优化：

1. 做 `CNN+SNN v2`：冻结 CNN encoder/GRU 前 2-3 epoch，只训练 SNN 分支和融合 head。
2. 小范围扫描 `snn-threshold=0.45/0.55/0.65` 与 `snn-decay=0.75/0.82/0.90`，只用验证集判断。
3. 将原始 GMDCSA Subject1/2 纳入训练时，不应全量拼接所有窗口；需要增加最大窗口数限制、视频级采样或 streaming dataset，避免内存暴涨。
