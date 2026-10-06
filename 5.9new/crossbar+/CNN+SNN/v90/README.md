# v90 版本说明与冻结记录

## 1. 版本定位

v90 是 S10 的独立归档版本，用于完整保存以下候选配置：S10 best-EMA
checkpoint、在 Subject3 + Zenodo val 上选定的 `stride=4 peak >= 0.90`
视频级规则，以及该规则的一次性 Subject4 独立测试结果。

- v90 根目录：`E:\rray\12.18-2\5.9new\crossbar+\CNN+SNN\v90`
- 正式基线：v8 p006，未被本版本修改、覆盖或替换。
- 研究版本：v88、v89、v90 均独立保存，不能与 `v8-final` 主代码混用。
- v90 不是新的正式基线。它在验证集视频级达标，但 Subject4 独立测试未迁移，
  该负结果必须一并保留和报告。

## 2. 文件与代码入口

| 路径 | 用途 |
|---|---|
| `models\automatic_fall_event_detector_v90.pth` | S10 best weight-EMA 模型，source epoch 6 |
| `code\automatic_fall_detector.py` | CNN-GRU-SNN 主模型、数据加载、Crossbar 与器件动态实现 |
| `code\run_refined_gate_r2.py` | group gate 细化和 late fusion 运行器 |
| `code\run_s9_temporal_speed.py` | S10 使用的 fall 时域减速/翻转训练增广运行器 |
| `code\evaluate_subject4_once.py` | 仅用于本冻结规则的一次性 Subject4 测试脚本 |
| `reports\frozen_config.json` | 冻结模型、阈值和数据隔离审计记录 |
| `reports\subject4_frozen\` | 一次性 Subject4 的逐视频 CSV 与汇总 JSON |

## 3. 运行环境

- Python：`D:\software\anaconde\envs\my_yizu_3.10\python.exe`（Python 3.10）
- PyTorch：2.5.1+cu121，CUDA GPU 训练与推理环境
- 依赖：`torch`、`numpy`、`opencv-python` (`cv2`)、`pandas`、`matplotlib`、
  Excel 器件曲线读取依赖（`openpyxl` / `xlrd`）。
- 器件数据目录：`E:\rray\12.18-2\data`
- 器件数据：HRS/LRS conductance，EPSC、PPF、LTP、LTD 响应曲线。

运行时必须在日志确认：

```text
external_labeled_windows=...
external_val_labeled_windows=1076
Loaded conductance data
Loaded device dynamics
```

## 4. 模型与前向逻辑

模型由 CNN、双向 GRU 和 LIF-SNN 时序支路组成。

1. 输入视频窗口先由 CNN 提取逐帧空间特征，再由双向 GRU 建模主时序特征，
   形成主分类 head 的 logits。
2. 同一窗口并行输入 SNN。S10 的 SNN 输入为 `scalar7_flow`：
   7 个形态/时序事件量（mass、motion、center_y、center_drop、soft_height、
   aspect、fall_event_like）加 128 维有向 Lucas-Kanade 光流场。
3. SNN 经过可学习衰减和阈值的 LIF 神经元，随后用 `seg4x3` 做四段
   mean/max/last 池化，保留事件的早、中、晚时序状态。
4. group-only gate 按三组池化语义调节 SNN 特征；gate context 中已移除
   `fall_event_score`。初始 group bias 为 0.0，gate floor 为 0.2。
5. 门控 SNN 特征与 CNN-GRU 主特征拼接进入主 head；SNN 同时有独立的
   `snn_class_head`。最终前向使用 late fusion：
   `final logits = main logits + 0.2 * snn_class_head logits`（normal/fall 位）。

训练时，主交叉熵、SNN 类别辅助损失和 ADL 惩罚都对 SNN 支路产生梯度：

- `snn_class_aux_weight=0.3`
- `temporal_aux_loss_weight=0.1`，辅助目标为 impact、center drop、posture
  change、ground hold
- `snn_adl_gate_penalty=0.06`，作用点为 `snn_pooled`

Crossbar、HRS/LRS 与 EPSC/PPF/LTP/LTD 器件动态全部开启；
`crossbar_readout_noise_scale=0.25`，`device_noise_scale=1.0`。

## 5. S10 训练配置

- 输入：64x64；窗口：16 帧；训练与窗口级验证 stride：16
- 初始模型：v7 checkpoint
- 优化：batch size 8，lr `8e-4`，weight decay `1e-4`，label smoothing `0.05`
- 权重 EMA：decay `0.999`，v90 使用按验证 balanced accuracy 选中的 epoch 6 EMA
- 三阶段训练：epoch 1-2 冻结 CNN/GRU；epoch 3-4 仅解冻 GRU；epoch 5-6 全量解冻
- SNN：hidden dim 32，decay 0.82（可学习），threshold 0.30，surrogate scale 8.0
- SNN 输入/池化：`scalar7_flow` / `seg4x3`
- fusion：group-only，floor 0.2，group bias 0.0，late fusion lambda 0.2
- teacher-student：Gaussian noise std 0.035，prob 0.75，frame drop 0.08
- hard-negative ADL：weight 1.10，quantile 0.88，gamma 2.0

## 6. 数据来源与严格隔离

| 集合 | 内容 | 使用方式 |
|---|---|---|
| 原始 train | GMDCSA Subject1-2 + Zenodo train，3539 窗口（ADL 1758 / Fall 1781） | 训练 |
| UP-Fall subject11 | 33 个视频，1114 窗口（ADL 950 / Fall 164） | 仅训练 |
| S10 增广 | 原始与 UP-Fall fall 窗口的 `slow_flip_half`，新增 1948 fall 窗口 | 仅训练 |
| S10 最终 train | 6601 窗口（ADL 2708 / Fall 3893） | 训练 |
| protocol val | GMDCSA Subject3 + Zenodo val，1076 窗口（ADL 573 / Fall 503） | 选模型和阈值 |
| Subject4 | 20 ADL + 17 Fall 视频 | 一次性独立测试 |

UP-Fall 只放在
`v8-0729-fusion-gate-r2\external_upfall\train\{adl,fall}`，未进入 val 或
Subject4。Subject4 从未参与 S10 训练、阈值选择或模型选择。

## 7. 冻结视频级决策规则

这是在 Subject3 + Zenodo val 的 61 个视频上确定的唯一规则：

```text
16-frame window, stride=4
video score = max(fall probability of all windows)
predict fall when video score >= 0.90
```

该阈值是验证集选择结果；迁移到 Subject4 后不得重新扫描、调高或调低。

## 8. 结果

### 8.1 Subject3 + Zenodo val

窗口级（stride 16，离线确定性评估）：

| 指标 | S10 / v90 |
|---|---:|
| Accuracy | 0.7193 |
| Balanced Accuracy | 0.7271 |
| ADL specificity | 0.6073 |
| Fall recall | 0.8469 |
| TP/TN/FP/FN | 426/348/225/77 |
| SNN auxiliary AUC | 0.6982 |

视频级（stride-4 peak，61 视频，冻结阈值 0.90）：

| 指标 | S10 / v90 |
|---|---:|
| Accuracy | 0.8197 |
| Balanced Accuracy | 0.8200 |
| ADL specificity | 0.8125 |
| Fall recall | 0.8276 |
| TP/TN/FP/FN | 24/26/6/5 |

### 8.2 Subject4 一次性独立测试

固定使用上节规则，未做任何 Subject4 调参：

| 指标 | Subject4 |
|---|---:|
| Accuracy | 0.5405 |
| Balanced Accuracy | 0.5397 |
| ADL specificity | 0.5500 |
| Fall recall | 0.5294 |
| TP/TN/FP/FN | 9/11/9/8 |

逐视频分数在 `reports\subject4_frozen\subject4_video_metrics.csv`，汇总在
`reports\subject4_frozen\subject4_summary.json`。此结果说明 S10 的验证集
视频级收益未迁移到 Subject4，因此 v90 只能作为完整候选/负结果归档，不能取代
v8 p006 的最终独立测试记录。

## 9. 注意事项

1. 严禁修改 v8 p006 / v8-final 主代码和正式模型。
2. 禁止把 UP-Fall、Subject3 或 Subject4 混入其他集合；窗口不能跨 subject 切分。
3. v90 的 Subject4 测试已执行一次，不能再将 Subject4 用于阈值、gate、loss 或
   输入特征调参。
4. 不可把 v90 的 0.90 阈值解释为跨域通用阈值，它只是在 protocol val 上选定的
   video-level operating point。
5. S11 是后续独立实验，位于
   `v8-0729-fusion-gate-r2\models\c1_ema0999_s11kin`；它不会覆盖 v90。

## 10. 当前研究进度（2026-08-06）

- v8 p006：正式基线，保持冻结。
- v88：当前 SNN 研究线的窗口级候选。
- v89：时域减速增广的 recall 取向候选。
- v90：S10 + UP-Fall 的完整归档，已完成一次性 Subject4 测试，Subject4 未达标。
- S11：训练与固定验证集离线评估已完成。它仅在 SNN 输入上将 `scalar7_flow`
  扩展为 `scalar7_flow_kinematic`（135 -> 143 维），新增带符号的运动学增量；
  训练、验证与数据隔离条件与 S10 相同。S11 窗口级 `bal=0.7241`、
  `specificity=0.5934`、`recall=0.8549`，SNN-aux AUC=`0.7131`，较 S10
  的 `0.6982` 提升；视频级最佳单模型工作点为 `threshold=0.875`，
  `accuracy=0.7705`、`specificity=0.7188`、`recall=0.8276`，未达到 S10
  在验证集上的 `accuracy=0.8197`。S11 不替代 v90，也未进行新的 Subject4
  测试；完整评估日志在
  `v8-0729-fusion-gate-r2\analysis\s11kin_eval.log`。
