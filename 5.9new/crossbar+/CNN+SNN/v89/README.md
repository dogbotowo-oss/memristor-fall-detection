# v89 版本说明

v89 = v88 全部配置 + 时域减速增广（slow_flip_half，等效增广权重 0.5）。
定位：SNN 融合研究线第二候选版本（recall 取向），冠军仍为 v88。
正式基线 v8 p006 保持不动；本版本全部文件独立存放在本目录，不与 v8-final 混用。

## 路径

- 版本根目录：`E:\rray\12.18-2\5.9new\crossbar+\CNN+SNN\v89`
- 模型：`models\automatic_fall_event_detector_v89.pth`
- 主代码：`code\automatic_fall_detector.py`
- 运行器（gate 细化 + late fusion）：`code\run_refined_gate_r2.py`
- 时域减速增广加载器（复现训练用）：`code\run_s9_temporal_speed.py`
- 验证脚本：`code\verify_v89.py`
- 训练来源：实验目录 `v8-0729-fusion-gate-r2\models\c1_ema0999_s9cwt` 的
  best-EMA checkpoint（epoch 3，训练内 val_bal=0.752）

## 模型定义

与 v88 相同：CNN + 双向 GRU + SNN（LIF 时序支路），group-only 融合门控 +
SNN 类别头 late fusion。最终 logits = 主 head logits + 0.2 * snn_class_head logits
（仅作用于 normal / fall 两个类别的 logit 位）。

## 与 v88 的唯一差异：训练数据增广

- 对 fall 窗口施加 0.5x 时域减速（可选水平翻转），slow 与 slow+flip 两种变体
  各按 50% 概率采样（slow_flip_half），等效增广权重 0.5
- 训练窗口 5331（normal 1758 / fall 3573，约 2:1）；验证集不增广，保持 1076
- 目的：修复慢跌倒盲区（Subject3 的 06/07/09/10），提升 fall recall

## 关键参数（未列出的与 v88 一致）

训练（seed=42，v7 checkpoint 初始化）：
- 优化：lr=8e-4，weight_decay=1e-4，batch_size=8，epochs=6，label_smoothing=0.05
- 三阶段冻结：epoch1-2 冻结 CNN/GRU（只训 SNN/gate/head），epoch3-4 只解冻 GRU，
  epoch5-6 全量解冻
- 权重 EMA：decay=0.999（逐步更新，按 val balanced accuracy 存档最优 EMA，
  本版本存档于 epoch 3）
- temporal 辅助监督（impact / center drop / posture change / ground hold）：权重 0.1
- 难负样本 ADL 加权采样：weight=1.10，quantile 0.88，gamma 2.0
- teacher-student 噪声一致性：std=0.035，prob=0.75，frame_drop=0.08

SNN 支路与门控：
- LIF：snn_hidden_dim=32，decay=0.82（可学），threshold=0.30，surrogate_scale=8.0
- SNN 输入：scalar7_flow（7 维手工标量特征 + 跨帧光流动态），池化 seg4x3
- 融合门控：group-only（mean / max / last 三组），初始 bias=0.0
- gate context 已移除 fall_event_score（诊断 AUC=0.505，近随机）
- ADL 惩罚：snn_adl_gate_penalty=0.06，作用点为 snn_pooled
- SNN 类别辅助头：snn_class_head，交叉熵权重 0.3
- late fusion：lambda=0.2（推理时内置，见 runner 的 forward_with_gate）

器件与 Crossbar：
- Crossbar、HRS/LRS、device dynamics（EPSC / PPF / LTP / LTD）全部开启
- crossbar_readout_noise_scale=0.25，device_noise_scale=1.0
- 器件数据目录：`E:\rray\12.18-2\data`

## 数据口径

- 图像 64x64，时序窗口 16 帧，external-labeled-stride=16
- 训练集 3539 窗口（ADL 1758 / Fall 1781）+ 增广 1792 窗口 = 5331：
  GMDCSA Subject1-2 + `E:\rray\12.18-2\zenodo_falldb_video_split\train`
- 验证集 1076 窗口（ADL 573 / Fall 503）：
  GMDCSA Subject3 + `E:\rray\12.18-2\zenodo_falldb_video_split\val`
- Subject4 仅允许用于最终独立测试，不得用于调参

## 验证集结果（离线确定性口径）

窗口级（stride16，argmax 口径，与 v88 官方一致，late fusion lambda=0.2）：

| 指标 | v89 | v88（对照） |
|---|---|---|
| Accuracy | 0.7491 | 0.7509 |
| Balanced Accuracy | 0.7519 | 0.7505 |
| ADL specificity | 0.7086 | 0.7574 |
| Fall recall | 0.7952 | 0.7435 |
| TP/TN/FP/FN | 400/406/167/103 | 374/434/139/129 |

注：S 系列横向对比用的 softmax-fall>=0.5 口径下，v89 为
acc=0.7454 bal=0.7465 spec=0.7295 rec=0.7634（TP/TN/FP/FN=384/418/155/119）。

SNN 通路状态：snn-aux AUC=0.7122，corr(main,snn)=0.5148。

视频级（stride-4 稠密推理，peak 规则，61 视频 = 29 fall / 32 ADL）：

| 工作点 | acc | spec | rec |
|---|---|---|---|
| thr=0.70（recall 取向） | 0.7541 | 0.6562 | 0.8621 |
| thr=0.825（最均衡） | 0.7541 | 0.7500 | 0.7586 |

慢跌倒盲区峰值（v88 -> v89）：06: 0.36->0.71，07: 0.22->0.60，
09: 0.55->0.83，10: 0.53->0.86；17.mp4 仍盲（0.08）。

## 复现验证

```powershell
D:\software\anaconde\envs\my_yizu_3.10\python.exe `
  E:\rray\12.18-2\5.9new\crossbar+\CNN+SNN\v89\code\verify_v89.py
```

预期输出：`bal=0.7519 spec=0.7086 rec=0.7952`，`TP/TN/FP/FN = 400/406/167/103`。

## 环境

- Python：`D:\software\anaconde\envs\my_yizu_3.10\python.exe`
- torch 2.5.1+cu121，GPU 推理/训练均可（CPU 可跑推理）
