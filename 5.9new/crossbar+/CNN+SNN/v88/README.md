# v88 版本说明

v88 = S3 训练方案 + 权重 EMA + SNN late fusion（lambda=0.2）。
定位：当前 SNN 融合研究线的最优候选版本。正式基线 v8 p006 保持不动，
本版本全部文件独立存放在本目录，不与 v8-final 混用。

## 路径

- 版本根目录：`E:\rray\12.18-2\5.9new\crossbar+\CNN+SNN\v88`
- 模型：`models\automatic_fall_event_detector_v88.pth`
- 主代码：`code\automatic_fall_detector.py`
- 运行器（gate 细化 + late fusion）：`code\run_refined_gate_r2.py`
- 验证脚本：`code\verify_v88.py`
- 训练来源：实验目录 `v8-0729-fusion-gate-r2\models\c1_ema0999_s3` 的
  EMA checkpoint（epoch 5 最优），加 late fusion 后转为本版本

## 模型定义

CNN + 双向 GRU + SNN（LIF 时序支路）+ group-only 融合门控 + SNN 类别头
late fusion。最终 logits = 主 head logits + 0.2 * snn_class_head logits
（仅作用于 normal / fall 两个类别的 logit 位）。

## 关键参数

训练（继承 C1/S1/S3 方案，seed=42，v7 checkpoint 初始化）：

- 优化：lr=8e-4，weight_decay=1e-4，batch_size=8，epochs=6，label_smoothing=0.05
- 三阶段冻结：epoch1-2 冻结 CNN/GRU（只训 SNN/gate/head），epoch3-4 只解冻 GRU，
  epoch5-6 全量解冻
- 权重 EMA：decay=0.999（逐步更新，每 epoch 评估并保存最优 EMA checkpoint）
- temporal 辅助监督（impact / center drop / posture change / ground hold）：权重 0.1
- 难负样本 ADL 加权采样：weight=1.10，quantile 0.88，gamma 2.0
- teacher-student 噪声一致性：std=0.035，prob=0.75，frame_drop=0.08

SNN 支路与门控：

- LIF：snn_hidden_dim=32，decay=0.82（可学），threshold=0.30，surrogate_scale=8.0
- SNN 输入：7 维手工标量特征（mass / motion / center_y / center_drop /
  soft_height / aspect / fall_event_like）
- 融合门控：group-only（mean / max / last 三组），初始 bias=0.0
- gate context 已移除 fall_event_score（诊断 AUC=0.505，近随机）
- ADL 惩罚：snn_adl_gate_penalty=0.06，作用点为 snn_pooled（压 SNN 放电本身，
  不再压 gate）
- SNN 类别辅助头：snn_class_head，交叉熵权重 0.3
- late fusion：lambda=0.2（推理时内置，见 runner 的 forward_with_gate）

器件与 Crossbar：

- Crossbar、HRS/LRS、device dynamics（EPSC / PPF / LTP / LTD）全部开启
- crossbar_readout_noise_scale=0.25，device_noise_scale=1.0
- 器件数据目录：`E:\rray\12.18-2\data`

## 数据口径

- 图像 64x64，时序窗口 16 帧，external-labeled-stride=16
- 训练集 3539 窗口（ADL 1758 / Fall 1781）：
  GMDCSA Subject1-2 + `E:\rray\12.18-2\zenodo_falldb_video_split\train`
- 验证集 1076 窗口（ADL 573 / Fall 503）：
  GMDCSA Subject3 + `E:\rray\12.18-2\zenodo_falldb_video_split\val`
- GMDCSA 根目录：
  `E:\rray\12.18-2\测试集\13354453\ekramalam\GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-v2.1\ekramalam-GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-5abac76`
- Subject4 仅允许用于最终独立测试，不得用于调参

## 验证集结果（Subject3 + Zenodo val，训练内 argmax 口径）

| 指标 | v88 | 备注 |
|---|---|---|
| Balanced Accuracy | 0.7505 | 主 head 单独为 0.7512，差异在噪声内 |
| ADL specificity | 0.7574 | |
| Fall recall | 0.7435 | |
| TP/TN/FP/FN | 374/434/139/129 | |

SNN 通路状态（详细分析见 v8-0729-fusion-gate-r2\analysis）：

- LIF 已激活，最佳 fall 选择性通道 AUC=0.710
- SNN 类别头单独分类：bal=0.6919，spec=0.6562，recall=0.7276
- late fusion lambda=0.2 为固定值（与 0.1 / 0.3 对比后选定，非阈值扫描）

## 复现验证

```powershell
D:\software\anaconde\envs\my_yizu_3.10\python.exe `
  E:\rray\12.18-2\5.9new\crossbar+\CNN+SNN\v88\code\verify_v88.py
```

预期输出：`bal=0.7505 spec=0.7574 rec=0.7435`，
`TP/TN/FP/FN = 374/434/139/129`。

## 环境

- Python：`D:\software\anaconde\envs\my_yizu_3.10\python.exe`
- torch 2.5.1+cu121，GPU 推理/训练均可（CPU 可跑推理）
