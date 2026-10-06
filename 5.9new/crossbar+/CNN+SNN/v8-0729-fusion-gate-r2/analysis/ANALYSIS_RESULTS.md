# SNN 通道分析与 ADL penalty 开关对照结论

日期：2026-07-30。分析对象：C1 checkpoint（group-only gate，val_bal 0.748）。
口径：Subject3+Zenodo val 1076 窗口，前向管线与训练时完全一致（base_seed=42+9999，augment off），
基线复现 bal=0.7484 / spec=0.7155 / rec=0.7813，与 C1 epoch3 训练日志一致。

## Step 1：通道可分性（channel_separability.csv）

- 96 个 SNN 通道中绝大多数 mean 激活为 0.0000——LIF 分支基本不放电，处于"死亡"状态。
- 最高 AUC 仅 0.68（LIF 通道 11，在 mean/max/last 三组中为同一通道的不同统计）。
- 三组平均 AUC 均 < 0.5，SNN 表征整体不携带可用类别信息。

## Step 1b：group gate 行为（group_gate_behavior.csv）

| group | gate(fall) | gate(adl) | gate(adl hard top10%) |
|---|---:|---:|---:|
| mean | 0.0006 | 0.0005 | 0.0009 |
| max | 0.0001 | 0.0001 | 0.0002 |
| last | 0.1107 | 0.0800 | **0.1991** |

- gate 已学会把 mean/max 组完全关闭，只有 last 组（末帧脉冲状态）有微弱通路。
- 最像跌倒的 ADL hard negative 上 last gate 反而最大（0.199 > fall 的 0.111）——误报通道。

## Step 2：掩码消融（channel_mask_ablation.csv）

- mask 任意单个通道：指标完全不变（delta=0.000）。
- mask 整个 mean 或 max 组：指标完全不变。
- mask 整个 last 组：bal 仅 -0.004。

**结论：C1 的 0.748 全部由 CNN/GRU 贡献，SNN 对决策的实际贡献约等于零。**
"哪些通道该压制"在当前 regime 下是伪问题——gate 已经把所有通道压死了。
真正的问题是上游：LIF 分支不放电，SNN 没有信号可传。

## Step 3：ADL penalty 开关对照（同 C1 配置，单变量）

| penalty | 最佳 epoch | val_bal | ADL spec | Fall recall |
|---:|---:|---:|---:|---:|
| 0.06（C1） | 3 | 0.748 | 0.716 | 0.781 |
| 0.00 | 3 | 0.744 | 0.709 | 0.779 |

penalty=0.06 带来 +0.4pt bal、+0.7pt spec，小而稳定的正收益。**保留 0.06，它不是瓶颈。**

## 总体结论与下一步方向

SNN 分支的失效链：LIF 不放电 → 表征无信息 → gate 学到全关 → 通道问题消失。
后续实验不应再做 gate 微调或通道压制，而应直接修复 LIF 放电：

1. 降低 threshold 初始化（0.55→0.3）或放大 snn_input 输出尺度；
2. 让 SNN 直接承担类别监督（当前 temporal aux 只监督启发式目标，不给 fall 梯度）；
3. 加固定旁路（如 0.3×snn_pooled 不过 gate），保证梯度流通。
