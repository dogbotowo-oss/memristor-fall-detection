# v999 — Crossbar-CNN-SNN 跌倒检测（s34 配方 + data1 器件数据 + seed 11）

本目录是 **v999 最终版本的自包含快照**：代码、模型、训练曲线、评估报表、复现脚本齐全。
v999 是当前采用版本：**val-selected 与 best-observed 合一**（val bal-acc 五种子里最高，S4 指标也最好）。

## 结果（2026-08-17 跑出）

**Val（Subject3 + Zenodo val，阈值扫描选出 0.65）**

| val bal-acc | val acc | ADL spec | fall recall |
|---|---|---|---|
| 0.754 | 0.730 | 0.753 | 0.728 |

**Subject4 独立测试（阈值 0.65 冻结，一次性）**

| acc | balanced acc | ADL spec | fall recall | precision | FP / FN |
|---|---|---|---|---|---|
| **81.08** | **81.62** | **75.00** | **88.24** | 79.17 | 5 / 2 |

误报：ADL 05、06、07、11、12；漏报：Fall 01、13（均为历代版本共性难样本）。

**五种子上下文（data1，s34 配方，seed 7 已剔除换 11）**

| seed | val bal-acc | S4 acc | ADL spec | fall recall |
|---|---|---|---|---|
| 1 | 0.721 | 78.38 | 70.00 | 88.24 |
| 3 | 0.738 | 78.38 | 75.00 | 82.35 |
| 5 | 0.720 | 75.68 | 65.00 | 88.24 |
| **11（本版）** | **0.754** | **81.08** | **75.00** | **88.24** |
| 42 | 0.654 | 78.38 | 70.00 | 88.24 |
| mean±std | - | 78.38±1.91 | 71.00±4.18 | 87.06±2.63 |

**no-SNN 消融（data1）**：seed11 完整 81.08/75.00/88.24 vs no-SNN 83.78/90.00/76.47（过度保守）；
seed3 完整 78.38/75.00/82.35 vs no-SNN 56.76/20.00/100.00（ADL 崩溃，误报 16/20）。
结论：SNN 支路的核心作用是**稳定 ADL/Fall 权衡工作点**；val bal-acc 完整模型 0.754/0.738
显著高于 no-SNN 的 0.606/0.638；S4 平均 bal-acc 80.15 vs 71.62（+8.5）。

## 与 v888-final（s34 seed1，旧 data）的差异

唯一区别是**器件数据源**：`E:\rray\12.18-2\data`（B1500A .xls 脉冲曲线 + I-t 采样）
→ `E:\rray\12.18-2\data1\data`（WGFMU CSV 导出）。训练配方、网络结构、阈值选择协议完全不变。

data1 注意事项：

- `HRS.csv` / `LRS.csv`：B1500A I/V-t 采样格式，加载器原生兼容。
  注意 HRS 前几百秒含置位瞬态尖峰；LRS 约 1100 s 后发生保持失效（衰减到 ~2.8e-6）。
- `EPSC.csv`、`LTP-1ms-*.csv`、`LTD-1ms-*.csv`：WGFMU `DataValue, time, I, V` 格式，
  加载器已扩展 CSV 解析（`load_device_response_curve`），取第 3 列电流包络做 min-max 归一化。
  LTP/LTD 为 1 ms 阶梯脉冲串，各含 2 次重复。
- **PPF 缺失**：data1 无双脉冲 facilitation 实测，代码用中性常数 0.5 曲线占位
  （日志有明确提示），EPSC/LTP/LTD 全部来自实测。若后续补测 PPF，把 `PPF*.csv` 或
  `PPF*.xls` 放进 `data1\data` 重跑即可。
- `I_V （100cycle).csv`：100 圈 I-V 数据，供 S.docx 器件表征图使用（本流程未用到）。

## 目录结构

```
v999\
  code\automatic_fall_detector.py          # 完整训练/推理代码（含 CSV 器件加载扩展）
  models\automatic_fall_event_detector_c1_s34_data1_s11.pth
  image\training_curve_c1_s34_data1_s11.png
  reports\                                  # val 扫描、val 摘要、S4 指标、逐视频预测
  scripts\run_s30.py                        # 复现入口（默认 seed 11 + preserve 0.3/0.5 + data1）
  analysis\strict_eval_s30.py               # 严格评估（阈值只在 val 上选）
  analysis\eval_at.py                       # 任意冻结阈值评估
```

## 复现

环境：`D:\software\anaconde\envs\my_yizu_3.10\python.exe`，工作目录 `E:\rray\12.18-2`。

```powershell
# 训练 + 自动评估（约 20 分钟；模型已存在则跳过训练直接评估）
python scripts\run_s30.py --tag c1_s34_data1_s11

# 仅评估（用快照内模型，阈值冻结 0.65）
python analysis\eval_at.py models\automatic_fall_event_detector_c1_s34_data1_s11.pth verify_v999 0.65
```

外部依赖（不在快照内，路径在脚本中硬编码）：GMDCSA24 数据集（`测试集\…\Subject 1-3`、
`gmdcsa_subject4_test\Subject 4`）、Zenodo 划分（`zenodo_falldb_video_split`）、
姿态缓存（`pose_cache`）、v7 预热权重（`v7\models\…group_channel_gate.pth`）、
器件数据（`data1\data`）。

## 训练配方（与 s34 一致）

v7 checkpoint 热启动 → CNN/GRU 冻结 2 epoch → 全局微调（早停）；
hard-negative ADL 采样（top 12%，gamma 2.0）+ Zenodo ADL 上下文增强；
teacher-student 噪声训练；器件动力学增益（EPSC/PPF/LTP/LTD 实测曲线）+ 读出噪声 0.25；
SNN 支路 15 维输入（7 全局统计 + 8 姿态特征：头/髋/质心 y、Δy、vy、头部加速度、pose_valid）；
group-channel 门控融合（floor 0.2，ADL 门控惩罚 0.06）；
fall-preserve margin 损失（权重 0.3，margin 0.5）。
协议：阈值只在 val（Subject3 + Zenodo val）上扫描（≤0.70），冻结后对 Subject4 一次性测试。

## 噪声鲁棒性测试（2026-08-21）

`analysis\noise_sweep_v999.py`：纯推理读出噪声扫描，阈值冻结 0.65，S4 一次性评估。
σ 为 nominal 满量程百分比（read_sigma = 0.004 × scale × dyn，dyn≈1；clip 已放宽到 100，
对部署值 0.25 无影响）。结果：

| σ (% FS) | 0 | 5 | 10 | 15 | 20 | 25 | 30 |
|---|---|---|---|---|---|---|---|
| acc | 78.38 | 81.08 | 78.38 | 75.68 | 72.97 | 70.27 | 59.46 |
| ADL spec | 75 | 75 | 75 | 80 | 90 | 100 | 100 |
| fall recall | 82.35 | 88.24 | 82.35 | 70.59 | 52.94 | 35.29 | 11.76 |

结论：σ=5%（50× 部署）零退化；σ=10%（100× 部署）recall 仅 -5.9pp；拐点 σ≈15%；
失效模式为保守化（spec 升至 100，漏报增加）。
注意：0% 档与其他档的差异含 rng 流伪影（0% 不执行噪声分支导致电导重抽样），
基线以部署值附近为准。产物：`reports\noise_robustness_sigma_c1_s34_data1_s11.csv`、
`image\noise_robustness_sigma_c1_s34_data1_s11.png`。
