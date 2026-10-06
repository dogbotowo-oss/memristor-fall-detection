# v5：CNN + SNN + Fusion Gate

## 版本目标

在 v4 的 CNN + GRU + SNN 动态分支基础上，增加一个可学习的 fusion gate，用来抑制不稳定的 SNN 干扰，让 SNN 更像辅助分支，而不是直接强行参与融合。

## 本版新增点

- 新增 `--use-snn-fusion-gate`
- gate 先根据 CNN/GRU 主干特征 + SNN 特征预测样本级权重
- 再用 `gate * SNN特征` 参与最终分类
- 目标是降低 SNN 在跨域场景下对主干的误导

## 7 个 SNN 输入维度

- `mass`
- `motion`
- `center_y`
- `center_drop`
- `soft_height`
- `aspect`
- `fall_event_like`

## 运行环境

- Python: `F:\ANACONDA\envs\my_yizu_3.10\python.exe`
- 项目根目录: `F:\12.18-2`

## 最终有效训练设置

本次有效训练使用：

- `--use-snn-temporal-branch`
- `--use-snn-fusion-gate`
- `--learnable-snn-dynamics`
- `--freeze-cnn-gru-epochs 2`
- `--teacher-student-noise`
- `--student-noise-mode gaussian`
- `--use-device-dynamics`
- `--crossbar-readout-noise-scale 0.25`
- `--hard-negative-normal-weight 1.10`
- `--hard-negative-min-percentile 0.88`
- `--hard-negative-score-gamma 2.00`
- `--normal-positive-penalty 0.00`
- `--image-size 64`
- `--external-labeled-stride 16`
- `--adl-context-min-percentile 0.95`
- `--adl-context-extra-copies 1`

说明：

- 前两次启动不是有效实验：一次未显式传入 `F:\12.18-2\data`，导致器件数据未加载；一次 `image-size=128` 且窗口数较多，触发内存不足。
- 最终有效实验已确认加载 `HRS/LRS/EPSC/PPF/LTP/LTD` 器件数据。
- `Subject4` 只用于最终一次测试，未参与阈值扫描。

## 训练数据统计

- 训练窗口：2508
  - normal: 1031
  - fall: 1477
- 验证窗口：564
  - normal: 197
  - fall: 367
- ADL context boost：24 个视频，额外加入 62 个核心窗口。

## 训练过程摘要

- Epoch 1：val_acc 74.60%，val_bal 69.10%，val_normal 50.80%，val_fall 87.50%
- Epoch 3：val_acc 71.80%，val_bal 72.20%，val_normal 73.60%，val_fall 70.80%
- Epoch 8：触发 early stopping

## 严格阈值扫描

阈值只在 `Subject3 + Zenodo val` 上扫描。

- 推荐阈值：0.35
- 注意：0.35、0.40、0.45 在验证集 balanced accuracy 上并列。
- 当前脚本按扫描顺序选择第一个最佳阈值，因此选中 0.35。

验证集最优区间：

- Accuracy: 70.49%
- Balanced Accuracy: 70.32%
- ADL Specificity: 60.00%
- Fall Recall: 80.65%
- Precision: 67.57%

## Subject4 最终测试

使用验证集推荐阈值 0.35，对 Subject4 做最终一次测试：

- Accuracy: 48.65%
- Balanced Accuracy: 50.29%
- ADL Specificity: 30.00%
- Fall Recall: 70.59%
- Precision: 46.15%
- TP / TN / FP / FN: 12 / 6 / 14 / 5

## 当前判断

Fusion gate 没有解决整体泛化问题。它提高了 Subject4 的 Fall recall，但 ADL specificity 明显下降，说明当前 gate 仍然没有学会在 ADL 场景下稳定压低 SNN 动态分支。

这个结果可以证明：

- SNN 动态信息确实能增强模型抓 Fall 的倾向。
- 但单纯的 sample-level gate 还不足以区分“真正跌倒动态”和“强运动 ADL”。
- 后续更值得尝试的是验证集内预定义 tie-break 策略、SNN gate 正则、或让 gate 输出类别相关/通道相关权重。

## 关键结果文件

- 代码：`code\automatic_fall_detector.py`
- 模型：`models\automatic_fall_event_detector_cnn_snn_v5_snn_gate.pth`
- 训练曲线：`image\training_curve_cnn_snn_v5_snn_gate.png`
- 阈值扫描：`reports\v5_strict_validation_threshold_scan.csv`
- 推荐阈值：`reports\v5_recommended_threshold.json`
- Subject4 最终结果：`reports\v5_subject4_final_metrics.json`
