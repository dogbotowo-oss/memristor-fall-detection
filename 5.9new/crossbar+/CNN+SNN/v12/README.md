# v12：强分类多任务 ADL subtype 监督

## 版本目标

v12 在 `v9 final` 基础上改造，但方法蓝本来自 `v8` 这条稳定的 CNN+SNN 主线。

这一版不再只做：

- `Fall / Non-Fall` 主分类
- `hard negative` 采样
- `SNN gate penalty` 抑制

而是新增一个真正的 **ADL subtype 强监督分支**，让模型显式学习：

- `sit_descent`
- `bend_reach`
- `squat_kneel`
- `lie_ground`
- `other`

目标是把原来“按 ADL 难样本压误报”的弱约束，升级成“按 ADL 子类学习判别边界”的强分类路线。

## 核心思路

v12 的网络仍保留：

- CNN encoder
- GRU temporal
- SNN temporal branch
- group_channel fusion gate
- Crossbar 映射
- device dynamics
- teacher-student noise

新增内容：

1. `ADL subtype head`
2. ADL 窗口伪子类标签生成
3. subtype auxiliary loss

因此当前模型变成双任务：

1. 主任务：`Fall / running / pre_fall / normal`
2. 辅任务：`normal ADL subtype`

## ADL subtype 定义

当前 v12 使用 5 个 ADL 子类：

- `other`
- `sit_descent`
- `bend_reach`
- `squat_kneel`
- `lie_ground`

注意：

- 现在还不是人工逐视频精标
- 而是基于窗口动态与姿态变化做伪 subtype 标注
- 这是第一版低风险工程实现，先验证“强分类路线是否有效”

## 代码改动

主文件：

- `code\\automatic_fall_detector.py`

主要新增：

1. `ADL_SUBTYPE_ID_TO_NAME`
2. `compute_adl_subtype_scores`
3. `assign_adl_subtype_id`
4. `--use-adl-subtype-head`
5. `--adl-subtype-loss-weight`
6. `FallEventDetector.adl_subtype_head`
7. 训练时对 ADL 样本计算 subtype CE loss

## 数据划分

严格保持：

- 训练：GMDCSA Subject1 + Subject2 + Zenodo train
- 验证与阈值校准：GMDCSA Subject3 + Zenodo val
- 最终独立测试：Subject4

Subject4 不参与任何模型选择、阈值扫描或参数调整。

## 运行环境

- 项目根目录：`F:\\12.18-2`
- Python：`F:\\ANACONDA\\envs\\my_yizu_3.10\\python.exe`
- 数据目录：`F:\\12.18-2\\data`

## 计划训练配置

v12 训练继承 v9/v8 的稳定设置：

- `group_channel` gate
- `teacher-student gaussian noise`
- `--use-device-dynamics`
- `--crossbar-readout-noise-scale 0.25`
- `hard_negative_normal_weight = 1.00`
- `snn_adl_gate_penalty = 0.06`
- `image_size = 64`
- `external_labeled_stride = 16`

并新增：

- `--use-adl-subtype-head`
- `--adl-subtype-loss-weight 0.30`

初始化模型：

- `models\\automatic_fall_event_detector_cnn_snn_v9_final.pth`

## 当前状态

- 已完成 v12 工程骨架复制
- 已完成 subtype 多任务代码改造
- 已通过 `py_compile`
- 下一步：启动训练 -> Subject3 + Zenodo val 阈值扫描 -> Subject4 final
