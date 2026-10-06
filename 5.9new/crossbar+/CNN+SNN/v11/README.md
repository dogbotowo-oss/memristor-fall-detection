# v11：ADL subtype-aware gate suppression

## 版本目标

v11 基于 `v8 final` 和 `v10` 继续优化：

- 不再把所有 ADL 都当成同一种 hard negative
- 将容易混淆的 ADL 拆成子类型
- 只对 gate penalty 做子类型细分，尽量保住真 Fall 召回

## 核心思路

本版把 normal ADL 分成 3 类：

1. `low_posture`
2. `bend`
3. `controlled_descent`

训练时：

- 先算 `hard_negative_score`
- 再算 3 个 subtype score
- 用加权后的 subtype score 调整 `snn_adl_gate_penalty`

新增约束：

- `--adl-subtype-penalty-cap`

它给 penalty factor 加上上限，避免 ADL 被压得过重，导致真 Fall 的 gate 也被一起压掉。

## 当前实现位置

- 代码：`code\\automatic_fall_detector.py`

主要参数：

- `--adl-subtype-aware-gate-penalty`
- `--adl-subtype-penalty-scale`
- `--adl-subtype-penalty-cap`
- `--adl-low-posture-gate-weight`
- `--adl-bend-gate-weight`
- `--adl-controlled-descent-gate-weight`

## 环境

- Python：`F:\\ANACONDA\\envs\\my_yizu_3.10\\python.exe`
- 项目根目录：`F:\\12.18-2`
- 数据目录：`F:\\12.18-2\\data`

## 数据划分

- 训练：GMDCSA Subject1-2 + Zenodo train
- 验证：GMDCSA Subject3 + Zenodo val
- 最终测试：Subject4

Subject4 不参与调参，只做最终一次独立测试。

## 当前状态

- 已完成代码修改
- 已通过 `py_compile`
- 下一步建议：在 `Subject3 + Zenodo val` 上做小范围扫描，再决定是否固定该版本
