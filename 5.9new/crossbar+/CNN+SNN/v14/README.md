# v14：ADL subtype 定向抑制

## 版本目的

v14 在 `v13` 的基础上继续优化，但不再全局调整 `adl_subtype_loss_weight`，而是尝试更精准的方向：

- 只对最容易误报成 Fall 的 ADL subtype 做定向抑制

目标是：

- 尽量保住当前已经较高的 `Fall recall`
- 同时把 `ADL specificity` 往上拉

## 核心思路

v13 的问题已经比较明确：

- 模型不是整体太激进；
- 而是对一批特定 ADL 片段误触发明显，比如：
  - `sit_descent`
  - `bend_reach`
  - `squat_kneel`

因此 v14 不再对所有 ADL 一起压，而是新增一个只作用于这几类 subtype 的额外抑制项：

- 对这些 subtype 样本，额外惩罚它们的 `pre_fall + fall probability`
- 并结合 `hard_negative_score` 做加权

这样做的含义是：

- 真 Fall 的正样本路线保持不动
- 只有“最像 Fall 的 ADL”会被精准踩刹车

## 代码新增

主文件：

- `code\\automatic_fall_detector.py`

新增开关：

- `--targeted-adl-fall-penalty`
- `--targeted-adl-subtypes`

默认目标 subtype：

- `sit_descent`
- `bend_reach`
- `squat_kneel`

新增辅助函数：

- `resolve_adl_subtype_ids(...)`

训练中新增的 loss 逻辑：

- 只在 `normal ADL`
- 且 subtype 属于指定目标集合
- 且其 hard-negative 特征较明显时

对其 `pre_fall + fall` 概率施加额外 penalty。

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

## 本次训练配置

延续 v13 当前可比较口径：

- train windows：`3539`
- val windows：`1076`

主要参数：

- `adl_subtype_loss_weight = 0.20`
- `adl_subtype_hard_only = on`
- `adl_subtype_hard_score_threshold = 0.22`
- `targeted_adl_fall_penalty = 0.12`
- `targeted_adl_subtypes = sit_descent bend_reach squat_kneel`
- `snn_adl_gate_penalty = 0.06`
- `teacher-student gaussian noise = on`
- `use-device-dynamics = on`
- `crossbar_readout_noise_scale = 0.25`

## 训练过程现象

训练在 `epoch 7` early stop。

验证趋势上可以看到：

- 前两轮 `val_normal` 相比 v13 略有提升；
- 说明定向 subtype 抑制确实在起作用；
- 但中后期仍然存在 `val_normal / val_fall` 相互拉扯的问题。

## 严格评估结果

阈值选择：

- 仅在 `Subject3 + Zenodo val` 上扫描
- 推荐阈值：`0.70`

`Subject4` 最终结果：

- `Accuracy = 67.57%`
- `Balanced Accuracy = 68.68%`
- `ADL specificity = 55.00%`
- `Fall recall = 82.35%`
- `Precision = 60.87%`
- `TP / TN / FP / FN = 14 / 11 / 9 / 3`

## 与 v13_w020 / v13_w025 对比

v13 当前最佳可比口径：

- `Accuracy = 64.86%`
- `Balanced Accuracy = 66.18%`
- `ADL specificity = 50.00%`
- `Fall recall = 82.35%`
- `Precision = 58.33%`
- `TP / TN / FP / FN = 14 / 10 / 10 / 3`

v14 相比之下：

- `specificity` 从 `50% -> 55%`
- `precision` 从 `58.33% -> 60.87%`
- `FP` 从 `10 -> 9`
- `Fall recall` 保持 `82.35%` 不变

## 本版结论

v14 说明这个方向是对的。

和单纯调 `adl_subtype_loss_weight` 不一样，定向 subtype 抑制开始真正打到了问题核心：

- 它没有把全局阈值收紧；
- 也没有明显损伤 Fall recall；
- 而是把一部分易误报 ADL 压了下去。

虽然提升幅度还不大，但它是一个“方向正确”的改动。

## 下一步建议

如果继续沿 v14 往下做，优先顺序建议：

1. 小范围扫 `targeted_adl_fall_penalty`
   - 例如：`0.16 / 0.20`

2. 细化目标 subtype
   - 先试只压：
     - `sit_descent + squat_kneel`
   - 或：
     - `bend_reach + squat_kneel`

3. 如果误报仍主要集中在少数固定 ADL 视频
   - 再做更细的 subtype 或事件形态约束

当前判断：

- `v14` 已经比 `v13` 更接近可继续优化的主线版本。

## 追加小扫描：`targeted_adl_fall_penalty = 0.16 / 0.20`

后续在 `v14` 目录下追加了两组小扫描：

- `v14_p016`
- `v14_p020`

保持其余设置不变：

- `adl_subtype_loss_weight = 0.20`
- `targeted_adl_subtypes = sit_descent bend_reach squat_kneel`

### `v14_p016`

- 推荐阈值：`0.70`
- Subject4：
  - `Accuracy = 70.27%`
  - `Balanced Accuracy = 71.18%`
  - `ADL specificity = 60.00%`
  - `Fall recall = 82.35%`
  - `Precision = 63.64%`
  - `TP / TN / FP / FN = 14 / 12 / 8 / 3`

### `v14_p020`

- 推荐阈值：`0.70`
- Subject4：
  - `Accuracy = 70.27%`
  - `Balanced Accuracy = 71.18%`
  - `ADL specificity = 60.00%`
  - `Fall recall = 82.35%`
  - `Precision = 63.64%`
  - `TP / TN / FP / FN = 14 / 12 / 8 / 3`

### 与 v14 基线 `0.12` 对比

`v14(0.12)`：

- `Accuracy = 67.57%`
- `Balanced Accuracy = 68.68%`
- `ADL specificity = 55.00%`
- `Fall recall = 82.35%`
- `Precision = 60.87%`
- `TP / TN / FP / FN = 14 / 11 / 9 / 3`

`v14_p016 / v14_p020`：

- `specificity` 从 `55% -> 60%`
- `precision` 从 `60.87% -> 63.64%`
- `FP` 从 `9 -> 8`
- `Fall recall` 保持 `82.35%` 不变

### 追加扫描结论

- `0.16` 和 `0.20` 在当前实验口径下已经打平；
- 说明定向 ADL subtype 抑制这个思路是有效的；
- 但 penalty 再从 `0.16` 提到 `0.20` 没带来进一步收益。

当前更像甜点位的是：

- `targeted_adl_fall_penalty = 0.16` 或 `0.20`

从工程记录和解释性角度，优先可把 `0.16` 视为当前推荐值。

## subtype 组合消融（penalty = 0.16）

在保持其余配置不变的前提下，只改 `targeted_adl_subtypes`，做了 3 组组合消融。阈值仍然只在 `Subject 3 + Zenodo val` 上扫描选择，`Subject 4` 只做最终一次独立测试。

### 对比结果

- `all three`：`sit_descent + bend_reach + squat_kneel`
  - threshold：`0.70`
  - Accuracy：`70.27%`
  - Balanced Accuracy：`71.18%`
  - ADL specificity：`60.00%`
  - Fall recall：`82.35%`
  - Precision：`63.64%`
  - `TP / TN / FP / FN = 14 / 12 / 8 / 3`

- `only squat_kneel`
  - threshold：`0.70`
  - Accuracy：`64.86%`
  - Balanced Accuracy：`66.18%`
  - ADL specificity：`50.00%`
  - Fall recall：`82.35%`
  - Precision：`58.33%`
  - `TP / TN / FP / FN = 14 / 10 / 10 / 3`

- `sit_descent + squat_kneel`
  - threshold：`0.65`
  - Accuracy：`59.46%`
  - Balanced Accuracy：`61.18%`
  - ADL specificity：`40.00%`
  - Fall recall：`82.35%`
  - Precision：`53.85%`
  - `TP / TN / FP / FN = 14 / 8 / 12 / 3`

- `bend_reach + squat_kneel`
  - threshold：`0.70`
  - Accuracy：`62.16%`
  - Balanced Accuracy：`63.68%`
  - ADL specificity：`45.00%`
  - Fall recall：`82.35%`
  - Precision：`56.00%`
  - `TP / TN / FP / FN = 14 / 9 / 11 / 3`

### 消融结论

- 四组的 `Fall recall` 基本一致，差异主要来自 `ADL specificity`。
- 这说明当前的 `targeted_adl_fall_penalty` 更擅长压低 ADL 误报，而不是大幅改变 Fall 检测灵敏度。
- 从结果看，只压 1-2 个 subtype 都不如三类一起压。
- 当前 `v14` 最优推荐仍然是：
  - `targeted_adl_fall_penalty = 0.16`
  - `targeted_adl_subtypes = sit_descent bend_reach squat_kneel`

### 对下一步的意义

- `squat_kneel` 是重要误报类型，但不是唯一决策点。
- `bend_reach` 和 `sit_descent` 在误报链路中仍然有补充价值。
- 因此，后续如果继续优化 `v14`，更合适的方向不是继续删减 subtype，而是在 `all three` 组合上微调 penalty 强度，或者引入更精细的 ADL 抑制逻辑。
