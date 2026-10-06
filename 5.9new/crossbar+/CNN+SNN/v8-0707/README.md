# v8-0707：SNN 分支消融实验

## 版本目的

本版本用于回答一个更关键的问题：

`v8 final` 虽然是当前综合最优版本，但从论文表达上看，SNN 接入后的改进幅度还不够直观。  
因此本轮不再继续先画图，而是先补齐一组严格同口径的 SNN 分支消融实验，明确回答：

- 不接 SNN 时，`Fall recall` 和 `ADL specificity` 分别是什么水平
- 接了 SNN 但不加 gate 时，是否已经有提升
- gate 采用 `scalar` 时提升多少
- gate 采用 `group_channel` 时是否确实更优

后续论文图可以据此画：

- `ADL specificity` 对比图
- `Fall recall` 对比图
- `ADL specificity - Fall recall` trade-off 散点图

## 严格数据口径

- 训练：`GMDCSA Subject1-2 + Zenodo train`
- 验证与阈值校准：`GMDCSA Subject3 + Zenodo val`
- 最终独立测试：`GMDCSA Subject4`

注意：

- `Subject4` 绝对不能参与消融选择、阈值选择或模型选择
- 这组消融实验的选择标准仍然只能来自 `Subject3 + Zenodo val`

## 当前目录结构

- 代码：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8-0707\code\automatic_fall_detector.py`
- 训练脚本：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8-0707\scripts`
- 配置清单：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8-0707\reports\ablation_manifest.json`

训练产物会分别落到：

- 模型：`models\<ablation_name>\`
- 日志与训练报告：`reports\<ablation_name>\`
- 训练曲线：`image\<ablation_name>\`

## 四个消融版本

### 1. `no_snn`

目标：

- 作为纯 CNN/GRU 对照基线

核心设置：

- 不启用 `--use-snn-temporal-branch`
- 不启用 `--use-snn-fusion-gate`
- `--snn-adl-gate-penalty 0.0`

运行脚本：

- `F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8-0707\scripts\run_no_snn.ps1`

### 2. `snn_no_gate`

目标：

- 检查仅接入 SNN temporal branch，但不加融合 gate 时，是否已经能提升 `Fall recall`

核心设置：

- 启用 `--use-snn-temporal-branch`
- 不启用 `--use-snn-fusion-gate`
- `--snn-adl-gate-penalty 0.0`

运行脚本：

- `F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8-0707\scripts\run_snn_no_gate.ps1`

### 3. `snn_scalar_gate`

目标：

- 检查引入最简单的标量 gate 后，是否能在 `ADL specificity` 与 `Fall recall` 间带来更好折中

核心设置：

- 启用 `--use-snn-temporal-branch`
- 启用 `--use-snn-fusion-gate`
- `--snn-fusion-gate-mode scalar`
- `--snn-adl-gate-penalty 0.06`

运行脚本：

- `F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8-0707\scripts\run_snn_scalar_gate.ps1`

### 4. `snn_group_channel_gate`

目标：

- 检查 `group_channel` 是否仍然是当前最有效的 SNN 融合方式

核心设置：

- 启用 `--use-snn-temporal-branch`
- 启用 `--use-snn-fusion-gate`
- `--snn-fusion-gate-mode group_channel`
- `--snn-adl-gate-penalty 0.06`

运行脚本：

- `F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8-0707\scripts\run_snn_group_channel_gate.ps1`

## 共用训练口径

四个版本默认共用以下设置，保证比较尽量公平：

- 初始化模型：`v7 group_channel gate` checkpoint
- `--external-labeled-stride 16`
- `--image-size 64`
- 开启 `teacher-student noise`
- 开启 `Crossbar + device dynamics`
- `--crossbar-readout-noise-scale 0.25`
- `--hard-negative-normal-weight 1.10`
- `--hard-negative-min-percentile 0.88`
- `--hard-negative-score-gamma 2.0`
- `--adl-context-path-keyword zenodo_falldb_video_split/train/adl`
- `--adl-context-min-percentile 0.95`
- `--adl-context-extra-copies 1`

## 执行说明

如果直接运行某个脚本，例如：

```powershell
F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8-0707\scripts\run_no_snn.ps1
```

则会自动：

- 读取 `code\automatic_fall_detector.py`
- 将该版本模型写入 `models\no_snn\`
- 将训练日志与 json 写入 `reports\no_snn\`
- 将训练曲线写入 `image\no_snn\`

## 当前状态

当前仅完成：

- 新版本目录建立
- 主代码复制
- 四个消融版本运行脚本建立
- 参数清单整理

当前尚未完成：

- 四个版本正式训练
- 验证阈值扫描
- `Subject4` 最终独立测试
- 最终对比作图数据汇总

下一步建议顺序：

1. 先跑 `no_snn`
2. 再跑 `snn_no_gate`
3. 再跑 `snn_scalar_gate`
4. 最后跑 `snn_group_channel_gate`

这样最容易从“有没有 SNN”到“gate 形式是否重要”逐层解释结果。
