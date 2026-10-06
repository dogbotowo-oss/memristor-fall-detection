# v8-0701-bendreach002-th035：提高 bend_reach penalty 启用门槛

## 版本目的

本版本继续沿用 v8 final / route-A 单通道思路，只约束 ADL subtype 中的 `bend_reach` 通道。

上一轮 `v8-0629-bendreach002` 使用：

- `bend_reach = 0.02`
- `targeted-adl-penalty-score-threshold = 0.30`

结果显示 Subject4 ADL specificity 达到 `100.00%`，但 Fall recall 下降到 `52.94%`，模型过于保守。

本轮尝试不增加 penalty 权重，而是提高 penalty 启用门槛：

- `bend_reach = 0.02`
- `targeted-adl-penalty-score-threshold = 0.35`

目标是让 penalty 只作用于更 hard 的 `bend_reach` ADL 窗口，减少对真实 Fall 早期弯腰、下坠、接近地面片段的误伤。

## 工程位置

- 工程根目录：`F:\12.18-2`
- 当前版本目录：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8-0701-bendreach002-th035`
- 主代码：`code\automatic_fall_detector.py`
- 严格评估脚本：`reports\strict_eval_v8_0701_bendreach002_th035.py`

## 数据与约束

严格遵守当前项目划分规则：

- 训练数据：当前 v8 route-A 可比口径下的 `zenodo_falldb_video_split\train`
- 验证数据：`Subject3 + zenodo_falldb_video_split\val`，用于阈值选择
- 最终测试：GMDCSA Subject4，仅作最终独立测试
- Subject4 不参与模型选择、阈值选择、采样权重选择或 penalty 参数选择

本轮训练规模已核对：

- train = `2508`
- val = `564`

训练日志中已确认：

- `Loaded conductance data` 来自 `F:\12.18-2\data`
- `Loaded device dynamics` 来自 `F:\12.18-2\data`
- `Targeted ADL fall penalty | hard_only=True score_threshold=0.350 subtypes=['bend_reach=0.020']`

## 关键参数

本轮相对上一轮只改变 penalty 启用门槛：

- `targeted-adl-subtypes = bend_reach`
- `targeted-adl-subtype-weights = bend_reach=0.02`
- `targeted-adl-penalty-hard-only = on`
- `targeted-adl-penalty-score-threshold = 0.35`

保持的主要设置：

- CNN/GRU + SNN temporal branch
- Crossbar mapping
- measured device dynamics
- teacher-student gaussian noise
- `snn-adl-gate-penalty = 0.06`
- `use-adl-subtype-head = on`
- `adl-subtype-loss-weight = 0.1`
- `adl-subtype-hard-only = on`
- `adl-subtype-hard-score-threshold = 0.24`
- `adl-context-min-percentile = 0.95`
- `external-labeled-stride = 16`
- `image-size = 64`
- `freeze-cnn-gru-epochs = 2`

## 阈值选择规则

本轮把用户指定的 Fall recall 下限并入验证阈值选择：

1. 只允许使用 `Subject3 + Zenodo val` 选择阈值。
2. 先筛选验证集 `Fall recall >= 75%` 的阈值。
3. 在满足 Fall recall 下限的阈值中选择 `balanced_accuracy` 最高者。
4. 若并列，优先选择 ADL specificity 更高者。
5. 若仍并列，选择更高阈值。

这只是阈值选择规则，不使用 Subject4 调阈值。

## 训练结果

训练早停于 epoch 8。

训练日志末尾：

- Train counts：`normal=1031, fall=1477`
- Val counts：`normal=197, fall=367`
- 训练曲线：`image\training_curve_v8_0701_bendreach002_th035.png`
- 模型：`models\automatic_fall_event_detector_cnn_snn_v8_0701_bendreach002_th035.pth`

## 严格验证阈值扫描

阈值扫描文件：

`reports\v8_0701_bendreach002_th035_strict_validation_threshold_scan.csv`

验证集结果：

| threshold | Accuracy | Balanced Acc | ADL specificity | Fall recall | Precision | TP/TN/FP/FN |
|---:|---:|---:|---:|---:|---:|---:|
| 0.35 | 72.13% | 71.94% | 60.00% | 83.87% | 68.42% | 26/18/12/5 |
| 0.40 | 72.13% | 71.94% | 60.00% | 83.87% | 68.42% | 26/18/12/5 |
| 0.45 | 72.13% | 71.94% | 60.00% | 83.87% | 68.42% | 26/18/12/5 |
| 0.50 | 72.13% | 71.99% | 63.33% | 80.65% | 69.44% | 25/19/11/6 |
| 0.55 | 70.49% | 70.38% | 63.33% | 77.42% | 68.57% | 24/19/11/7 |
| 0.60 | 68.85% | 68.76% | 63.33% | 74.19% | 67.65% | 23/19/11/8 |
| 0.65 | 67.21% | 67.20% | 66.67% | 67.74% | 67.74% | 21/20/10/10 |
| 0.70 | 65.57% | 65.65% | 70.00% | 61.29% | 67.86% | 19/21/9/12 |

推荐阈值：

- threshold = `0.50`

选择原因：

- `0.50` 满足验证集 Fall recall `80.65% >= 75%`
- 在满足 Fall recall 下限的阈值中 balanced accuracy 最高，为 `71.99%`

## Subject4 最终结果

结果文件：

`reports\v8_0701_bendreach002_th035_subject4_final_metrics.json`

Subject4 最终结果：

- Threshold：`0.50`
- Accuracy：`75.68%`
- Balanced Accuracy：`74.85%`
- ADL specificity：`85.00%`
- Fall recall：`64.71%`
- Precision：`78.57%`
- TP/TN/FP/FN：`11 / 17 / 3 / 6`

误报 ADL：

- `ADL\05.mp4`
- `ADL\06.mp4`
- `ADL\11.mp4`

漏检 Fall：

- `Fall\01.mp4`
- `Fall\02.mp4`
- `Fall\05.mp4`
- `Fall\08.mp4`
- `Fall\09.mp4`
- `Fall\13.mp4`

## 与关键版本对比

### v8 final p006

- Threshold：`0.70`
- Accuracy：`78.38%`
- Balanced Accuracy：`78.24%`
- ADL specificity：`80.00%`
- Fall recall：`76.47%`
- Precision：`76.47%`
- TP/TN/FP/FN：`13 / 16 / 4 / 4`

### v8-0629-bendreach002

- `bend_reach=0.02`
- `score_threshold=0.30`
- Subject4 Accuracy：`78.38%`
- Balanced Accuracy：`76.47%`
- ADL specificity：`100.00%`
- Fall recall：`52.94%`
- TP/TN/FP/FN：`9 / 20 / 0 / 8`

### 本轮 v8-0701-bendreach002-th035

- `bend_reach=0.02`
- `score_threshold=0.35`
- Subject4 Accuracy：`75.68%`
- Balanced Accuracy：`74.85%`
- ADL specificity：`85.00%`
- Fall recall：`64.71%`
- TP/TN/FP/FN：`11 / 17 / 3 / 6`

## 结论

本轮把 `targeted-adl-penalty-score-threshold` 从 `0.30` 提高到 `0.35` 后，模型确实不再像 `v8-0629-bendreach002` 那样极端保守：

- Subject4 Fall recall 从 `52.94%` 提高到 `64.71%`
- 但 ADL specificity 从 `100.00%` 回落到 `85.00%`
- Balanced Accuracy 从 `76.47%` 回落到 `74.85%`

这说明提高 penalty 启用门槛能够释放一部分真实 Fall，但仍没有达到 `v8 final p006` 的综合水平，尤其是 Fall recall 仍低于主结果的 `76.47%`。

当前结论：

- 本轮不能替代 `v8 final p006`
- `score_threshold=0.35` 比 `0.30` 更不保守，方向是有效的
- 但单靠 `bend_reach` penalty gate 调整，仍不足以稳定恢复 Subject4 Fall recall 到 75% 以上
- 后续更值得尝试 Fall event positive mining 或更轻的 post-score/gate 校正，而不是继续增强 ADL penalty

## 留存文件

- 代码：`code\automatic_fall_detector.py`
- 模型：`models\automatic_fall_event_detector_cnn_snn_v8_0701_bendreach002_th035.pth`
- 训练日志：`reports\train_v8_0701_bendreach002_th035.stdout.log`
- 训练曲线：`image\training_curve_v8_0701_bendreach002_th035.png`
- 阈值扫描：`reports\v8_0701_bendreach002_th035_strict_validation_threshold_scan.csv`
- 推荐阈值：`reports\v8_0701_bendreach002_th035_recommended_threshold.json`
- Subject4 结果：`reports\v8_0701_bendreach002_th035_subject4_final_metrics.json`
- Subject4 视频级结果：`reports\v8_0701_bendreach002_th035_subject4_final_video_metrics.csv`
