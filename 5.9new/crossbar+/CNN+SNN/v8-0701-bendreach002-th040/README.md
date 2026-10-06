# v8-0701-bendreach002-th040：继续提高 bend_reach penalty 启用门槛

## 版本目的

本版本延续 `v8-0701-bendreach002-th035` 的思路，只约束 `bend_reach` 一个 ADL subtype 通道。

本轮只改变 targeted ADL penalty 的启用门槛：

- `bend_reach = 0.02`
- `targeted-adl-penalty-score-threshold = 0.40`

阈值选择规则继续使用用户指定的 Fall recall 下限：

- 只使用 `Subject3 + Zenodo val` 选阈值
- 先筛选验证集 `Fall recall >= 75%`
- 再选择 balanced accuracy 最高的阈值
- 并列时优先 ADL specificity 更高，再优先更高阈值

## 工程位置

- 工程根目录：`F:\12.18-2`
- 当前版本目录：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8-0701-bendreach002-th040`
- 主代码：`code\automatic_fall_detector.py`
- 严格评估脚本：`reports\strict_eval_v8_0701_bendreach002_th040.py`

## 数据与约束

本轮仍保持 route-A / v8-style 可比数据规模：

- train = `2508`
- val = `564`

训练日志已确认：

- `Loaded conductance data`
- `Loaded device dynamics`
- `Targeted ADL fall penalty | hard_only=True score_threshold=0.400 subtypes=['bend_reach=0.020']`

Subject4 只作为最终独立测试，不参与阈值、模型、采样权重或 penalty 参数选择。

## 关键参数

- `targeted-adl-subtypes = bend_reach`
- `targeted-adl-subtype-weights = bend_reach=0.02`
- `targeted-adl-penalty-hard-only = on`
- `targeted-adl-penalty-score-threshold = 0.40`
- `external-labeled-stride = 16`
- `image-size = 64`
- `adl-context-min-percentile = 0.95`
- `use-snn-temporal-branch = on`
- `use-snn-fusion-gate = on`
- `use-device-dynamics = on`
- `teacher-student-noise = on`

## 训练结果

训练早停于 epoch 6。

训练规模：

- Train counts：`normal=1031, fall=1477`
- Val counts：`normal=197, fall=367`

输出文件：

- 模型：`models\automatic_fall_event_detector_cnn_snn_v8_0701_bendreach002_th040.pth`
- 训练日志：`reports\train_v8_0701_bendreach002_th040.stdout.log`
- 训练曲线：`image\training_curve_v8_0701_bendreach002_th040.png`

## 严格验证阈值扫描

阈值扫描文件：

`reports\v8_0701_bendreach002_th040_strict_validation_threshold_scan.csv`

| threshold | Accuracy | Balanced Acc | ADL specificity | Fall recall | Precision | TP/TN/FP/FN |
|---:|---:|---:|---:|---:|---:|---:|
| 0.35 | 68.85% | 68.66% | 56.67% | 80.65% | 65.79% | 25/17/13/6 |
| 0.40 | 68.85% | 68.66% | 56.67% | 80.65% | 65.79% | 25/17/13/6 |
| 0.45 | 68.85% | 68.66% | 56.67% | 80.65% | 65.79% | 25/17/13/6 |
| 0.50 | 67.21% | 67.04% | 56.67% | 77.42% | 64.86% | 24/17/13/7 |
| 0.55 | 68.85% | 68.71% | 60.00% | 77.42% | 66.67% | 24/18/12/7 |
| 0.60 | 72.13% | 72.04% | 66.67% | 77.42% | 70.59% | 24/20/10/7 |
| 0.65 | 72.13% | 72.04% | 66.67% | 77.42% | 70.59% | 24/20/10/7 |
| 0.70 | 67.21% | 67.20% | 66.67% | 67.74% | 67.74% | 21/20/10/10 |

推荐阈值：

- threshold = `0.65`

选择原因：

- `0.65` 满足验证集 Fall recall `77.42% >= 75%`
- `0.60` 和 `0.65` 的 balanced accuracy、ADL specificity 相同
- 按规则选择更高阈值 `0.65`

## Subject4 最终结果

结果文件：

`reports\v8_0701_bendreach002_th040_subject4_final_metrics.json`

Subject4 结果：

- Threshold：`0.65`
- Accuracy：`75.68%`
- Balanced Accuracy：`75.74%`
- ADL specificity：`75.00%`
- Fall recall：`76.47%`
- Precision：`72.22%`
- TP/TN/FP/FN：`13 / 15 / 5 / 4`

误报 ADL：

- `ADL\05.mp4`
- `ADL\06.mp4`
- `ADL\07.mp4`
- `ADL\11.mp4`
- `ADL\12.mp4`

漏检 Fall：

- `Fall\01.mp4`
- `Fall\02.mp4`
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

### v8-0701-bendreach002-th035

- Threshold：`0.50`
- Accuracy：`75.68%`
- Balanced Accuracy：`74.85%`
- ADL specificity：`85.00%`
- Fall recall：`64.71%`
- Precision：`78.57%`
- TP/TN/FP/FN：`11 / 17 / 3 / 6`

### 本轮 th040

- Threshold：`0.65`
- Accuracy：`75.68%`
- Balanced Accuracy：`75.74%`
- ADL specificity：`75.00%`
- Fall recall：`76.47%`
- Precision：`72.22%`
- TP/TN/FP/FN：`13 / 15 / 5 / 4`

## 结论

把 `targeted-adl-penalty-score-threshold` 从 `0.35` 提高到 `0.40` 后，Subject4 Fall recall 从 `64.71%` 恢复到 `76.47%`，达到了用户希望的 75% 左右目标，也与 `v8 final p006` 的 Fall recall 持平。

但代价是 ADL specificity 从 `85.00%` 降到 `75.00%`，误报从 3 个增加到 5 个。因此本轮仍没有超过 `v8 final p006`：

- Fall recall：持平
- ADL specificity：低 5 个百分点
- Balanced Accuracy：低约 2.50 个百分点
- Precision：更低

当前判断：

- `score_threshold=0.40` 是恢复 Fall recall 的有效方向
- 但它放松了 ADL 抑制，ADL 误报增加
- 不能替代 `v8 final p006`
- 如果继续沿这条线，下一步更像是在 `0.35` 和 `0.40` 之间找折中，例如 `0.375`，或改做 Fall event positive mining 来提升 Fall 动态而不是继续放松 ADL penalty

## 留存文件

- 代码：`code\automatic_fall_detector.py`
- 模型：`models\automatic_fall_event_detector_cnn_snn_v8_0701_bendreach002_th040.pth`
- 训练日志：`reports\train_v8_0701_bendreach002_th040.stdout.log`
- 训练曲线：`image\training_curve_v8_0701_bendreach002_th040.png`
- 阈值扫描：`reports\v8_0701_bendreach002_th040_strict_validation_threshold_scan.csv`
- 推荐阈值：`reports\v8_0701_bendreach002_th040_recommended_threshold.json`
- Subject4 结果：`reports\v8_0701_bendreach002_th040_subject4_final_metrics.json`
- Subject4 视频级结果：`reports\v8_0701_bendreach002_th040_subject4_final_video_metrics.csv`
