# sit_descent001：在 th040 基础上增加 sit_descent 轻量通道

## 版本目的

本子实验放在当前版本目录下：

`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8-0701-bendreach002-th040\sit_descent001`

目的：不再改变 `score_threshold`，在 th040 单通道基础上增加一个很轻的 `sit_descent` 通道，看是否能压低 Subject4 ADL 误报，同时尽量保持 Fall recall。

## 参数设置

保持 th040 的主要设置不变：

- `targeted-adl-penalty-score-threshold = 0.40`
- 验证阈值规则：验证集 `Fall recall >= 75%`，再选 balanced accuracy 最优
- 训练规模保持 route-A / v8-style 可比口径：`train=2508`, `val=564`
- Crossbar / device dynamics / teacher-student noise / SNN temporal branch 均保留

本轮新增通道：

- `bend_reach = 0.02`
- `sit_descent = 0.01`

训练日志确认：

`Targeted ADL fall penalty | hard_only=True score_threshold=0.400 subtypes=['sit_descent=0.010', 'bend_reach=0.020']`

## 训练核对

训练日志确认：

- external_labeled_windows = `2508`
- external_val_labeled_windows = `564`
- Train counts：`normal=1031, fall=1477`
- Val counts：`normal=197, fall=367`
- `Loaded conductance data`
- `Loaded device dynamics`

训练早停于 epoch 6。

## 严格验证阈值扫描

阈值扫描文件：

`reports\v8_0701_bendreach002_sit001_th040_strict_validation_threshold_scan.csv`

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

## Subject4 最终结果

结果文件：

`reports\v8_0701_bendreach002_sit001_th040_subject4_final_metrics.json`

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

## 与 th040 单通道对比

th040 单通道结果：

- `bend_reach=0.02`
- `score_threshold=0.40`
- Threshold：`0.65`
- Accuracy：`75.68%`
- Balanced Accuracy：`75.74%`
- ADL specificity：`75.00%`
- Fall recall：`76.47%`
- TP/TN/FP/FN：`13 / 15 / 5 / 4`

本轮双通道结果：

- `bend_reach=0.02`
- `sit_descent=0.01`
- `score_threshold=0.40`
- Threshold：`0.65`
- Accuracy：`75.68%`
- Balanced Accuracy：`75.74%`
- ADL specificity：`75.00%`
- Fall recall：`76.47%`
- TP/TN/FP/FN：`13 / 15 / 5 / 4`

## 结论

增加很轻的 `sit_descent=0.01` 后，验证扫描和 Subject4 最终结果与 th040 单通道完全一致。

当前判断：

- `sit_descent=0.01` 没有伤害 Fall recall
- 但也没有提升 ADL specificity
- 说明该权重太轻，或者当前误报 ADL 并不主要受 `sit_descent` 轻量 penalty 控制
- 本轮不能替代 `v8 final p006`
- 本轮也没有超过 th040 单通道

如果继续沿“增加通道”方向，下一步可以考虑：

- `sit_descent=0.02`，但存在压低 Fall recall 的风险
- 或转向 `post-score/gate` 校正，而不是继续增强训练 penalty
- 更推荐后续尝试 Fall event positive mining，提高真实 Fall 动态响应

## 留存文件

- 代码：`code\automatic_fall_detector.py`
- 模型：`models\automatic_fall_event_detector_cnn_snn_v8_0701_bendreach002_sit001_th040.pth`
- 训练日志：`reports\train_v8_0701_bendreach002_sit001_th040.stdout.log`
- 训练曲线：`image\training_curve_v8_0701_bendreach002_sit001_th040.png`
- 阈值扫描：`reports\v8_0701_bendreach002_sit001_th040_strict_validation_threshold_scan.csv`
- 推荐阈值：`reports\v8_0701_bendreach002_sit001_th040_recommended_threshold.json`
- Subject4 结果：`reports\v8_0701_bendreach002_sit001_th040_subject4_final_metrics.json`
- Subject4 视频级结果：`reports\v8_0701_bendreach002_sit001_th040_subject4_final_video_metrics.csv`

## 追加尝试：sit_descent=0.02

用户后续要求在当前 `sit_descent001` 文件夹内直接尝试：

- `bend_reach = 0.02`
- `sit_descent = 0.02`
- `targeted-adl-penalty-score-threshold = 0.40`

本轮没有新建工程目录。训练尝试未形成有效模型和 Subject4 最终结果，不能把本目录中原有 `sit001` 的 JSON 指标当作 `sit_descent=0.02` 的结果。

已尝试的运行状态：

- 首次 GPU / batch-size 8：训练过程中 CUDA out of memory。
- 第二次加入 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`：数据加载阶段出现 NumPy 内存分配失败。
- 第三次 CPU / batch-size 8：视频读取阶段出现 OpenCV / 系统内存异常。
- 第四次 GPU / batch-size 4：视频读取阶段仍出现 OpenCV 内存分配失败。

日志中确认参数曾正确进入加载阶段：

`Targeted ADL fall penalty | hard_only=True score_threshold=0.400 subtypes=['sit_descent=0.020', 'bend_reach=0.020']`

因此当前结论仅能写为：

- `sit_descent=0.02` 方案尚未完成训练。
- 当前没有严格验证阈值扫描结果。
- 当前没有 Subject4 final 结果。
- 已有可引用结果仍是 `sit_descent=0.01` 版本：Subject4 `75.68% / 75.74% / 75.00% / 76.47%`。

后续如果要继续跑 `sit_descent=0.02`，建议先释放系统内存和 OpenCV 读帧资源，或改代码为分块/缓存式窗口加载；否则继续重复训练命令大概率仍会在数据加载阶段失败。

## 追加尝试结果：sit_descent=0.02 rerun4

本次按用户要求继续在当前 `sit_descent001` 文件夹内重跑，没有新建工程目录，也没有覆盖原 `sit001` 结果。

本次训练参数：

- `bend_reach = 0.02`
- `sit_descent = 0.02`
- `targeted-adl-penalty-score-threshold = 0.40`
- `batch_size = 4`
- `image_size = 64`
- 输出前缀：`v8_0701_bendreach002_sit002_th040_b4_rerun4`

训练已完成并保存模型：

- `models\automatic_fall_event_detector_cnn_snn_v8_0701_bendreach002_sit002_th040_b4_rerun4.pth`
- `image\training_curve_v8_0701_bendreach002_sit002_th040_b4_rerun4.png`
- `reports\train_v8_0701_bendreach002_sit002_th040_b4_rerun4.combined.log`

训练日志确认数据规模仍为当前 v8 路线口径：

- train windows = `2508`
- val windows = `564`

严格验证结果：

- 验证来源：`Subject3 + Zenodo val`
- 阈值选择规则：先要求验证集 `Fall recall >= 75%`，再选 balanced accuracy 最优
- 本轮没有任何阈值达到 `Fall recall >= 75%`
- 因此没有选择推荐阈值
- 因此没有运行 Subject4 final test

验证扫描中最高 Fall recall 仅为：

- threshold `0.35 / 0.40 / 0.45`
- Fall recall `38.71%`
- ADL specificity `73.33%`
- Balanced Accuracy `56.02%`
- TP/TN/FP/FN = `12 / 22 / 8 / 19`

较高阈值虽然进一步提高 ADL specificity，但 Fall recall 继续下降。例如：

- threshold `0.65`
- ADL specificity `100.00%`
- Fall recall `25.81%`
- TP/TN/FP/FN = `8 / 30 / 0 / 23`

本轮结论：

- `sit_descent=0.02` 明显过强，会把 Fall recall 压得过低。
- 该方案在验证阶段已经不满足用户设定的 `Fall recall >= 75%` 下限。
- 按严格实验规则，本轮不能进入 Subject4 final，也不能作为候选结果。
- 当前可引用的双通道结果仍是 `sit_descent=0.01`，其 Subject4 指标为 Accuracy `75.68%`、Balanced Accuracy `75.74%`、ADL specificity `75.00%`、Fall recall `76.47%`。

本轮留存文件：

- 验证扫描：`reports\v8_0701_bendreach002_sit002_th040_b4_rerun4_strict_validation_threshold_scan.csv`
- 不合格阈值记录：`reports\v8_0701_bendreach002_sit002_th040_b4_rerun4_recommended_threshold.json`
- scan-only 日志：`reports\strict_eval_v8_0701_bendreach002_sit002_th040_b4_rerun4_scan_only.combined.log`
