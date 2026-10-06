# v4：CNN + SNN 动态分支版本

## 版本目的
在原有 Crossbar / CNN / GRU 基础上，引入 SNN 时间分支，并进一步加强对跌倒瞬间快速状态变化的刻画，重点改善：

- 真正跌倒瞬间的快速识别
- ADL 与躺下 / 坐下 / 蹲下等动作的区分
- 器件噪声、阈值和时序特征对泛化能力的影响

## 运行环境

- Python：`F:\ANACONDA\envs\my_yizu_3.10\python.exe`
- 项目根目录：`F:\12.18-2`

## 数据划分

- 训练：GMDCSA Subject 1-2 + Zenodo train
- 验证：GMDCSA Subject 3 + Zenodo val
- 最终测试：GMDCSA Subject 4

严格要求：

- Subject 4 只用于最终一次测试
- 阈值只在 `Subject3 + Zenodo val` 上扫描
- 不允许用测试集反向调参

## 代码核心改动

1. 新增 `--use-snn-temporal-branch`
2. 新增 `--learnable-snn-dynamics`
3. SNN 分支改为可学习的 per-channel decay / threshold
4. 增强事件时序特征：
   - `center_drop_accel`
   - `relative_low_position`
   - `post_peak_hold`
   - `fall_event_like`
5. 冻结策略：
   - 前 `freeze_cnn_gru_epochs=2` 轮冻结 CNN / GRU
   - 只训练 SNN 分支和融合头

## 严格测试结果

阈值来自 `Subject3 + Zenodo val`，推荐阈值：

- `0.55`

Subject4 最终一次测试：

- Accuracy: `64.86%`
- Balanced Accuracy: `63.97%`
- ADL Specificity: `75.00%`
- Fall Recall: `52.94%`
- Precision: `64.29%`

## 结果文件

- 训练模型：`models\automatic_fall_event_detector_cnn_snn_v4_learnable_snn.pth`
- 训练曲线：`image\training_curve_cnn_snn_v4_learnable_snn.png`
- 阈值扫描：`reports\v4_strict_validation_threshold_scan.csv`
- 推荐阈值：`reports\v4_recommended_threshold.json`
- Subject4 最终结果：`reports\v4_subject4_final_metrics.json`

## 当前判断

本版已经完成严格验证流程，没有用 Subject4 参与阈值选择或调参。当前结果说明：

- SNN 动态分支是有效方向，但仍不足以单独突破域差异
- 后续应继续关注真实 Fall 瞬间、hard negative 选择、以及验证集策略

