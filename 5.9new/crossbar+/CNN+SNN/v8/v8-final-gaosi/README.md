# v8 final 图像噪声鲁棒性数据

本目录保存 `v8 final p006` 固定模型在 Subject4 上的测试阶段图像噪声扫描数据。

- 模型：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8\models\automatic_fall_event_detector_cnn_snn_v8_final_p006_penalty006.pth`
- 阈值：`0.7`
- 数据：GMDCSA Subject4 独立最终测试集
- 说明：这里不是重新训练曲线，横轴应使用 `noise_percent`；如果要画严格的 `Training epoch - Accuracy`，需要重新训练并保存每个 epoch 的模型。
- 噪声类型：`gaosi`
- 主数据：`shuju.csv`
- 视频级数据：`video_shuju.csv`
- 基线第一行 Accuracy：`78.38%`

Origin 建议：

1. 导入 `shuju.csv`。
2. X 轴选择 `noise_percent`。
3. Y 轴选择 `accuracy_percent`。
4. 如果需要补充曲线，可同时画 `balanced_accuracy_percent`、`adl_specificity_percent`、`fall_recall_percent`。
