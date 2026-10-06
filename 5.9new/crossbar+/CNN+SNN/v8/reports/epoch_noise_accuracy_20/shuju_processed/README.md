# 噪声鲁棒性数据处理说明

本目录保存 `epoch_noise_accuracy_20/shuju` 的后处理结果，原始 CSV 未被修改。

## 文件说明

- `*_smoothed3.csv`：3 点滑动平均，适合画更平滑的 epoch-accuracy 曲线。
- `*_smoothed5.csv`：5 点滑动平均，趋势更平缓，但会削弱局部起伏。
- `*_noise_level_accuracy_summary.csv`：把每个噪声强度在 20 个 epoch 上做统计汇总。
- `combined_noise_level_accuracy_summary.csv`：三类噪声放在一起，便于直接做“噪声强度-准确率”关系图。
- `*_smoothed3_recommended.csv`：保留推荐画图的 5 条曲线（0%、25%、50%、75%、87.5%）。

## 推荐画图方式

如果目标是证明“识别准确率与图像模糊程度匹配”，建议优先画两类图：

1. `Gaussian` 和 `Poisson` 的 `*_smoothed3_recommended.csv`
   - 横轴：Epoch
   - 纵轴：Accuracy
   - 用于展示随训练推进时，不同噪声强度的分层趋势

2. `combined_noise_level_accuracy_summary.csv`
   - 横轴：NoisePercent
   - 纵轴建议优先用 `MeanAccuracy_AllEpochs` 或 `MeanAccuracy_Last5Epochs`
   - 用于展示噪声越强，平均识别准确率越低，更适合支撑论文中的真实性论证

## 结果解释建议

- `Gaussian`、`Poisson` 更适合作为主图，因为梯度更连续。
- `SaltPepper` 在当前强度下退化过快，适合作为补充图，不建议放在主图中心位置。
- 如果想让图更“论文化”，建议作图时优先使用平滑后的 3 点滑动平均数据，不要改原始实验值。
