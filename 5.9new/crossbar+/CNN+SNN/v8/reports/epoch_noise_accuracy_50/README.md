# v8-final epoch 噪声准确率实验（50 epoch）

## 目的

本目录用于生成 `Gaussian` 与 `Poisson` 的 50 epoch 噪声准确率曲线数据，供 Origin 绘制：

- 横轴：`Training epoch`
- 纵轴：`Accuracy (%)`

## 当前运行口径

- Python：`F:\ANACONDA\envs\my_yizu_3.10\python.exe`
- 数据根目录：`F:\12.18-2\data`
- 初始化模型：`F:\12.18-2\5.9new\crossbar+\CNN+SNN\v7\models\automatic_fall_event_detector_cnn_snn_v7_group_channel_gate.pth`
- 训练数据：`GMDCSA Subject1-2 + Zenodo train`
- 验证数据：`GMDCSA Subject3 + Zenodo val`
- 当前命令显式使用：`--external-labeled-stride 16`
- 当前运行窗口规模：`train=3539`，`val=1076`

## 当前输出

- 代码：`code\automatic_fall_detector_epoch_noise.py`
- 启动脚本：`launch_epoch_noise_50.ps1`
- 训练日志：`logs\train_epoch_noise_50.stdout.log`
- 错误日志：`logs\train_epoch_noise_50.stderr.log`
- 曲线数据：`shuju\`
- 模型：`models\automatic_fall_event_detector_v8_epoch_noise_50.pth`
- 训练曲线：`image\training_curve_epoch_noise_50.png`

## 当前状态

该任务已切换为后台运行，目的是避免会话中断时前台长任务被杀掉。

如需确认是否继续推进，可优先查看：

- `logs\train_epoch_noise_50.stdout.log`
- `shuju\Gaussian_epoch_accuracy.csv`
- `shuju\Poisson_epoch_accuracy.csv`

当 `Gaussian_epoch_accuracy.csv` 的最后一行 epoch 到达 `50` 时，说明本轮数据已跑完。
