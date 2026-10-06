# epoch_noise_fallrecall_50_repeat5_old8levels

本目录是在 `epoch_noise_fallrecall_50_repeat5_4lines` 基础上扩展出的旧档梯度版本。

## 目的

- 保持 `50 epoch` 训练口径不变。
- 继续使用 `repeat=5` 的噪声重复评估，尽量减小单次随机加噪带来的跳动。
- 将噪声梯度恢复为之前更细的 `8` 档，减轻 `0 / 33.3 / 66.7 / 100` 这种大跨度带来的剧烈抖动。

## 本版噪声梯度

- `0%`
- `12.5%`
- `25%`
- `37.5%`
- `50%`
- `62.5%`
- `75%`
- `87.5%`

## 输出说明

`shuju` 目录下会生成：

- `Gaussian_epoch_fall_recall.csv`
- `Poisson_epoch_fall_recall.csv`
- `SaltPepper_epoch_fall_recall.csv`

以上 3 张表对应三种噪声，横坐标都是 `Epoch 1-50`，纵坐标是 `5 次重复评估后的 mean Fall Recall`。

同时会生成对应的标准差表：

- `Gaussian_epoch_fall_recall_std.csv`
- `Poisson_epoch_fall_recall_std.csv`
- `SaltPepper_epoch_fall_recall_std.csv`

以及长表：

- `All_noise_epoch_metrics_long.csv`

## 关键参数

- `--epoch-noise-levels 0,0.125,0.25,0.375,0.5,0.625,0.75,0.875`
- `--epoch-noise-eval-repeats 5`

## 说明

- 数据划分仍然严格保持：`Subject1-2 + Zenodo train` 训练，`Subject3 + Zenodo val` 验证。
- 该目录仅用于 epoch 噪声曲线绘图支撑，不改变 `v8 final p006` 的主结果结论。
- 原始 `epoch_noise_fallrecall_50_repeat5_4lines` 目录不覆盖、不替换。
