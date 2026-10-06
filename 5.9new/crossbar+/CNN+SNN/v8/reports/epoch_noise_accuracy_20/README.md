# v8-final epoch 噪声准确率 CSV 实验（20 epoch）

## 实验目的

本实验用于生成 20 epoch 版本、类似论文图 8(E) 的数据表：横坐标为 Training epoch，纵坐标为 Accuracy (%)，每一种噪声单独一张图，同一张图中用多条曲线表示不同噪声强度。

## 数据与约束

- 基于 `v8 final` p006 训练口径。
- 训练集使用 Zenodo/GMDCSA 训练侧数据；验证曲线使用训练过程中的 external validation split。
- 不使用 Subject4 进行阈值、参数或曲线选择。
- 噪声为运行时加入到输入张量中，不覆盖、不另存、不污染原始数据集。

## 噪声强度

八档强度为：`0%, 12.5%, 25%, 37.5%, 50%, 62.5%, 75%, 87.5%`。

- Gaussian：`sigma = 0.30 * level`。
- Salt-and-pepper：`level` 表示像素被置为 0 或 1 的概率。
- Poisson：用光子峰值表示噪声强弱，峰值越低噪声越强；从 `220` 几何下降到 `6`。

## 输出文件

CSV 输出目录：`shuju/`

- `Gaussian_epoch_accuracy.csv`
- `Poisson_epoch_accuracy.csv`
- `SaltPepper_epoch_accuracy.csv`
- `All_noise_epoch_accuracy_long.csv`

宽表第一列为 `Epoch`，后续列为不同噪声强度，数值单位为 Accuracy (%)，可直接导入 Origin 画多曲线图。


Note: this folder is the 20-epoch extension and does not overwrite the 6-epoch folder.
