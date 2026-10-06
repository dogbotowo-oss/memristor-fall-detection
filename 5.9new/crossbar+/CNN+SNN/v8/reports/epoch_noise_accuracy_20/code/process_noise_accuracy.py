from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean


BASE_DIR = Path(
    r"F:\12.18-2\5.9new\crossbar+\CNN+SNN\v8\reports\epoch_noise_accuracy_20"
)
INPUT_DIR = BASE_DIR / "shuju"
OUTPUT_DIR = BASE_DIR / "shuju_processed"

INPUT_FILES = {
    "Gaussian": INPUT_DIR / "Gaussian_epoch_accuracy.csv",
    "Poisson": INPUT_DIR / "Poisson_epoch_accuracy.csv",
    "SaltPepper": INPUT_DIR / "SaltPepper_epoch_accuracy.csv",
}


def read_accuracy_csv(path: Path) -> tuple[list[int], list[str], list[list[float]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        noise_columns = [name for name in fieldnames if name != "Epoch"]
        epochs: list[int] = []
        data_by_row: list[list[float]] = []
        for row in reader:
            epochs.append(int(row["Epoch"]))
            data_by_row.append([float(row[col]) for col in noise_columns])
    return epochs, noise_columns, data_by_row


def write_table(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def moving_average(values: list[float], window: int) -> list[float]:
    radius = window // 2
    smoothed: list[float] = []
    for i in range(len(values)):
        left = max(0, i - radius)
        right = min(len(values), i + radius + 1)
        smoothed.append(round(mean(values[left:right]), 4))
    return smoothed


def transpose_rows_to_columns(rows: list[list[float]]) -> list[list[float]]:
    if not rows:
        return []
    return [list(col) for col in zip(*rows)]


def build_smoothed_rows(
    epochs: list[int], noise_columns: list[str], data_rows: list[list[float]], window: int
) -> list[list[object]]:
    columns = transpose_rows_to_columns(data_rows)
    smoothed_columns = [moving_average(col, window) for col in columns]
    output_rows: list[list[object]] = []
    for row_idx, epoch in enumerate(epochs):
        row: list[object] = [epoch]
        for col in smoothed_columns:
            row.append(f"{col[row_idx]:.4f}")
        output_rows.append(row)
    return output_rows


def summarize_noise_levels(
    noise_columns: list[str], data_rows: list[list[float]], smoothed3_rows: list[list[object]]
) -> list[list[object]]:
    columns = transpose_rows_to_columns(data_rows)
    smoothed3_columns = transpose_rows_to_columns(
        [[float(v) for v in row[1:]] for row in smoothed3_rows]
    )
    summary_rows: list[list[object]] = []
    for idx, noise_label in enumerate(noise_columns):
        values = columns[idx]
        first_half = values[:10]
        second_half = values[10:]
        last_five = values[-5:]
        smoothed_last = smoothed3_columns[idx][-1]
        noise_percent = noise_label.replace("Noise_", "").replace("%_Test", "")
        summary_rows.append(
            [
                noise_label,
                noise_percent,
                f"{mean(values):.4f}",
                f"{mean(first_half):.4f}",
                f"{mean(second_half):.4f}",
                f"{mean(last_five):.4f}",
                f"{smoothed_last:.4f}",
                f"{max(values):.4f}",
                int(values.index(max(values)) + 1),
            ]
        )
    return summary_rows


def build_recommended_subset(
    epochs: list[int], smoothed3_rows: list[list[object]]
) -> tuple[list[str], list[list[object]]]:
    keep_columns = [0, 1, 3, 5, 7, 8]
    header = [
        "Epoch",
        "Noise_0%_Test",
        "Noise_25%_Test",
        "Noise_50%_Test",
        "Noise_75%_Test",
        "Noise_87.5%_Test",
    ]
    rows: list[list[object]] = []
    for row in smoothed3_rows:
        rows.append([row[idx] for idx in keep_columns])
    return header, rows


def write_readme(output_dir: Path) -> None:
    readme = """# 噪声鲁棒性数据处理说明

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
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined_summary_rows: list[list[object]] = []

    for noise_type, csv_path in INPUT_FILES.items():
        epochs, noise_columns, data_rows = read_accuracy_csv(csv_path)

        smoothed3_rows = build_smoothed_rows(epochs, noise_columns, data_rows, window=3)
        smoothed5_rows = build_smoothed_rows(epochs, noise_columns, data_rows, window=5)
        summary_rows = summarize_noise_levels(noise_columns, data_rows, smoothed3_rows)
        rec_header, rec_rows = build_recommended_subset(epochs, smoothed3_rows)

        write_table(
            OUTPUT_DIR / f"{noise_type}_epoch_accuracy_smoothed3.csv",
            ["Epoch", *noise_columns],
            smoothed3_rows,
        )
        write_table(
            OUTPUT_DIR / f"{noise_type}_epoch_accuracy_smoothed5.csv",
            ["Epoch", *noise_columns],
            smoothed5_rows,
        )
        write_table(
            OUTPUT_DIR / f"{noise_type}_noise_level_accuracy_summary.csv",
            [
                "NoiseLabel",
                "NoisePercent",
                "MeanAccuracy_AllEpochs",
                "MeanAccuracy_Epoch1to10",
                "MeanAccuracy_Epoch11to20",
                "MeanAccuracy_Last5Epochs",
                "Smoothed3_LastEpoch",
                "PeakAccuracy",
                "PeakEpoch",
            ],
            summary_rows,
        )
        write_table(
            OUTPUT_DIR / f"{noise_type}_epoch_accuracy_smoothed3_recommended.csv",
            rec_header,
            rec_rows,
        )

        for row in summary_rows:
            combined_summary_rows.append([noise_type, *row])

    write_table(
        OUTPUT_DIR / "combined_noise_level_accuracy_summary.csv",
        [
            "NoiseType",
            "NoiseLabel",
            "NoisePercent",
            "MeanAccuracy_AllEpochs",
            "MeanAccuracy_Epoch1to10",
            "MeanAccuracy_Epoch11to20",
            "MeanAccuracy_Last5Epochs",
            "Smoothed3_LastEpoch",
            "PeakAccuracy",
            "PeakEpoch",
        ],
        combined_summary_rows,
    )
    write_readme(OUTPUT_DIR)
    print(f"Processed files written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
