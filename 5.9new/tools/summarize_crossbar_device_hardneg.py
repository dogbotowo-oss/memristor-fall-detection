from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(r"F:\12.18-2")
PLUS = ROOT / "5.9new" / "crossbar+"
EXP_ROOT = PLUS / "teacher_crossbar_device_hardneg"
SHARED_IMAGE = ROOT / "5.9new" / "image"


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def add_precision(row: dict[str, object], adl_total: int = 20) -> dict[str, object]:
    adl_correct = int(row["adl_correct"])
    fall_correct = int(row["fall_correct"])
    false_alarms = adl_total - adl_correct
    predicted_falls = fall_correct + false_alarms
    row["false_alarms"] = false_alarms
    row["precision"] = fall_correct / predicted_falls if predicted_falls else 0.0
    row["balanced_accuracy"] = (float(row["adl_specificity"]) + float(row["fall_recall"])) / 2.0
    return row


def load_gaussian() -> dict[str, object]:
    payload = json.loads((PLUS / "teacher_gaussian_noise" / "reports" / "recommended_threshold.json").read_text(encoding="utf-8"))
    threshold = payload["recommended_threshold"]
    result = payload["subject4_calibrated_result"]
    return add_precision(
        {
            "group": "Gaussian best",
            "threshold": f"{float(threshold['fall_prob_threshold']):.2f}, {float(threshold['min_fall_duration_sec']):.2f}s",
            "accuracy": float(result["video_accuracy"]),
            "adl_specificity": float(result["adl_specificity"]),
            "fall_recall": float(result["fall_recall"]),
            "adl_correct": int(result["adl_correct"]),
            "fall_correct": int(result["fall_correct"]),
            "note": "Previous best",
        }
    )


def load_scan_point(label: str, root: Path, prob: float, margin: float, duration: float, note: str) -> dict[str, object]:
    scan_path = root / "reports" / "threshold_scan" / "threshold_scan_summary.csv"
    for item in csv.DictReader(scan_path.open(encoding="utf-8")):
        if (
            abs(float(item["fall_prob_threshold"]) - prob) < 1e-8
            and abs(float(item["fall_margin_threshold"]) - margin) < 1e-8
            and abs(float(item["min_fall_duration_sec"]) - duration) < 1e-8
        ):
            return add_precision(
                {
                    "group": label,
                    "threshold": f"{prob:.2f}, {duration:.2f}s",
                    "accuracy": float(item["video_accuracy"]),
                    "adl_specificity": float(item["adl_specificity"]),
                    "fall_recall": float(item["fall_recall"]),
                    "adl_correct": int(item["adl_correct"]),
                    "fall_correct": int(item["fall_correct"]),
                    "note": note,
                }
            )
    raise ValueError(f"Missing scan point: {scan_path} {prob} {margin} {duration}")


def load_new_recommended() -> dict[str, object]:
    payload = json.loads((EXP_ROOT / "reports" / "recommended_threshold.json").read_text(encoding="utf-8"))
    threshold = payload["recommended_threshold"]
    result = payload["subject4_result"]
    return add_precision(
        {
            "group": "Crossbar device + hard neg",
            "threshold": f"{float(threshold['fall_prob_threshold']):.2f}, {float(threshold['min_fall_duration_sec']):.2f}s",
            "accuracy": float(result["video_accuracy"]),
            "adl_specificity": float(result["adl_specificity"]),
            "fall_recall": float(result["fall_recall"]),
            "adl_correct": int(result["adl_correct"]),
            "fall_correct": int(result["fall_correct"]),
            "note": "B + D optimized",
        }
    )


def build_rows() -> list[dict[str, object]]:
    return [
        load_gaussian(),
        load_scan_point("Direct hybrid HR", PLUS / "teacher_hybrid_noise", 0.50, 0.08, 1.00, "Input noise high recall"),
        load_scan_point("Variance hybrid HR", PLUS / "teacher_variance_hybrid_noise", 0.50, 0.08, 1.00, "Variance-preserving"),
        load_new_recommended(),
    ]


def write_reports(rows: list[dict[str, object]]) -> None:
    reports = EXP_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    csv_path = reports / "crossbar_device_hardneg_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (reports / "crossbar_device_hardneg_comparison.json").write_text(
        json.dumps({"rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def draw_figure(rows: list[dict[str, object]]) -> None:
    image_dir = EXP_ROOT / "image"
    image_dir.mkdir(parents=True, exist_ok=True)
    SHARED_IMAGE.mkdir(parents=True, exist_ok=True)

    labels = [str(row["group"]) for row in rows]
    x = list(range(len(rows)))
    width = 0.18
    acc = [float(row["accuracy"]) * 100 for row in rows]
    adl = [float(row["adl_specificity"]) * 100 for row in rows]
    fall = [float(row["fall_recall"]) * 100 for row in rows]
    precision = [float(row["precision"]) * 100 for row in rows]
    balanced = [float(row["balanced_accuracy"]) * 100 for row in rows]

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(15.5, 8.8), dpi=180)
    grid = fig.add_gridspec(2, 1, height_ratios=[1.05, 0.95], hspace=0.22)
    ax = fig.add_subplot(grid[0])
    ax.bar([i - 1.5 * width for i in x], acc, width=width, color="#386cb0", label="Accuracy")
    ax.bar([i - 0.5 * width for i in x], adl, width=width, color="#35a16b", label="ADL specificity")
    ax.bar([i + 0.5 * width for i in x], fall, width=width, color="#d95f5f", label="Fall recall")
    ax.bar([i + 1.5 * width for i in x], precision, width=width, color="#8c6bb1", label="Precision")
    ax.plot(x, balanced, color="#222222", linewidth=2.2, marker="o", label="Balanced accuracy")
    for i, value in enumerate(balanced):
        ax.text(i, min(value + 3, 102), f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Crossbar Device Noise + ADL Hard Negative Optimization", fontsize=21, fontweight="bold", loc="left")
    ax.set_ylabel("Video-level metric (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", frameon=False, ncols=5)

    ax_table = fig.add_subplot(grid[1])
    ax_table.axis("off")
    columns = ["Group", "Threshold", "Acc", "ADL", "Fall", "Precision", "Bal Acc", "Note"]
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row["group"],
                row["threshold"],
                pct(float(row["accuracy"])),
                f"{pct(float(row['adl_specificity']))}\n({row['adl_correct']}/20)",
                f"{pct(float(row['fall_recall']))}\n({row['fall_correct']}/17)",
                f"{pct(float(row['precision']))}\nFP={row['false_alarms']}",
                pct(float(row["balanced_accuracy"])),
                row["note"],
            ]
        )
    table = ax_table.table(
        cellText=table_rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.2)
    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_edgecolor("#d8dee8")
        if row_index == 0:
            cell.set_facecolor("#1f2937")
            cell.set_text_props(color="white", fontweight="bold")
        elif col_index == 0:
            cell.set_facecolor("#eef4fb")
            cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("#ffffff" if row_index % 2 else "#f8fafc")
    for col_index in range(len(columns)):
        table[len(rows), col_index].set_facecolor("#e8f6ee")

    output = image_dir / "crossbar_device_hardneg_comparison_ppt.png"
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    (SHARED_IMAGE / output.name).write_bytes(output.read_bytes())


def main() -> int:
    rows = build_rows()
    write_reports(rows)
    draw_figure(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
