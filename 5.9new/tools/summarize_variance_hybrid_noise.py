from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(r"F:\12.18-2")
PLUS = ROOT / "5.9new" / "crossbar+"
VAR_ROOT = PLUS / "teacher_variance_hybrid_noise"
OUT_REPORTS = VAR_ROOT / "reports"
OUT_IMAGE = VAR_ROOT / "image"
SHARED_IMAGE = ROOT / "5.9new" / "image"


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def precision(fall_correct: int, adl_correct: int, adl_total: int = 20) -> float:
    false_alarms = adl_total - adl_correct
    predicted_falls = fall_correct + false_alarms
    if predicted_falls <= 0:
        return 0.0
    return fall_correct / predicted_falls


def make_row(
    label: str,
    threshold: str,
    accuracy: float,
    adl_specificity: float,
    fall_recall: float,
    adl_correct: int,
    fall_correct: int,
    note: str,
) -> dict[str, object]:
    return {
        "group": label,
        "threshold": threshold,
        "accuracy": accuracy,
        "adl_specificity": adl_specificity,
        "fall_recall": fall_recall,
        "precision": precision(fall_correct, adl_correct),
        "balanced_accuracy": (adl_specificity + fall_recall) / 2.0,
        "adl_correct": adl_correct,
        "fall_correct": fall_correct,
        "false_alarms": 20 - adl_correct,
        "note": note,
    }


def load_gaussian() -> dict[str, object]:
    path = PLUS / "teacher_gaussian_noise" / "reports" / "recommended_threshold.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    threshold = payload["recommended_threshold"]
    result = payload["subject4_calibrated_result"]
    return make_row(
        "Gaussian",
        f"{float(threshold['fall_prob_threshold']):.2f}, {float(threshold['min_fall_duration_sec']):.2f}s",
        float(result["video_accuracy"]),
        float(result["adl_specificity"]),
        float(result["fall_recall"]),
        int(result["adl_correct"]),
        int(result["fall_correct"]),
        "Best overall",
    )


def load_device_best() -> dict[str, object]:
    path = PLUS / "teacher_device_noise" / "reports" / "device_noise_scale_summary.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    best = max(rows, key=lambda item: float(item["calibrated_balanced_accuracy"]))
    return make_row(
        f"Device {float(best['scale']):.2f}x",
        f"{float(best['fall_prob_threshold']):.2f}, {float(best['min_fall_duration_sec']):.2f}s",
        float(best["calibrated_accuracy"]),
        float(best["calibrated_adl_specificity"]),
        float(best["calibrated_fall_recall"]),
        int(best["calibrated_adl_correct"]),
        int(best["calibrated_fall_correct"]),
        "Device-only best",
    )


def find_scan_row(root: Path, prob: float, margin: float, duration: float) -> dict[str, str]:
    path = root / "reports" / "threshold_scan" / "threshold_scan_summary.csv"
    rows = csv.DictReader(path.open(encoding="utf-8"))
    for row in rows:
        if (
            abs(float(row["fall_prob_threshold"]) - prob) < 1e-8
            and abs(float(row["fall_margin_threshold"]) - margin) < 1e-8
            and abs(float(row["min_fall_duration_sec"]) - duration) < 1e-8
        ):
            return row
    raise ValueError(f"Missing scan row in {path}: {prob}, {margin}, {duration}")


def load_scan(
    label: str,
    root: Path,
    prob: float,
    margin: float,
    duration: float,
    note: str,
) -> dict[str, object]:
    row = find_scan_row(root, prob, margin, duration)
    return make_row(
        label,
        f"{prob:.2f}, {duration:.2f}s",
        float(row["video_accuracy"]),
        float(row["adl_specificity"]),
        float(row["fall_recall"]),
        int(row["adl_correct"]),
        int(row["fall_correct"]),
        note,
    )


def build_rows() -> list[dict[str, object]]:
    return [
        load_gaussian(),
        load_device_best(),
        load_scan("Direct hybrid HR", PLUS / "teacher_hybrid_noise", 0.50, 0.08, 1.00, "Direct mix, high recall"),
        load_scan("Direct hybrid ADL", PLUS / "teacher_hybrid_noise", 0.62, 0.08, 0.50, "Direct mix, ADL friendly"),
        load_scan("Variance hybrid HR", VAR_ROOT, 0.50, 0.08, 1.00, "Variance-preserving, high recall"),
        load_scan("Variance hybrid ADL", VAR_ROOT, 0.62, 0.08, 0.50, "Variance-preserving, ADL friendly"),
    ]


def write_reports(rows: list[dict[str, object]]) -> None:
    OUT_REPORTS.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_REPORTS / "variance_hybrid_noise_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (OUT_REPORTS / "variance_hybrid_noise_comparison.json").write_text(
        json.dumps({"rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def draw_figure(rows: list[dict[str, object]]) -> None:
    OUT_IMAGE.mkdir(parents=True, exist_ok=True)
    SHARED_IMAGE.mkdir(parents=True, exist_ok=True)

    labels = [str(row["group"]) for row in rows]
    x = list(range(len(rows)))
    width = 0.18

    acc = [float(row["accuracy"]) * 100 for row in rows]
    adl = [float(row["adl_specificity"]) * 100 for row in rows]
    fall = [float(row["fall_recall"]) * 100 for row in rows]
    prec = [float(row["precision"]) * 100 for row in rows]
    bal = [float(row["balanced_accuracy"]) * 100 for row in rows]

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold"})
    fig = plt.figure(figsize=(17, 9.5), dpi=180)
    grid = fig.add_gridspec(2, 1, height_ratios=[1.02, 0.98], hspace=0.22)
    ax = fig.add_subplot(grid[0])

    ax.bar([i - 1.5 * width for i in x], acc, width=width, color="#386cb0", label="Accuracy")
    ax.bar([i - 0.5 * width for i in x], adl, width=width, color="#35a16b", label="ADL specificity")
    ax.bar([i + 0.5 * width for i in x], fall, width=width, color="#d95f5f", label="Fall recall")
    ax.bar([i + 1.5 * width for i in x], prec, width=width, color="#8c6bb1", label="Precision")
    ax.plot(x, bal, color="#222222", linewidth=2.2, marker="o", markersize=5.5, label="Balanced accuracy")

    for i, value in enumerate(bal):
        ax.text(i, min(value + 3, 102), f"{value:.1f}", ha="center", va="bottom", fontsize=8.5, color="#222222")

    ax.set_title("Subject 4 Noise Strategy Comparison", fontsize=22, loc="left", pad=12)
    ax.set_ylabel("Video-level metric (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.24)
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
    table.set_fontsize(8.4)
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
    for row_index in [1, 5]:
        for col_index in range(len(columns)):
            table[row_index, col_index].set_facecolor("#fff4d6")

    output = OUT_IMAGE / "variance_hybrid_noise_comparison_ppt.png"
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
