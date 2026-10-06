from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(r"F:\12.18-2")
HYBRID_ROOT = ROOT / "5.9new" / "crossbar+" / "teacher_hybrid_noise"
DEVICE_SUMMARY = ROOT / "5.9new" / "crossbar+" / "teacher_device_noise" / "reports" / "device_noise_scale_summary.csv"
GAUSSIAN_RECOMMENDED = ROOT / "5.9new" / "crossbar+" / "teacher_gaussian_noise" / "reports" / "recommended_threshold.json"


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def load_gaussian() -> dict[str, object]:
    payload = json.loads(GAUSSIAN_RECOMMENDED.read_text(encoding="utf-8"))
    threshold = payload["recommended_threshold"]
    result = payload["subject4_calibrated_result"]
    return {
        "group": "Gaussian",
        "threshold": f"{float(threshold['fall_prob_threshold']):.2f}, {float(threshold['min_fall_duration_sec']):.2f}s",
        "accuracy": float(result["video_accuracy"]),
        "adl_specificity": float(result["adl_specificity"]),
        "fall_recall": float(result["fall_recall"]),
        "balanced_accuracy": (float(result["adl_specificity"]) + float(result["fall_recall"])) / 2.0,
        "adl_correct": int(result["adl_correct"]),
        "fall_correct": int(result["fall_correct"]),
    }


def load_device_best() -> dict[str, object]:
    rows = list(csv.DictReader(DEVICE_SUMMARY.open(encoding="utf-8")))
    best = max(rows, key=lambda row: float(row["calibrated_balanced_accuracy"]))
    return {
        "group": f"Device {float(best['scale']):.2f}x",
        "threshold": f"{float(best['fall_prob_threshold']):.2f}, {float(best['min_fall_duration_sec']):.2f}s",
        "accuracy": float(best["calibrated_accuracy"]),
        "adl_specificity": float(best["calibrated_adl_specificity"]),
        "fall_recall": float(best["calibrated_fall_recall"]),
        "balanced_accuracy": float(best["calibrated_balanced_accuracy"]),
        "adl_correct": int(best["calibrated_adl_correct"]),
        "fall_correct": int(best["calibrated_fall_correct"]),
    }


def find_scan_row(prob: float, margin: float, duration: float) -> dict[str, str]:
    scan_path = HYBRID_ROOT / "reports" / "threshold_scan" / "threshold_scan_summary.csv"
    rows = list(csv.DictReader(scan_path.open(encoding="utf-8")))
    for row in rows:
        if (
            abs(float(row["fall_prob_threshold"]) - prob) < 1e-8
            and abs(float(row["fall_margin_threshold"]) - margin) < 1e-8
            and abs(float(row["min_fall_duration_sec"]) - duration) < 1e-8
        ):
            return row
    raise ValueError(f"Missing hybrid scan row prob={prob} margin={margin} duration={duration}")


def hybrid_row(label: str, prob: float, margin: float, duration: float) -> dict[str, object]:
    row = find_scan_row(prob, margin, duration)
    return {
        "group": label,
        "threshold": f"{prob:.2f}, {duration:.2f}s",
        "accuracy": float(row["video_accuracy"]),
        "adl_specificity": float(row["adl_specificity"]),
        "fall_recall": float(row["fall_recall"]),
        "balanced_accuracy": float(row["balanced_accuracy"]),
        "adl_correct": int(row["adl_correct"]),
        "fall_correct": int(row["fall_correct"]),
    }


def build_rows() -> list[dict[str, object]]:
    return [
        load_gaussian(),
        load_device_best(),
        hybrid_row("Hybrid high recall", 0.50, 0.08, 1.00),
        hybrid_row("Hybrid ADL friendly", 0.62, 0.08, 0.50),
    ]


def write_reports(rows: list[dict[str, object]]) -> None:
    reports = HYBRID_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    csv_path = reports / "hybrid_noise_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    (reports / "hybrid_noise_comparison.json").write_text(
        json.dumps({"rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def draw_figure(rows: list[dict[str, object]]) -> None:
    image_dir = HYBRID_ROOT / "image"
    shared_image_dir = ROOT / "5.9new" / "image"
    image_dir.mkdir(parents=True, exist_ok=True)
    shared_image_dir.mkdir(parents=True, exist_ok=True)

    labels = [str(row["group"]) for row in rows]
    x = list(range(len(rows)))
    width = 0.22

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(15, 8.5), dpi=180)
    grid = fig.add_gridspec(2, 1, height_ratios=[1.05, 0.95], hspace=0.18)
    ax = fig.add_subplot(grid[0])

    acc = [float(row["accuracy"]) * 100 for row in rows]
    adl = [float(row["adl_specificity"]) * 100 for row in rows]
    fall = [float(row["fall_recall"]) * 100 for row in rows]
    bal = [float(row["balanced_accuracy"]) * 100 for row in rows]

    ax.bar([i - width for i in x], acc, width=width, color="#2f6fbb", label="Accuracy")
    ax.bar(x, adl, width=width, color="#2f9e72", label="ADL specificity")
    ax.bar([i + width for i in x], fall, width=width, color="#c94b4b", label="Fall recall")
    ax.plot(x, bal, color="#222222", linewidth=2.3, marker="o", label="Balanced accuracy")

    ax.set_title("Gaussian vs Device vs Hybrid Noise on Subject 4", fontsize=22, fontweight="bold", loc="left", pad=14)
    ax.set_ylabel("Video-level metric (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", frameon=False, ncols=4)

    ax_table = fig.add_subplot(grid[1])
    ax_table.axis("off")
    columns = ["Group", "Threshold", "Acc", "ADL", "Fall", "Bal Acc", "Interpretation"]
    interpretations = {
        "Gaussian": "Best overall",
        "Device 0.25x": "Device-only best",
        "Hybrid high recall": "Fewest missed falls",
        "Hybrid ADL friendly": "Fewer ADL false alarms",
    }
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row["group"],
                row["threshold"],
                pct(float(row["accuracy"])),
                f"{pct(float(row['adl_specificity']))}\n({row['adl_correct']}/20)",
                f"{pct(float(row['fall_recall']))}\n({row['fall_correct']}/17)",
                pct(float(row["balanced_accuracy"])),
                interpretations.get(str(row["group"]), ""),
            ]
        )
    table = ax_table.table(
        cellText=table_rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d7d7d7")
        if r == 0:
            cell.set_facecolor("#202833")
            cell.set_text_props(color="white", fontweight="bold")
        elif c == 0:
            cell.set_facecolor("#edf2f7")
            cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("#ffffff" if r % 2 else "#f8fafc")
    for c in range(len(columns)):
        table[1, c].set_facecolor("#fff4d6")

    output = image_dir / "hybrid_noise_comparison_ppt.png"
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    (shared_image_dir / output.name).write_bytes(output.read_bytes())


def main() -> int:
    rows = build_rows()
    write_reports(rows)
    draw_figure(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
