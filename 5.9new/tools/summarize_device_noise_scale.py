from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(r"F:\12.18-2")
DEVICE_ROOT = ROOT / "5.9new" / "crossbar+" / "teacher_device_noise"
GAUSSIAN_RECOMMENDED = ROOT / "5.9new" / "crossbar+" / "teacher_gaussian_noise" / "reports" / "recommended_threshold.json"


def true_label(video_path: str) -> str | None:
    parts = {part.lower() for part in Path(video_path).parts}
    if "fall" in parts:
        return "fall"
    if "adl" in parts:
        return "adl"
    return None


def summarize_batch(summary_path: Path) -> dict[str, float | int]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    tp = fn = tn = fp = 0
    for row in summary.get("results", []):
        label = true_label(str(row.get("video_path", "")))
        detected = row.get("process_segment") is not None
        if label == "fall":
            if detected:
                tp += 1
            else:
                fn += 1
        elif label == "adl":
            if detected:
                fp += 1
            else:
                tn += 1
    total = tp + fn + tn + fp
    fall_total = tp + fn
    adl_total = tn + fp
    fall_recall = tp / fall_total if fall_total else 0.0
    adl_specificity = tn / adl_total if adl_total else 0.0
    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "balanced_accuracy": (fall_recall + adl_specificity) / 2.0 if total else 0.0,
        "adl_specificity": adl_specificity,
        "fall_recall": fall_recall,
        "adl_correct": tn,
        "adl_total": adl_total,
        "fall_correct": tp,
        "fall_total": fall_total,
    }


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def read_scale_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scale_name, scale_value in [
        ("scale_025", 0.25),
        ("scale_050", 0.50),
        ("scale_075", 0.75),
        ("scale_100", 1.00),
    ]:
        reports = DEVICE_ROOT / scale_name / "reports"
        default_metrics = summarize_batch(reports / "sub4_summary.json")
        recommended = json.loads((reports / "recommended_threshold.json").read_text(encoding="utf-8"))
        threshold = recommended["recommended_threshold"]
        calibrated = recommended["subject4_result"]
        rows.append(
            {
                "scale_name": scale_name,
                "scale": scale_value,
                "default_accuracy": default_metrics["accuracy"],
                "default_balanced_accuracy": default_metrics["balanced_accuracy"],
                "default_adl_specificity": default_metrics["adl_specificity"],
                "default_fall_recall": default_metrics["fall_recall"],
                "default_adl_correct": default_metrics["adl_correct"],
                "default_fall_correct": default_metrics["fall_correct"],
                "fall_prob_threshold": threshold["fall_prob_threshold"],
                "fall_margin_threshold": threshold["fall_margin_threshold"],
                "min_fall_duration_sec": threshold["min_fall_duration_sec"],
                "calibrated_accuracy": calibrated["video_accuracy"],
                "calibrated_balanced_accuracy": calibrated["balanced_accuracy"],
                "calibrated_adl_specificity": calibrated["adl_specificity"],
                "calibrated_fall_recall": calibrated["fall_recall"],
                "calibrated_adl_correct": calibrated["adl_correct"],
                "calibrated_fall_correct": calibrated["fall_correct"],
            }
        )
    return rows


def read_gaussian_reference() -> dict[str, object] | None:
    if not GAUSSIAN_RECOMMENDED.exists():
        return None
    payload = json.loads(GAUSSIAN_RECOMMENDED.read_text(encoding="utf-8"))
    result = payload["subject4_calibrated_result"]
    threshold = payload["recommended_threshold"]
    return {
        "threshold": threshold,
        "accuracy": result["video_accuracy"],
        "adl_specificity": result["adl_specificity"],
        "fall_recall": result["fall_recall"],
    }


def write_summary_files(rows: list[dict[str, object]], gaussian: dict[str, object] | None) -> None:
    report_dir = DEVICE_ROOT / "reports"
    image_dir = DEVICE_ROOT / "image"
    report_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    csv_path = report_dir / "device_noise_scale_summary.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    json_path = report_dir / "device_noise_scale_summary.json"
    json_path.write_text(
        json.dumps({"scale_rows": rows, "gaussian_reference": gaussian}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def draw_ppt_figure(rows: list[dict[str, object]], gaussian: dict[str, object] | None) -> None:
    image_dir = DEVICE_ROOT / "image"
    scale_values = [float(row["scale"]) for row in rows]
    acc = [float(row["calibrated_accuracy"]) * 100 for row in rows]
    adl = [float(row["calibrated_adl_specificity"]) * 100 for row in rows]
    fall = [float(row["calibrated_fall_recall"]) * 100 for row in rows]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
        }
    )
    fig = plt.figure(figsize=(15, 8.5), dpi=180)
    grid = fig.add_gridspec(2, 1, height_ratios=[1.12, 0.88], hspace=0.18)
    ax = fig.add_subplot(grid[0])

    ax.plot(scale_values, acc, marker="o", linewidth=2.6, color="#2f6fbb", label="Accuracy")
    ax.plot(scale_values, adl, marker="s", linewidth=2.6, color="#2f9e72", label="ADL specificity")
    ax.plot(scale_values, fall, marker="^", linewidth=2.6, color="#c94b4b", label="Fall recall")
    if gaussian is not None:
        ax.axhline(float(gaussian["accuracy"]) * 100, color="#2f6fbb", linestyle="--", alpha=0.35)
        ax.axhline(float(gaussian["adl_specificity"]) * 100, color="#2f9e72", linestyle="--", alpha=0.35)
        ax.axhline(float(gaussian["fall_recall"]) * 100, color="#c94b4b", linestyle="--", alpha=0.35)
        ax.text(
            0.245,
            float(gaussian["accuracy"]) * 100 + 1.2,
            "Gaussian calibrated reference",
            fontsize=10,
            color="#555555",
        )

    best_row = max(rows, key=lambda item: float(item["calibrated_balanced_accuracy"]))
    ax.scatter(
        [float(best_row["scale"])],
        [float(best_row["calibrated_accuracy"]) * 100],
        s=150,
        facecolors="none",
        edgecolors="#111111",
        linewidths=2.0,
        zorder=5,
    )
    ax.set_title("Device Noise Scale Scan on Subject 4", fontsize=22, fontweight="bold", loc="left", pad=14)
    ax.set_xlabel("Device noise scale")
    ax.set_ylabel("Video-level metric (%)")
    ax.set_xticks(scale_values)
    ax.set_xticklabels([f"{value:.2f}x" for value in scale_values])
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", frameon=False, ncols=3)

    ax_table = fig.add_subplot(grid[1])
    ax_table.axis("off")
    columns = [
        "Scale",
        "Default\nAcc",
        "Default\nADL",
        "Default\nFall",
        "Best\nThreshold",
        "Calib\nAcc",
        "Calib\nADL",
        "Calib\nFall",
        "Bal Acc",
    ]
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                f"{float(row['scale']):.2f}x",
                pct(float(row["default_accuracy"])),
                f"{pct(float(row['default_adl_specificity']))}\n({row['default_adl_correct']}/20)",
                f"{pct(float(row['default_fall_recall']))}\n({row['default_fall_correct']}/17)",
                f"{float(row['fall_prob_threshold']):.2f}, {float(row['min_fall_duration_sec']):.2f}s",
                pct(float(row["calibrated_accuracy"])),
                f"{pct(float(row['calibrated_adl_specificity']))}\n({row['calibrated_adl_correct']}/20)",
                f"{pct(float(row['calibrated_fall_recall']))}\n({row['calibrated_fall_correct']}/17)",
                pct(float(row["calibrated_balanced_accuracy"])),
            ]
        )
    if gaussian is not None:
        threshold = gaussian["threshold"]
        table_rows.append(
            [
                "Gaussian",
                "-",
                "-",
                "-",
                f"{float(threshold['fall_prob_threshold']):.2f}, {float(threshold['min_fall_duration_sec']):.2f}s",
                pct(float(gaussian["accuracy"])),
                pct(float(gaussian["adl_specificity"])),
                pct(float(gaussian["fall_recall"])),
                pct((float(gaussian["adl_specificity"]) + float(gaussian["fall_recall"])) / 2.0),
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
    table.set_fontsize(9.5)
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
    best_index = rows.index(best_row) + 1
    for c in range(len(columns)):
        table[best_index, c].set_facecolor("#fff4d6")

    output_path = image_dir / "device_noise_scale_scan_ppt.png"
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    shared_image_dir = ROOT / "5.9new" / "image"
    shared_image_dir.mkdir(parents=True, exist_ok=True)
    shared_output = shared_image_dir / "device_noise_scale_scan_ppt.png"
    shared_output.write_bytes(output_path.read_bytes())


def main() -> int:
    rows = read_scale_rows()
    gaussian = read_gaussian_reference()
    write_summary_files(rows, gaussian)
    draw_ppt_figure(rows, gaussian)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
