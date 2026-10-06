from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def parse_float_list(values: list[str]) -> list[float]:
    parsed: list[float] = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                parsed.append(float(item))
    return parsed


def video_true_label(video_path: str) -> str | None:
    normalized = video_path.replace("\\", "/").lower()
    parts = [part for part in normalized.split("/") if part]
    if "fall" in parts:
        return "fall"
    if "adl" in parts:
        return "adl"
    name = Path(video_path).stem.lower()
    if "fall" in name:
        return "fall"
    if "adl" in name:
        return "adl"
    return None


def summarize_batch(summary_json: Path) -> dict[str, float | int]:
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    tp = fn = tn = fp = 0
    unknown = 0
    for result in summary.get("results", []):
        true_label = video_true_label(str(result.get("video_path", "")))
        process = result.get("process_segment")
        predicted_fall = bool(process and process.get("label_name") == "fall")
        if true_label == "fall":
            if predicted_fall:
                tp += 1
            else:
                fn += 1
        elif true_label == "adl":
            if predicted_fall:
                fp += 1
            else:
                tn += 1
        else:
            unknown += 1

    total = tp + fn + tn + fp
    fall_total = tp + fn
    adl_total = tn + fp
    accuracy = (tp + tn) / total if total else 0.0
    fall_recall = tp / fall_total if fall_total else 0.0
    adl_specificity = tn / adl_total if adl_total else 0.0
    balanced_accuracy = (fall_recall + adl_specificity) / 2.0 if total else 0.0
    return {
        "video_accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "adl_specificity": adl_specificity,
        "fall_recall": fall_recall,
        "adl_correct": tn,
        "adl_total": adl_total,
        "fall_correct": tp,
        "fall_total": fall_total,
        "false_alarms": fp,
        "missed_falls": fn,
        "unknown_videos": unknown,
        "total_videos": total,
    }


def threshold_stem(prob: float, margin: float, duration: float) -> str:
    return f"p{prob:.2f}_m{margin:.2f}_d{duration:.2f}".replace(".", "p")


def run_batch_test(args: argparse.Namespace, prob: float, margin: float, duration: float, run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_json = run_dir / "sub4_summary.json"
    if summary_json.exists() and not args.force:
        return summary_json

    command = [
        args.python_exe,
        str(args.detector),
        "--mode",
        "batch_test",
        "--skip-visual-outputs",
        "--data-dir",
        str(args.data_dir),
        "--model-path",
        str(args.model_path),
        "--test-video-dir",
        str(args.test_video_dir),
        "--test-output-dir",
        str(run_dir / "outputs"),
        "--test-batch-summary-json",
        str(summary_json),
        "--fall-prob-threshold",
        f"{prob:.4f}",
        "--fall-margin-threshold",
        f"{margin:.4f}",
        "--min-fall-duration-sec",
        f"{duration:.4f}",
        "--batch-size",
        str(args.batch_size),
    ]
    subprocess.run(command, check=True)
    return summary_json


def write_csv(rows: list[dict[str, object]], csv_path: Path) -> None:
    fieldnames = [
        "fall_prob_threshold",
        "fall_margin_threshold",
        "min_fall_duration_sec",
        "video_accuracy",
        "balanced_accuracy",
        "adl_specificity",
        "fall_recall",
        "adl_correct",
        "adl_total",
        "fall_correct",
        "fall_total",
        "false_alarms",
        "missed_falls",
        "summary_json",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def choose_recommended(rows: list[dict[str, object]], min_fall_recall: float) -> dict[str, object]:
    viable = [row for row in rows if float(row["fall_recall"]) >= min_fall_recall]
    candidates = viable if viable else rows
    return max(
        candidates,
        key=lambda row: (
            float(row["balanced_accuracy"]),
            float(row["video_accuracy"]),
            float(row["fall_recall"]),
            float(row["adl_specificity"]),
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run threshold scan for batch_test outputs.")
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--test-video-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment", default="")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--prob-values", nargs="+", default=["0.55,0.60,0.63,0.65,0.68,0.70,0.72,0.75"])
    parser.add_argument("--margin-values", nargs="+", default=["0.10"])
    parser.add_argument("--duration-values", nargs="+", default=["0.50,0.75"])
    parser.add_argument("--min-fall-recall", type=float, default=0.50)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = args.output_dir / "runs"
    rows: list[dict[str, object]] = []
    prob_values = parse_float_list(args.prob_values)
    margin_values = parse_float_list(args.margin_values)
    duration_values = parse_float_list(args.duration_values)
    total_runs = len(prob_values) * len(margin_values) * len(duration_values)
    run_index = 0

    for prob in prob_values:
        for margin in margin_values:
            for duration in duration_values:
                run_index += 1
                stem = threshold_stem(prob, margin, duration)
                print(f"[{run_index}/{total_runs}] {stem}", flush=True)
                summary_json = run_batch_test(args, prob, margin, duration, runs_dir / stem)
                metrics = summarize_batch(summary_json)
                row: dict[str, object] = {
                    "fall_prob_threshold": prob,
                    "fall_margin_threshold": margin,
                    "min_fall_duration_sec": duration,
                    "summary_json": str(summary_json),
                }
                row.update(metrics)
                rows.append(row)
                write_csv(rows, args.output_dir / "threshold_scan_summary.csv")

    recommended = choose_recommended(rows, min_fall_recall=args.min_fall_recall)
    full_report = {
        "experiment": args.experiment,
        "model_path": str(args.model_path),
        "test_video_dir": str(args.test_video_dir),
        "min_fall_recall": args.min_fall_recall,
        "recommended": recommended,
        "rows": rows,
    }
    (args.output_dir / "threshold_scan_full.json").write_text(
        json.dumps(full_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output_dir / "recommended_threshold.json").write_text(
        json.dumps(
            {
                "experiment": args.experiment,
                "recommended_threshold": {
                    "fall_prob_threshold": recommended["fall_prob_threshold"],
                    "fall_margin_threshold": recommended["fall_margin_threshold"],
                    "min_fall_duration_sec": recommended["min_fall_duration_sec"],
                },
                "subject4_result": {
                    key: recommended[key]
                    for key in [
                        "video_accuracy",
                        "balanced_accuracy",
                        "adl_specificity",
                        "fall_recall",
                        "adl_correct",
                        "adl_total",
                        "fall_correct",
                        "fall_total",
                        "false_alarms",
                        "missed_falls",
                    ]
                },
                "selection_rule": f"Highest balanced accuracy with fall_recall >= {args.min_fall_recall:.2f}; falls back to max balanced accuracy if none qualify.",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps(full_report["recommended"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
