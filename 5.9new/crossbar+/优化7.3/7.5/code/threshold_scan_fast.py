from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


def parse_float_list(values: list[str]) -> list[float]:
    parsed: list[float] = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                parsed.append(float(item))
    return parsed


def load_detector_module(detector_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("automatic_fall_detector_runtime", detector_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import detector from {detector_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def true_label_from_path(video_path: Path) -> str | None:
    parts = {part.lower() for part in video_path.parts}
    if "fall" in parts:
        return "fall"
    if "adl" in parts:
        return "adl"
    lower_name = video_path.stem.lower()
    if "fall" in lower_name:
        return "fall"
    if "adl" in lower_name:
        return "adl"
    return None


def safe_video_stem(video_path: Path, root: Path) -> str:
    try:
        relative = video_path.relative_to(root).with_suffix("")
        raw = "_".join(relative.parts)
    except ValueError:
        raw = video_path.stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")


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
        "detected_processes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def metric_summary(video_rows: list[dict[str, object]]) -> dict[str, float | int]:
    tp = fn = tn = fp = unknown = 0
    detected_processes = 0
    for row in video_rows:
        true_label = row.get("true_label")
        detected = bool(row.get("process_segment") is not None)
        if detected:
            detected_processes += 1
        if true_label == "fall":
            if detected:
                tp += 1
            else:
                fn += 1
        elif true_label == "adl":
            if detected:
                fp += 1
            else:
                tn += 1
        else:
            unknown += 1
    total = tp + fn + tn + fp
    fall_total = tp + fn
    adl_total = tn + fp
    fall_recall = tp / fall_total if fall_total else 0.0
    adl_specificity = tn / adl_total if adl_total else 0.0
    return {
        "video_accuracy": (tp + tn) / total if total else 0.0,
        "balanced_accuracy": (fall_recall + adl_specificity) / 2.0 if total else 0.0,
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
        "detected_processes": detected_processes,
    }


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
    parser = argparse.ArgumentParser(description="Fast threshold scan using cached per-frame probabilities.")
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--test-video-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment", default="")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--prob-values", nargs="+", default=["0.55,0.60,0.63,0.65,0.68,0.70,0.72,0.75"])
    parser.add_argument("--margin-values", nargs="+", default=["0.10"])
    parser.add_argument("--duration-values", nargs="+", default=["0.50,0.75"])
    parser.add_argument("--min-fall-recall", type=float, default=0.50)
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args()

    detector = load_detector_module(args.detector)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "frame_probability_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = detector.get_device(False)
    model, config = detector.load_model_checkpoint(args.model_path, device)
    window_size = int(config.get("window_size", 16))
    image_size = int(config.get("image_size", 128))
    infer_stride = int(config.get("infer_stride", 4))
    use_crossbar = bool(config.get("use_crossbar", True))
    use_device_dynamics = bool(config.get("use_device_dynamics", False))
    crossbar_readout_noise_scale = float(config.get("crossbar_readout_noise_scale", 0.0))
    use_pixel_human = bool(config.get("use_pixel_human", False))
    use_silhouette_human = bool(config.get("use_silhouette_human", False))
    pixel_grid_size = int(config.get("pixel_grid_size", 20))
    hrs_values, lrs_values = detector.load_conductance_pair(args.data_dir) if use_crossbar else (
        np.array([0.1], dtype=np.float32),
        np.array([0.9], dtype=np.float32),
    )
    device_dynamics = detector.load_device_dynamics(args.data_dir) if (
        use_crossbar and (use_device_dynamics or crossbar_readout_noise_scale > 0)
    ) else None

    raw_rows: list[dict[str, object]] = []
    videos = detector.list_test_videos(args.test_video_dir)
    for index, video_path in enumerate(videos, start=1):
        stem = safe_video_stem(video_path, args.test_video_dir)
        cache_path = cache_dir / f"{stem}.npz"
        if cache_path.exists() and not args.force_cache:
            cached = np.load(cache_path, allow_pickle=False)
            frame_probs = cached["frame_probs"]
            fps = float(cached["fps"])
            num_frames = int(cached["num_frames"])
        else:
            print(f"[{index}/{len(videos)}] infer {video_path.name}", flush=True)
            frames, fps = detector.read_video_frames(
                video_path,
                image_size=image_size,
                use_pixel_human=use_pixel_human,
                use_silhouette_human=use_silhouette_human,
                pixel_grid_size=pixel_grid_size,
            )
            inference = detector.predict_video_frames(
                model=model,
                frames=frames,
                fps=fps,
                window_size=window_size,
                stride=infer_stride,
                use_crossbar=use_crossbar,
                hrs_values=hrs_values,
                lrs_values=lrs_values,
                device_dynamics=device_dynamics,
                crossbar_readout_noise_scale=crossbar_readout_noise_scale,
                device=device,
                batch_size=max(1, args.batch_size),
                fall_prob_threshold=0.0,
                fall_margin_threshold=0.0,
                min_fall_duration_sec=0.0,
            )
            frame_probs = np.asarray(inference["frame_probabilities"], dtype=np.float32)
            num_frames = int(inference["num_frames"])
            np.savez_compressed(cache_path, frame_probs=frame_probs, fps=np.asarray(fps), num_frames=np.asarray(num_frames))
        raw_rows.append(
            {
                "video_path": str(video_path),
                "true_label": true_label_from_path(video_path),
                "fps": fps,
                "num_frames": num_frames,
                "frame_probs": frame_probs,
            }
        )

    prob_values = parse_float_list(args.prob_values)
    margin_values = parse_float_list(args.margin_values)
    duration_values = parse_float_list(args.duration_values)
    scan_rows: list[dict[str, object]] = []
    full_video_rows_by_key: dict[str, list[dict[str, object]]] = {}

    for prob in prob_values:
        for margin in margin_values:
            for duration in duration_values:
                threshold_key = f"p{prob:.2f}_m{margin:.2f}_d{duration:.2f}"
                video_rows: list[dict[str, object]] = []
                for raw in raw_rows:
                    fps = float(raw["fps"])
                    frame_probs = np.asarray(raw["frame_probs"], dtype=np.float32)
                    frame_preds = detector.calibrate_frame_predictions(
                        frame_probs,
                        fps=fps,
                        fall_prob_threshold=prob,
                        fall_margin_threshold=margin,
                        min_fall_duration_sec=duration,
                    )
                    process_segment = detector.select_process_segment(
                        frame_probs,
                        frame_preds,
                        fps,
                        fall_prob_threshold=prob,
                        fall_margin_threshold=margin,
                        min_fall_duration_sec=duration,
                    )
                    video_rows.append(
                        {
                            "video_path": raw["video_path"],
                            "true_label": raw["true_label"],
                            "fps": fps,
                            "num_frames": raw["num_frames"],
                            "process_segment": process_segment,
                        }
                    )
                metrics = metric_summary(video_rows)
                row: dict[str, object] = {
                    "fall_prob_threshold": prob,
                    "fall_margin_threshold": margin,
                    "min_fall_duration_sec": duration,
                }
                row.update(metrics)
                scan_rows.append(row)
                full_video_rows_by_key[threshold_key] = video_rows

    scan_rows.sort(
        key=lambda row: (
            float(row["fall_prob_threshold"]),
            float(row["fall_margin_threshold"]),
            float(row["min_fall_duration_sec"]),
        )
    )
    write_csv(scan_rows, args.output_dir / "threshold_scan_summary.csv")
    recommended = choose_recommended(scan_rows, min_fall_recall=args.min_fall_recall)
    recommended_key = (
        f"p{float(recommended['fall_prob_threshold']):.2f}_"
        f"m{float(recommended['fall_margin_threshold']):.2f}_"
        f"d{float(recommended['min_fall_duration_sec']):.2f}"
    )
    recommended_video_rows = full_video_rows_by_key[recommended_key]

    full_report = {
        "experiment": args.experiment,
        "model_path": str(args.model_path),
        "test_video_dir": str(args.test_video_dir),
        "min_fall_recall": args.min_fall_recall,
        "recommended": recommended,
        "rows": scan_rows,
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
                        "detected_processes",
                    ]
                },
                "selection_rule": f"Highest balanced accuracy with fall_recall >= {args.min_fall_recall:.2f}; falls back to max balanced accuracy if none qualify.",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    calibrated_summary = {
        "test_video_dir": str(args.test_video_dir),
        "num_videos": len(recommended_video_rows),
        "videos_with_detected_process": int(recommended["detected_processes"]),
        "threshold": {
            "fall_prob_threshold": recommended["fall_prob_threshold"],
            "fall_margin_threshold": recommended["fall_margin_threshold"],
            "min_fall_duration_sec": recommended["min_fall_duration_sec"],
        },
        "video_metrics": {
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
        "results": recommended_video_rows,
    }
    (args.output_dir.parent / "sub4_calibrated_summary.json").write_text(
        json.dumps(calibrated_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(recommended, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
