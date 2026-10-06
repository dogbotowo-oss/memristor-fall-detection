from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(r"F:\12.18-2")
EXP_DIR = ROOT / "5.9new" / "crossbar+" / "CNN+SNN" / "v14"
CODE_PATH = EXP_DIR / "code" / "automatic_fall_detector.py"
DATA_DIR = ROOT / "data"
GMDCSA_ROOT = (
    ROOT
    / "测试集"
    / "13354453"
    / "ekramalam"
    / "GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-v2.1"
    / "ekramalam-GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-5abac76"
)
SUBJECT3_DIR = GMDCSA_ROOT / "Subject 3"
SUBJECT4_DIR = ROOT / "gmdcsa_subject4_test" / "Subject 4"
ZENODO_VAL_DIR = ROOT / "zenodo_falldb_video_split" / "val"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict Subject3/Subject4 eval for v14 variants.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--prefix", type=str, required=True)
    return parser.parse_args()


def load_module():
    spec = importlib.util.spec_from_file_location("automatic_fall_detector_v14_variant", CODE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import module from {CODE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def video_true_label(mod, video_path: Path) -> int:
    label_id = int(mod.infer_video_level_label_id(video_path, "fall"))
    return 1 if label_id in {mod.LABEL_NAME_TO_ID["pre_fall"], mod.LABEL_NAME_TO_ID["fall"]} else 0


def predict_video(mod, model, config: dict[str, object], video_path: Path, device: torch.device) -> dict[str, object]:
    window_size = int(config.get("window_size", 16))
    image_size = int(config.get("image_size", 64))
    infer_stride = int(config.get("infer_stride", 4))
    use_crossbar = bool(config.get("use_crossbar", True))
    use_device_dynamics = bool(config.get("use_device_dynamics", False))
    crossbar_readout_noise_scale = float(config.get("crossbar_readout_noise_scale", 0.0))
    use_pixel_human = bool(config.get("use_pixel_human", False))
    use_silhouette_human = bool(config.get("use_silhouette_human", False))
    pixel_grid_size = int(config.get("pixel_grid_size", 20))
    hrs_values, lrs_values = mod.load_conductance_pair(DATA_DIR) if use_crossbar else (
        np.array([0.1], dtype=np.float32),
        np.array([0.9], dtype=np.float32),
    )
    device_dynamics = mod.load_device_dynamics(DATA_DIR) if (
        use_crossbar and (use_device_dynamics or crossbar_readout_noise_scale > 0)
    ) else None

    frames, fps = mod.read_video_frames(
        video_path,
        image_size=image_size,
        use_pixel_human=use_pixel_human,
        use_silhouette_human=use_silhouette_human,
        pixel_grid_size=pixel_grid_size,
    )
    inference = mod.predict_video_frames(
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
        batch_size=max(1, int(config.get("batch_size", 8))),
        fall_prob_threshold=0.0,
        fall_margin_threshold=0.0,
        min_fall_duration_sec=0.0,
    )
    return {
        "video_path": str(video_path),
        "fps": float(fps),
        "num_frames": int(inference["num_frames"]),
        "frame_probabilities": inference["frame_probabilities"],
        "frame_predictions": inference["frame_predictions"],
        "true_label": video_true_label(mod, video_path),
    }


def classify_video(mod, result: dict[str, object], threshold: float) -> int:
    frame_probs = np.asarray(result["frame_probabilities"], dtype=np.float32)
    fps = float(result["fps"])
    frame_preds = mod.calibrate_frame_predictions(
        frame_probs,
        fps=fps,
        fall_prob_threshold=float(threshold),
        fall_margin_threshold=0.0,
        min_fall_duration_sec=0.0,
    )
    process_segment = mod.select_process_segment(
        frame_probs,
        frame_preds,
        fps,
        fall_prob_threshold=float(threshold),
        fall_margin_threshold=0.0,
        min_fall_duration_sec=0.0,
    )
    return 1 if process_segment is not None else 0


def compute_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    tp = sum(1 for row in rows if row["true_label"] == 1 and row["pred_label"] == 1)
    tn = sum(1 for row in rows if row["true_label"] == 0 and row["pred_label"] == 0)
    fp = sum(1 for row in rows if row["true_label"] == 0 and row["pred_label"] == 1)
    fn = sum(1 for row in rows if row["true_label"] == 1 and row["pred_label"] == 0)
    total = len(rows)
    accuracy = (tp + tn) / total if total else 0.0
    adl_specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fall_recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    balanced_accuracy = 0.5 * (adl_specificity + fall_recall)
    return {
        "total": total,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "adl_specificity": adl_specificity,
        "fall_recall": fall_recall,
        "precision": precision,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    mod = load_module()
    mod.log = lambda *_args, **_kwargs: None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = mod.load_model_checkpoint(args.model_path, device)

    validation_videos = sorted(dict.fromkeys(mod.list_video_files(SUBJECT3_DIR) + mod.list_video_files(ZENODO_VAL_DIR)))
    cached = [predict_video(mod, model, config, path, device) for path in validation_videos]
    thresholds = [round(value, 2) for value in np.arange(0.35, 0.71, 0.05)]
    rows: list[dict[str, object]] = []
    best_threshold = thresholds[0]
    best_score = -1.0
    best_spec = -1.0
    tolerance = 1e-9

    for threshold in thresholds:
        eval_rows = [
            {
                "video_path": item["video_path"],
                "true_label": int(item["true_label"]),
                "pred_label": int(classify_video(mod, item, threshold)),
            }
            for item in cached
        ]
        metrics = compute_metrics(eval_rows)
        row = {"threshold": threshold, **metrics}
        rows.append(row)
        score = float(metrics["balanced_accuracy"])
        spec = float(metrics["adl_specificity"])
        better = score > best_score + tolerance
        tied = abs(score - best_score) <= tolerance
        better_tie = tied and (
            spec > best_spec + tolerance
            or (abs(spec - best_spec) <= tolerance and float(threshold) > float(best_threshold))
        )
        if better or better_tie:
            best_threshold = threshold
            best_score = score
            best_spec = spec

    scan_csv = EXP_DIR / "reports" / f"{args.prefix}_strict_validation_threshold_scan.csv"
    write_csv(
        scan_csv,
        rows,
        ["threshold", "total", "accuracy", "balanced_accuracy", "adl_specificity", "fall_recall", "precision", "tp", "tn", "fp", "fn"],
    )
    best_payload = {
        "threshold": float(best_threshold),
        "source": "Subject3 + Zenodo val",
        "scan_csv": str(scan_csv),
        "threshold_selection_policy": "maximize balanced_accuracy; ties choose higher ADL specificity; remaining ties choose higher threshold",
    }
    (EXP_DIR / "reports" / f"{args.prefix}_recommended_threshold.json").write_text(
        json.dumps(best_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    subject4_rows: list[dict[str, object]] = []
    false_alarms: list[str] = []
    missed_falls: list[str] = []
    for video_path in mod.list_video_files(SUBJECT4_DIR):
        item = predict_video(mod, model, config, video_path, device)
        pred_label = classify_video(mod, item, best_threshold)
        true_label = int(item["true_label"])
        subject4_rows.append({"video_path": str(video_path), "true_label": true_label, "pred_label": pred_label})
        if true_label == 0 and pred_label == 1:
            false_alarms.append(str(video_path))
        elif true_label == 1 and pred_label == 0:
            missed_falls.append(str(video_path))

    summary = compute_metrics(subject4_rows)
    summary.update(
        {
            "threshold": float(best_threshold),
            "false_alarms": "; ".join(false_alarms),
            "missed_falls": "; ".join(missed_falls),
            "false_alarm_count": len(false_alarms),
            "missed_fall_count": len(missed_falls),
            "source": "Subject4",
        }
    )
    subject4_csv = EXP_DIR / "reports" / f"{args.prefix}_subject4_final_video_metrics.csv"
    write_csv(subject4_csv, subject4_rows, ["video_path", "true_label", "pred_label"])
    summary["video_metrics_csv"] = str(subject4_csv)
    (EXP_DIR / "reports" / f"{args.prefix}_subject4_final_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"recommended_threshold": best_threshold, "subject4": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
