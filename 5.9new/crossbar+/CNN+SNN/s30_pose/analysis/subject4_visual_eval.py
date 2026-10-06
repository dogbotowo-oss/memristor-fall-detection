"""Batch Subject4 inference with per-video visual outputs for s32/s33.

For each model x threshold config, runs pose-aware inference on all 37
Subject4 videos and saves: detection heatmap PNG, key-frame JPG, and a
per-video CSV (true/pred label, max fall prob, detected segment).

Usage: python subject4_visual_eval.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import strict_eval_s30 as sev

S30 = HERE.parent
MODELS = S30 / "models"
REPORTS = S30 / "reports"
IMAGE_ROOT = S30 / "image"

JOBS = [
    (MODELS / "automatic_fall_event_detector_c1_s32_posefull_floor020.pth", "s32", 0.70),
    (MODELS / "automatic_fall_event_detector_c1_s33_posefull_floor020_lam020_aux030.pth", "s33", 0.60),
]


def run_model(model_path: Path, name: str, threshold: float) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mod = sev.load_module()
    model, config = mod.load_model_checkpoint(model_path, device)
    out_dir = IMAGE_ROOT / f"subject4_{name}_thr{int(round(threshold * 100)):03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = mod.list_video_files(sev.SUBJECT4_DIR)
    rows = []
    for i, video_path in enumerate(videos, 1):
        item = sev.predict_video(mod, model, config, video_path, device)
        frame_probs = np.asarray(item["frame_probabilities"], dtype=np.float32)
        fps = float(item["fps"])
        frame_preds = mod.calibrate_frame_predictions(
            frame_probs, fps=fps, fall_prob_threshold=threshold,
            fall_margin_threshold=0.0, min_fall_duration_sec=0.0,
        )
        segment = mod.select_process_segment(
            frame_probs, frame_preds, fps, fall_prob_threshold=threshold,
            fall_margin_threshold=0.0, min_fall_duration_sec=0.0,
        )
        pred = 1 if segment is not None else 0
        true = int(item["true_label"])
        fall_id = mod.LABEL_NAME_TO_ID["fall"]
        max_fall_prob = float(frame_probs[:, fall_id].max())
        rel = f"{video_path.parent.name}/{video_path.name}"
        stem = f"{video_path.parent.name}_{video_path.stem}"
        mod.save_detection_plot(
            out_dir / f"{stem}_detection.png", fps, frame_probs, frame_preds,
            None, show_plot=False, save_plot=True,
        )
        key_frame = mod.select_key_frame(frame_probs)
        mod.save_key_frame_visual(video_path, out_dir / f"{stem}_keyframe.jpg", key_frame, fps)
        seg_text = ""
        if segment is not None:
            seg_text = f"{int(segment['start_frame'])}-{int(segment['end_frame'])}"
        rows.append({
            "video": rel, "true": true, "pred": pred,
            "max_fall_prob": round(max_fall_prob, 4),
            "segment_frames": seg_text,
            "status": "OK" if pred == true else ("MISS" if true == 1 else "FALSE_ALARM"),
        })
        print(f"[{name} {i}/{len(videos)}] {rel} true={true} pred={pred} maxP={max_fall_prob:.2f} {rows[-1]['status']}", flush=True)

    csv_path = REPORTS / f"subject4_pervideo_{name}_thr{int(round(threshold * 100)):03d}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "true", "pred", "max_fall_prob", "segment_frames", "status"])
        writer.writeheader()
        writer.writerows(rows)
    ok = sum(1 for r in rows if r["status"] == "OK")
    print(f"[{name}] thr={threshold} correct={ok}/{len(rows)} csv={csv_path}", flush=True)


def main() -> None:
    for model_path, name, threshold in JOBS:
        if not model_path.exists():
            print(f"missing model: {model_path}", flush=True)
            continue
        run_model(model_path, name, threshold)
    print("VISUAL EVAL DONE", flush=True)


if __name__ == "__main__":
    main()
