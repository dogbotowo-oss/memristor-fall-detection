"""Pose-off ablation: zero out pose features and re-evaluate a trained model.

Usage:
    python pose_off_ablation.py <model_path> <tag>
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import strict_eval_s30 as sev


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: pose_off_ablation.py <model_path> <tag>")
    model_path = Path(sys.argv[1])
    tag = sys.argv[2]
    threshold = 0.70

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mod = sev.load_module()

    original_loader = mod.load_pose_frame_features

    def zeroed_pose(video_path, frame_count, fps, pose_cache_dir):
        return np.zeros((frame_count, mod.POSE_FEATURE_DIM), dtype=np.float32)

    mod.load_pose_frame_features = zeroed_pose
    try:
        model, config = mod.load_model_checkpoint(model_path, device)

        validation_videos = mod.list_video_files(sev.SUBJECT3_DIR) + mod.list_video_files(sev.ZENODO_VAL_DIR)
        validation_videos = sorted(dict.fromkeys(validation_videos))
        _, val_summary = sev.evaluate_at_threshold(mod, model, config, validation_videos, threshold, device)
        val_summary.pop("false_alarms", None)
        val_summary.pop("missed_falls", None)

        subject4_videos = mod.list_video_files(sev.SUBJECT4_DIR)
        _, s4_summary = sev.evaluate_at_threshold(mod, model, config, subject4_videos, threshold, device)

        result = {"tag": tag, "model_path": str(model_path),
                  "val_at_070": val_summary, "subject4": s4_summary}
        out_path = sev.REPORTS_DIR / f"{tag}.json"
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        mod.load_pose_frame_features = original_loader


if __name__ == "__main__":
    main()
