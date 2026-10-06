"""Probe: S1 ADL max-frame-prob distribution under the LOSO fold-1 model.

Answers "is spec=100 on S1 overfitting?" by showing how far S1 ADL scores
sit below the frozen threshold 0.70.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

S30 = Path(r"E:\rray\12.18-2\5.9new\crossbar+\CNN+SNN\s30_pose")
sys.path.insert(0, str(S30 / "analysis"))

import strict_eval_s30 as sev

sev.DATA_DIR = Path(r"E:\rray\12.18-2\data1\data")
MODEL = S30 / "models" / "automatic_fall_event_detector_c1_s34_data1_loso_ts1_s11.pth"
S1 = (
    Path(r"E:\rray\12.18-2\测试集\13354453\ekramalam")
    / "GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-v2.1"
    / "ekramalam-GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-5abac76"
    / "Subject 1"
)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mod = sev.load_module()
    model, config = mod.load_model_checkpoint(MODEL, device)
    fall_id = mod.LABEL_NAME_TO_ID["fall"]
    pre_fall_id = mod.LABEL_NAME_TO_ID["pre_fall"]
    videos = sorted(mod.list_video_files(S1))
    adl_scores, fall_scores = [], []
    for v in videos:
        res = sev.predict_video(mod, model, config, v, device)
        p = np.asarray(res["frame_probabilities"], dtype=np.float32)
        pos = np.maximum(p[:, fall_id], p[:, pre_fall_id])  # fall-class prob
        name = f"{v.parent.name}/{v.stem}"
        (fall_scores if res["true_label"] == 1 else adl_scores).append((float(pos.max()), name))
    adl_scores.sort(reverse=True)
    fall_scores.sort()
    print("S1 ADL max-prob (top5 of %d):" % len(adl_scores))
    for s, n in adl_scores[:5]:
        print(f"  {n:12s} {s:.3f}")
    print("S1 Fall max-prob (bottom5 of %d):" % len(fall_scores))
    for s, n in fall_scores[:5]:
        print(f"  {n:12s} {s:.3f}")
    a = np.array([s for s, _ in adl_scores])
    print(f"ADL max-prob: max={a.max():.3f} p75={np.percentile(a,75):.3f} median={np.median(a):.3f} (thr=0.70)")


if __name__ == "__main__":
    main()
