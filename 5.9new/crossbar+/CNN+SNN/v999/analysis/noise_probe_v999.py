"""Sanity probe: how much do per-video max frame probabilities actually move
under readout noise 15% vs 90%? Explains the flat robustness curve."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import strict_eval_s30 as sev

SNAPSHOT = HERE.parent
MODEL_PATH = SNAPSHOT / "models" / "automatic_fall_event_detector_c1_s34_data1_s11.pth"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mod = sev.load_module()
    model, config = mod.load_model_checkpoint(MODEL_PATH, device)
    videos = sorted(mod.list_video_files(sev.SUBJECT4_DIR))
    picks = [v for v in videos if any(k in str(v) for k in ("ADL\\05", "ADL\\06", "Fall\\08", "Fall\\09", "Fall\\13", "ADL\\10"))]
    for pct in (15, 90):
        cfg = dict(config)
        cfg["crossbar_readout_noise_scale"] = pct / 100.0
        print(f"--- noise {pct}% ---")
        for v in picks:
            res = sev.predict_video(mod, model, cfg, v, device)
            probs = np.asarray(res["frame_probabilities"], dtype=np.float32)
            name = f"{v.parent.name}/{v.stem}"
            print(f"{name:12s} max={probs.max():.3f} p95={np.percentile(probs, 95):.3f} "
                  f"mean={probs.mean():.3f} true={res['true_label']}", flush=True)


if __name__ == "__main__":
    main()
