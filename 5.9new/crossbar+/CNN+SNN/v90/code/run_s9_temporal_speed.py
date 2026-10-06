"""S9: temporal speed augmentation on top of S8 (train-time late fusion).

Patches load_external_labeled_windows to expand the training set with
temporally slowed-down fall windows (0.5x speed via linear interpolation
along the time axis) plus horizontal flips, then runs the r2/S8 pipeline.
Val windows (Subject3 + Zenodo val) are never augmented.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# --- load r2 runner (which loads base) ---
R2_PATH = Path(__file__).with_name("run_refined_gate_r2.py")
spec = importlib.util.spec_from_file_location("r2_s9", R2_PATH)
r2 = importlib.util.module_from_spec(spec)
sys.modules["r2_s9"] = r2
spec.loader.exec_module(r2)
base = r2.base


def _parse_s9_aug_mode() -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--s9-aug-mode", choices=["slow_flip", "slow", "slow_flip_half"], default="slow_flip")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return str(args.s9_aug_mode)


S9_AUG_MODE = _parse_s9_aug_mode()


def temporal_speed_augment(windows: np.ndarray, speed: float) -> np.ndarray:
    """Slow down windows by ``speed`` factor via temporal interpolation.

    windows: [N, T, H, W] float32
    Returns: [N, T, H, W] with motion stretched by 1/speed.
    """
    n, t_len, h, w = windows.shape
    target = max(t_len + 1, int(round(t_len / speed)))
    x = torch.from_numpy(windows).float()
    # [N*H*W, 1, T] for linear interpolate along time
    x = x.permute(0, 2, 3, 1).reshape(-1, 1, t_len)
    x = F.interpolate(x, size=target, mode="linear", align_corners=True)
    idx = torch.linspace(0, target - 1, t_len).long()
    x = x[:, :, idx]
    x = x.reshape(n, h, w, t_len).permute(0, 3, 1, 2)
    return x.numpy().astype(np.float32)


_original_load = base.load_external_labeled_windows


def patched_load(video_dir, **kwargs):
    result = _original_load(video_dir, **kwargs)
    windows, labels, summary = result
    # Detect val dirs: Subject 3 or "val" in path
    dirs = video_dir if isinstance(video_dir, list) else [video_dir]
    is_val = any("Subject 3" in str(d) or "val" in str(d).lower() for d in dirs)
    if is_val or len(windows) == 0:
        return result
    fall_id = base.LABEL_NAME_TO_ID["fall"]
    fall_mask = labels == fall_id
    if fall_mask.sum() == 0:
        return result
    fall_windows = windows[fall_mask]
    n_fall = int(fall_mask.sum())
    # 0.5x slowed + horizontal flip of the slowed
    slow = temporal_speed_augment(fall_windows, 0.5)
    if S9_AUG_MODE == "slow":
        variants = [slow]
    else:
        variants = [slow, slow[:, :, :, ::-1].copy()]
    if S9_AUG_MODE == "slow_flip_half":
        # S9c-wt: keep 50% of each augmented variant (seeded), which matches an
        # 0.5 per-sample loss weight in expectation while preserving both
        # slow and flipped-slow diversity.
        rng = np.random.default_rng(20260804)
        halves = []
        for variant in variants:
            keep = rng.random(len(variant)) < 0.5
            halves.append(variant[keep])
        variants = halves
    aug = np.concatenate(variants, axis=0).astype(np.float32)
    aug_labels = np.full(len(aug), fall_id, dtype=np.int64)
    exp_windows = np.concatenate([windows, aug], axis=0).astype(np.float32)
    exp_labels = np.concatenate([labels, aug_labels], axis=0).astype(np.int64)
    print(f"S9 temporal-speed augment | mode={S9_AUG_MODE} fall_windows={n_fall} "
          f"augmented={len(aug)} total={len(exp_windows)}", flush=True)
    return exp_windows, exp_labels, summary


base.load_external_labeled_windows = patched_load


if __name__ == "__main__":
    r2.main()
