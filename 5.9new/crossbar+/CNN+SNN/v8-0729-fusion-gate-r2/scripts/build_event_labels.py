"""Build event-localized fall labels for Zenodo and UP-Fall TRAIN fall videos.

Zenodo (URFD) fall mp4s carry video-level labels: every window is labeled fall,
although a 26-69s sequence contains only a 1-17s fall event (and long stretches
of empty room). This contradicts the GMDCSA frame-level convention (fall event
= fall, everything else = normal) and blurs the ADL/fall decision boundary.

UP-Fall fall trials (~9s) have the same problem at a smaller scale: the fall
occupies only ~1-2s of the scripted trial. UP-Fall events are localized from
the inter-frame motion-energy peak (the fall is the dominant motion burst).

Zenodo events are localized per sequence from the URFD skeleton.txt 3D
head/spine height trajectory (rapid sustained drop), and frame-index intervals
are written to external_event_labels.json. The training loader then treats
overlapping windows as fall, other in-range windows as normal, and skips
windows outside the tracked-person range (empty room).

TRAIN videos only: val/test labeling is never touched by this file.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

EXP_ROOT = Path(__file__).resolve().parents[1]
FOLDER_SPLIT = Path(r"E:\rray\12.18-2\zenodo_falldb_folder_split") / "train" / "Fall"
UPFALL_TRAIN = EXP_ROOT / "external_upfall" / "train" / "fall"
OUT_PATH = EXP_ROOT / "external_event_labels.json"

FPS = 15.0
PRE_MARGIN_S = 0.5
POST_MARGIN_S = 0.3
MIN_DROP_MM = 200.0
MIN_VALID_FRAMES = 30


def load_skeleton(seq_dir: Path) -> np.ndarray | None:
    skel = seq_dir / "skeleton.txt"
    if not skel.exists():
        return None
    rows = []
    with skel.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh, delimiter=";")
        next(reader)
        for row in reader:
            if len(row) < 21:
                continue
            try:
                rows.append([int(row[0])] + [float(x) if x else 0.0 for x in row[1:29]])
            except ValueError:
                continue
    return np.array(rows) if rows else None


def detect_fall_interval(arr: np.ndarray) -> dict | None:
    """Return fall/valid frame intervals from the head/spine height trajectory."""
    frames = arr[:, 0].astype(int)
    head_y, head_conf = arr[:, 18], arr[:, 20]
    spine_y, spine_conf = arr[:, 10], arr[:, 12]
    y = np.where(head_conf > 0.5, head_y, np.where(spine_conf > 0.5, spine_y, np.nan))
    valid = ~np.isnan(y)
    if int(valid.sum()) < MIN_VALID_FRAMES:
        return None
    idx = np.arange(len(y))
    y = np.interp(idx, idx[valid], y[valid])
    kernel = 7
    y_smooth = np.convolve(y, np.ones(kernel) / kernel, mode="same")
    lo, hi = np.percentile(y_smooth, 5), np.percentile(y_smooth, 95)
    if hi - lo < MIN_DROP_MM:
        return None
    y_norm = (y_smooth - lo) / (hi - lo)

    below = y_norm < 0.40
    best_len, best_end = 0, -1
    i = 0
    while i < len(below):
        if below[i]:
            j = i
            while j < len(below) and below[j]:
                j += 1
            if j - i > best_len:
                best_len, best_end = j - i, j
            i = j
        else:
            i += 1
    if best_end < 0:
        return None
    lie_start = best_end - best_len
    above = np.where(y_norm[: lie_start + 1] > 0.70)[0]
    if len(above) == 0:
        return None
    descent_start, descent_end = int(above[-1]), lie_start

    pre = int(round(PRE_MARGIN_S * FPS))
    post = int(round(POST_MARGIN_S * FPS))
    fall_start = max(int(frames[0]), int(frames[descent_start]) - pre)
    fall_end = min(int(frames[-1]), int(frames[descent_end]) + post)
    return {
        "fall_start": fall_start,
        "fall_end": fall_end,
        "valid_start": int(frames[0]),
        "valid_end": int(frames[-1]),
        "drop_mm": round(float(hi - lo), 1),
        "fall_seconds": round((fall_end - fall_start) / FPS, 2),
    }


def detect_upfall_interval(video_path: Path) -> dict | None:
    """Localize the fall in a UP-Fall trial from the motion-energy peak."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    prev = None
    energy = []
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if prev is not None:
            energy.append(float(np.abs(gray - prev).mean()))
        prev = gray
    cap.release()
    if len(energy) < 30 or fps <= 0:
        return None
    e = np.asarray(energy, dtype=np.float64)
    kernel = max(3, int(fps // 4))
    e_smooth = np.convolve(e, np.ones(kernel) / kernel, mode="same")
    if float(e_smooth.max()) <= 0:
        return None
    above = np.where(e_smooth > 0.35 * e_smooth.max())[0]
    pre = int(round(PRE_MARGIN_S * fps))
    post = int(round(POST_MARGIN_S * fps))
    frame_count = len(energy) + 1
    fall_start = max(0, int(above[0]) - pre)
    fall_end = min(frame_count - 1, int(above[-1]) + post)
    return {
        "fall_start": fall_start,
        "fall_end": fall_end,
        "valid_start": 0,
        "valid_end": frame_count - 1,
        "drop_mm": None,
        "fall_seconds": round((fall_end - fall_start) / fps, 2),
        "source": "motion_energy",
    }


def main() -> int:
    if not FOLDER_SPLIT.exists():
        print(f"missing skeleton source: {FOLDER_SPLIT}", flush=True)
        return 1
    videos = {}
    failures = []
    for seq_dir in sorted(FOLDER_SPLIT.iterdir()):
        if not seq_dir.is_dir():
            continue
        arr = load_skeleton(seq_dir)
        result = detect_fall_interval(arr) if arr is not None else None
        if result is None:
            failures.append(seq_dir.name)
            continue
        result["source"] = "skeleton_height"
        videos[f"{seq_dir.name}.mp4"] = result
    upfall_count = 0
    if UPFALL_TRAIN.exists():
        for video_path in sorted(UPFALL_TRAIN.glob("*.mp4")):
            result = detect_upfall_interval(video_path)
            if result is None:
                failures.append(video_path.name)
                continue
            videos[video_path.name] = result
            upfall_count += 1
    payload = {
        "meta": {
            "source": f"{FOLDER_SPLIT} + {UPFALL_TRAIN}",
            "fps": FPS,
            "pre_margin_s": PRE_MARGIN_S,
            "post_margin_s": POST_MARGIN_S,
            "scope": "Zenodo URFD + UP-Fall TRAIN fall videos only; val/test untouched",
            "num_videos": len(videos),
            "failures": failures,
        },
        "videos": videos,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT_PATH} videos={len(videos)} (zenodo={len(videos) - upfall_count}, upfall={upfall_count}) failures={len(failures)} {failures}", flush=True)
    for name, info in videos.items():
        print(f"  {name}: fall=[{info['fall_start']},{info['fall_end']}] "
              f"valid=[{info['valid_start']},{info['valid_end']}] drop={info['drop_mm']}mm", flush=True)
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
