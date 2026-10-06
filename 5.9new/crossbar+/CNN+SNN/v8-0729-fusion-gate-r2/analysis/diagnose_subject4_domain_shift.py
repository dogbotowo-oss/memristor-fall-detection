"""Read-only domain-shift diagnostic: Subject1-4 + Zenodo video properties.

Compares fps / resolution / duration / brightness / inter-frame motion between
the GMDCSA subjects used for train (S1,S2), val (S3) and test (S4), plus the
Zenodo split, to identify why S10 loses fall/ADL score separation on Subject4.
Does not write anything except the printed JSON summary.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(r"E:\rray\12.18-2")
GMDCSA_ROOT = ROOT / "测试集" / "13354453" / "ekramalam" / "GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-v2.1" / "ekramalam-GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-5abac76"
SUBJECT4_DIR = ROOT / "gmdcsa_subject4_test" / "Subject 4"
ZENODO_TRAIN = ROOT / "zenodo_falldb_video_split" / "train"
ZENODO_VAL = ROOT / "zenodo_falldb_video_split" / "val"

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
MAX_SAMPLE_FRAMES = 60


def iter_videos(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in VIDEO_EXTS)


def video_stats(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"video": str(video_path), "error": "cannot open"}
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    brightness, motion = [], []
    prev_gray = None
    step = max(frame_count // MAX_SAMPLE_FRAMES, 1)
    idx, sampled = 0, 0
    while sampled < MAX_SAMPLE_FRAMES:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            brightness.append(float(gray.mean()))
            if prev_gray is not None:
                motion.append(float(np.abs(gray - prev_gray).mean()))
            prev_gray = gray
            sampled += 1
        idx += 1
    cap.release()
    duration_s = frame_count / fps if fps > 0 else 0.0
    return {
        "video": str(video_path), "fps": fps, "width": width, "height": height,
        "frames": frame_count, "duration_s": duration_s,
        "brightness_mean": float(np.mean(brightness)) if brightness else None,
        "motion_mean": float(np.mean(motion)) if motion else None,
    }


def summarize_group(name: str, videos: list[Path]) -> dict:
    stats = [video_stats(v) for v in videos]
    ok = [s for s in stats if "error" not in s]
    if not ok:
        return {"group": name, "num_videos": len(videos), "error": "no readable videos"}

    def agg(key: str) -> dict:
        vals = [s[key] for s in ok if s.get(key) is not None]
        arr = np.asarray(vals, dtype=np.float64)
        return {"mean": float(arr.mean()), "std": float(arr.std()),
                "min": float(arr.min()), "max": float(arr.max())}

    resolutions = {}
    for s in ok:
        res = f'{s["width"]}x{s["height"]}'
        resolutions[res] = resolutions.get(res, 0) + 1
    return {
        "group": name, "num_videos": len(ok),
        "fps": agg("fps"), "duration_s": agg("duration_s"),
        "brightness_mean": agg("brightness_mean"), "motion_mean": agg("motion_mean"),
        "resolutions": resolutions,
    }


def main() -> None:
    groups = []
    for subject in ["Subject 1", "Subject 2", "Subject 3"]:
        base = GMDCSA_ROOT / subject
        for cls in ["ADL", "Fall"]:
            groups.append((f"GMDCSA {subject}/{cls} (train/val)", iter_videos(base / cls)))
    for cls in ["ADL", "Fall"]:
        groups.append((f"GMDCSA Subject4/{cls} (TEST)", iter_videos(SUBJECT4_DIR / cls)))
    for split_name, split_dir in [("train", ZENODO_TRAIN), ("val", ZENODO_VAL)]:
        for cls_dir in sorted(split_dir.iterdir()) if split_dir.exists() else []:
            if cls_dir.is_dir():
                groups.append((f"Zenodo {split_name}/{cls_dir.name}", iter_videos(cls_dir)))

    summary = [summarize_group(name, videos) for name, videos in groups]
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
