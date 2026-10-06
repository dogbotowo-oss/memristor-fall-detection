"""Extract tiled sample frames for visual domain comparison (read-only on data)."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"E:\rray\12.18-2")
GMDCSA_ROOT = ROOT / "测试集" / "13354453" / "ekramalam" / "GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-v2.1" / "ekramalam-GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-5abac76"
SUBJECT4_DIR = ROOT / "gmdcsa_subject4_test" / "Subject 4"
OUT_DIR = Path(__file__).resolve().parent / "subject4_domain_frames"
TILE_W, TILE_H = 320, 180


def tile_video(video_path: Path, out_path: Path, n_frames: int = 4) -> None:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(total - 1, 0), n_frames).astype(int)
    tiles = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
        frame = cv2.resize(frame, (TILE_W, TILE_H))
        cv2.putText(frame, f"f{idx}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        tiles.append(frame)
    cap.release()
    cv2.imwrite(str(out_path), np.hstack(tiles))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for name in ["06", "07", "10", "17"]:
        jobs.append((SUBJECT4_DIR / "ADL" / f"{name}.mp4", f"S4_ADL_{name}_FP.jpg"))
    for name in ["01", "02", "13", "04"]:
        jobs.append((SUBJECT4_DIR / "Fall" / f"{name}.mp4", f"S4_Fall_{name}.jpg"))
    for name in ["09", "13"]:
        jobs.append((SUBJECT4_DIR / "ADL" / f"{name}.mp4", f"S4_ADL_{name}_TN.jpg"))
    s1_adl = sorted((GMDCSA_ROOT / "Subject 1" / "ADL").glob("*.mp4"))[:2]
    s2_adl = sorted((GMDCSA_ROOT / "Subject 2" / "ADL").glob("*.mp4"))[:2]
    for i, p in enumerate(s1_adl):
        jobs.append((p, f"S1_ADL_ref{i}.jpg"))
    for i, p in enumerate(s2_adl):
        jobs.append((p, f"S2_ADL_ref{i}.jpg"))
    for src, out in jobs:
        if not src.exists():
            print(f"missing: {src}", flush=True)
            continue
        tile_video(src, OUT_DIR / out)
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
