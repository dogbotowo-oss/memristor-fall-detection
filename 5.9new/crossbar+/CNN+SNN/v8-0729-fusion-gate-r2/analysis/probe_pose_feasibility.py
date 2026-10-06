"""Pose feasibility probe: can MediaPipe extract usable 33-point skeletons
from the project's videos (low-res, far-view)? Samples GMDCSA + Zenodo
videos, reports per-video frame detection rate and landmark visibility.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"E:\rray\12.18-2")
GMDCSA = next(
    p for p in (ROOT / "测试集" / "13354453" / "ekramalam").glob("GMDCSA24*/ekramalam-GMDCSA24*5abac76")
)
VIDEOS = [
    GMDCSA / "Subject 1" / "Fall" / "01.mp4",
    GMDCSA / "Subject 1" / "ADL" / "01.mp4",
    GMDCSA / "Subject 3" / "Fall" / "01.mp4",
    GMDCSA / "Subject 3" / "ADL" / "01.mp4",
    ROOT / "zenodo_falldb_video_split" / "val" / "Fall" / "01.mp4",
    ROOT / "zenodo_falldb_video_split" / "val" / "ADL" / "person2b_14.mp4",
]

import mediapipe as mp

KEY_JOINTS = [0, 11, 12, 23, 24, 25, 26, 27, 28]  # nose, shoulders, hips, knees, ankles


def probe(video: Path, pose) -> None:
    cap = cv2.VideoCapture(str(video))
    n = det = 0
    vis_sum = 0.0
    res = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if res is None:
            res = (frame.shape[1], frame.shape[0])
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out = pose.process(rgb)
        if out.pose_landmarks is not None:
            lm = out.pose_landmarks.landmark
            vis = float(np.mean([lm[j].visibility for j in KEY_JOINTS]))
            if vis > 0.5:
                det += 1
            vis_sum += vis
    cap.release()
    rate = det / max(n, 1)
    mean_vis = vis_sum / max(n, 1)
    tag = "OK " if rate > 0.7 else ("WEAK" if rate > 0.4 else "BAD ")
    print(f"[{tag}] {video.parent.parent.name}/{video.parent.name}/{video.name} "
          f"res={res} frames={n} detect(vis>0.5)={rate:.2%} mean_vis={mean_vis:.3f}", flush=True)


def main() -> None:
    with mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1,
                                min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for v in VIDEOS:
            if not v.exists():
                print(f"[MISS] {v}", flush=True)
                continue
            probe(v, pose)


if __name__ == "__main__":
    main()
