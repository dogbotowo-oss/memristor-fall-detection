"""Build MediaPipe pose cache for all GMDCSA videos (Subject 1-4).
Per video: (num_frames, 33, 3) float32 array of x, y, visibility, saved as
.npy mirroring the source layout under pose_cache/. Zenodo videos are NOT
cached (probe showed 3-19% detection there - pose route is GMDCSA-only).
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"E:\rray\12.18-2")
GMDCSA = next(
    p for p in (ROOT / "测试集" / "13354453" / "ekramalam").glob("GMDCSA24*/ekramalam-GMDCSA24*5abac76")
)
SUBJECT4 = ROOT / "gmdcsa_subject4_test" / "Subject 4"
CACHE = ROOT / "pose_cache"

VIDEO_ROOTS = [GMDCSA / f"Subject {i}" for i in (1, 2, 3)] + [SUBJECT4]


def cache_path_for(video: Path) -> Path:
    parts = video.parts
    subj = next((p for p in parts if p.lower().startswith("subject ")), "misc")
    return CACHE / subj / video.parent.name / (video.stem + ".npy")


def process(video: Path, pose) -> str:
    out_path = cache_path_for(video)
    if out_path.exists():
        return "skip"
    cap = cv2.VideoCapture(str(video))
    rows = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if out.pose_landmarks is not None:
            rows.append([[lm.x, lm.y, lm.visibility] for lm in out.pose_landmarks.landmark])
        else:
            rows.append(np.zeros((33, 3), dtype=np.float32))
    cap.release()
    arr = np.asarray(rows, dtype=np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, arr)
    return f"ok frames={len(arr)}"


def main() -> None:
    import mediapipe as mp
    videos = []
    for root in VIDEO_ROOTS:
        if root.exists():
            videos.extend(sorted(root.rglob("*.mp4")))
    print(f"videos={len(videos)}", flush=True)
    t0 = time.time()
    with mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1,
                                min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for i, v in enumerate(videos, 1):
            try:
                status = process(v, pose)
            except Exception as exc:
                status = f"FAIL {exc}"
            print(f"[{i}/{len(videos)}] {v.parent.parent.name}/{v.parent.name}/{v.name} {status} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
