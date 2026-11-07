# -*- coding: utf-8 -*-
"""Build label tables for the GMDCSA-24 folder layout.

The dataset ships as  <root>/Subject <n>/{ADL,Fall}/<video>.  Folder names are the only
label source, so they are used to build

  * a video-level table  (video_name, relative_path, subject, category, label_id)
  * optionally a frame-level table for one video, which is what
    ``automatic_fall_detector.py --labels-csv`` expects for single-video training.

Usage:
    python tools/build_video_labels.py --root "D:/GMDCSA24/Subject 1/.." --out data/video_labels.csv
    python tools/build_video_labels.py --root ... --frames-video "Subject 1/Fall/01.mp4" \
        --frames-out data/fall_video_frame_labels.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".m4v", ".mkv"}
LABEL_BY_FOLDER = {"fall": 1, "adl": 0}


def iter_videos(root: Path):
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        parts = path.relative_to(root).parts
        if len(parts) < 3:
            continue
        subject, category = parts[0], parts[1]
        label = LABEL_BY_FOLDER.get(category.lower())
        if label is None:
            continue
        yield path, subject, category, label


def frame_count(video: Path) -> int:
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True, help="dataset root containing 'Subject N' folders")
    ap.add_argument("--out", type=Path, default=Path("data/video_labels.csv"))
    ap.add_argument("--frames-video", type=str, default=None,
                    help="relative path of one video, e.g. 'Subject 1/Fall/01.mp4'")
    ap.add_argument("--frames-out", type=Path, default=Path("data/fall_video_frame_labels.csv"))
    args = ap.parse_args()

    rows = []
    for video, subject, category, label in iter_videos(args.root):
        rows.append({
            "video_name": video.name,
            "relative_path": video.relative_to(args.root).as_posix(),
            "subject": subject,
            "category": category,
            "label_id": label,
        })
    if not rows:
        raise SystemExit(f"no videos found under {args.root}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out} with {len(rows)} videos "
          f"(fall={sum(r['label_id'] for r in rows)}, adl={sum(1 - r['label_id'] for r in rows)})")

    if args.frames_video:
        target = args.root / args.frames_video
        if not target.exists():
            raise SystemExit(f"video not found: {target}")
        label = LABEL_BY_FOLDER[target.parent.name.lower()]
        n = frame_count(target)
        args.frames_out.parent.mkdir(parents=True, exist_ok=True)
        with args.frames_out.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["video_name", "frame_index", "label_id"])
            writer.writerows((target.name, i, label) for i in range(n))
        print(f"wrote {args.frames_out} with {n} frames for {target.name} (label_id={label})")


if __name__ == "__main__":
    main()
