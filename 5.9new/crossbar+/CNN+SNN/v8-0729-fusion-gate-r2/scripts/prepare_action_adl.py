"""S22 data prep: ADL-adddataset PNG frame dirs -> action-tagged mp4s.

Reads every SubjectNActivityMTrialKCameraK directory under ADL-adddataset
{train,val}, re-encodes one mp4 per dir, named adlact_a<MM>_upf_s<NN>_t<T>.mp4
so the runner's "--external-action-adl-dir" loader can parse the UP-Fall
activity id from "_a<NN>_". Train split is TRAIN-ONLY; the val split
(UP-Fall subject 3) is a held-out probe for confusable-ADL scores and must
never be added to training inputs.
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

import cv2

SRC_ROOT = Path(r"E:\rray\12.18-2\ADL-adddataset")
OUT_ROOT = Path(
    r"E:\rray\12.18-2\5.9new\crossbar+\CNN+SNN\v8-0729-fusion-gate-r2"
    r"\external_action_adl"
)

NAME_RE = re.compile(r"Subject(\d+)Activity(\d+)Trial(\d+)Camera(\d+)")
TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2})T(\d{2})_(\d{2})_(\d{2})\.(\d+)")


def parse_ts(name: str) -> float:
    m = TS_RE.search(name)
    if m is None:
        return 0.0
    d, hh, mm, ss, frac = m.groups()
    dt = datetime.strptime(f"{d} {hh}:{mm}:{ss}", "%Y-%m-%d %H:%M:%S")
    return dt.timestamp() + float(f"0.{frac.rstrip('0') or '0'}")


def log(msg: str) -> None:
    print(msg, flush=True)


def encode_split(split: str, out_name: str) -> None:
    src = SRC_ROOT / split
    out_dir = OUT_ROOT / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    trial_dirs = sorted(
        p for p in src.rglob("*") if p.is_dir() and NAME_RE.match(p.name)
    )
    assert trial_dirs, f"no trial dirs under {src}"
    total_videos = 0
    total_frames = 0
    for trial_dir in trial_dirs:
        sub, act, trial, cam = (int(g) for g in NAME_RE.match(trial_dir.name).groups())
        frames = sorted(
            (p for p in trial_dir.glob("*.png")),
            key=lambda p: parse_ts(p.name),
        )
        if not frames:
            log(f"skip empty dir: {trial_dir}")
            continue
        out_path = out_dir / f"adlact_a{act:02d}_upf_s{sub:02d}_t{trial}_cam{cam}.mp4"
        ts = [parse_ts(p.name) for p in frames]
        diffs = [b - a for a, b in zip(ts, ts[1:]) if b > a]
        fps = 1.0 / float(sorted(diffs)[len(diffs) // 2]) if diffs else 18.0
        fps = float(min(max(round(fps, 2), 5.0), 60.0))
        writer = None
        n = 0
        for frame_path in frames:
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                )
            writer.write(frame)
            n += 1
        if writer is not None:
            writer.release()
        total_videos += 1
        total_frames += n
        log(f"{trial_dir.name}: frames={n} fps={fps:.2f} -> {out_path.name}")
    log(f"split={split}: videos={total_videos} frames={total_frames} -> {out_dir}")


def main() -> None:
    encode_split("train", "train")
    encode_split("val", "val_probe")


if __name__ == "__main__":
    sys.exit(main())
