"""S10 data prep: UP-Fall subject11 zips -> train-only mp4 dataset.

Reads every Subject11ActivityXTrialYCamera1.zip, decodes PNG frames in memory
(no intermediate PNG dump), re-encodes one mp4 per trial, and writes them to
external_upfall\\train\\{fall,adl}\\ so the existing external labeled-window
loader can consume them directly (label from the "fall" name pattern).

UP-Fall activity ids: 1-5 = falls, 6-11 = ADL.
This data is TRAIN-ONLY: it must never be added to val/test inputs.
"""

from __future__ import annotations

import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

SRC = Path(r"E:\rray\external test\external test\subject11")
OUT = Path(
    r"E:\rray\12.18-2\5.9new\crossbar+\CNN+SNN\v8-0729-fusion-gate-r2"
    r"\external_upfall\train"
)

FALL_ACTIVITIES = {1, 2, 3, 4, 5}
NAME_RE = re.compile(r"Subject(\d+)Activity(\d+)Trial(\d+)Camera(\d+)")
TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2})T(\d{2})_(\d{2})_(\d{2})\.(\d+)")


def parse_ts(name: str) -> float:
    m = TS_RE.search(name)
    d, hh, mm, ss, frac = m.groups()
    dt = datetime.strptime(f"{d} {hh}:{mm}:{ss}", "%Y-%m-%d %H:%M:%S")
    return dt.timestamp() + float(f"0.{frac}")


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    zips = sorted(SRC.glob("*.zip"))
    assert zips, f"no zips under {SRC}"
    log(f"zips={len(zips)} out={OUT}")
    total = {"fall": [0, 0], "adl": [0, 0]}  # videos, frames
    for zp in zips:
        m = NAME_RE.match(zp.stem)
        sub, act, trial, cam = (int(g) for g in m.groups())
        label = "fall" if act in FALL_ACTIVITIES else "adl"
        out_dir = OUT / label
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{label}_upfall_s{sub:02d}_a{act:02d}_t{trial}.mp4"
        with zipfile.ZipFile(zp) as zf:
            entries = [n for n in zf.namelist() if n.lower().endswith(".png")]
            entries.sort(key=parse_ts)
            ts = [parse_ts(n) for n in entries]
            diffs = np.diff(ts)
            fps = 1.0 / float(np.median(diffs)) if len(diffs) else 18.0
            fps = float(np.clip(round(fps, 2), 5.0, 60.0))
            writer = None
            n = 0
            for name in entries:
                buf = np.frombuffer(zf.read(name), dtype=np.uint8)
                frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
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
        dur = n / fps if fps > 0 else 0.0
        win = max(0, (n - 16) // 16 + 1)
        total[label][0] += 1
        total[label][1] += n
        log(f"{zp.stem}: {label} frames={n} fps={fps:.2f} dur={dur:.1f}s "
            f"est_windows={win} -> {out_path.name}")
    log(f"done. fall={total['fall'][0]} videos/{total['fall'][1]} frames | "
        f"adl={total['adl'][0]} videos/{total['adl'][1]} frames")


if __name__ == "__main__":
    sys.exit(main())
