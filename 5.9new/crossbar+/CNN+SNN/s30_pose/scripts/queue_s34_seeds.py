"""Sequential queue: s34 (fall-preserve) x 3 seeds + s32 x 2 extra seeds."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PYTHON = Path(r"D:\software\anaconde\envs\my_yizu_3.10\python.exe")
S30 = Path(r"E:\rray\12.18-2\5.9new\crossbar+\CNN+SNN\s30_pose")
RUN_S30 = S30 / "scripts" / "run_s30.py"
MODELS = S30 / "models"
REPORTS = S30 / "reports"

JOBS = [
    ["--floor", "0.2", "--preserve", "0.3", "--preserve-margin", "0.5", "--seed", "42", "--tag", "c1_s34_preserve_s42"],
    ["--floor", "0.2", "--preserve", "0.3", "--preserve-margin", "0.5", "--seed", "1", "--tag", "c1_s34_preserve_s1"],
    ["--floor", "0.2", "--preserve", "0.3", "--preserve-margin", "0.5", "--seed", "7", "--tag", "c1_s34_preserve_s7"],
    ["--floor", "0.2", "--seed", "1", "--tag", "c1_s32_posefull_floor020_s1"],
    ["--floor", "0.2", "--seed", "7", "--tag", "c1_s32_posefull_floor020_s7"],
]


def main() -> None:
    for args in JOBS:
        tag = args[args.index("--tag") + 1]
        model_path = MODELS / f"automatic_fall_event_detector_{tag}.pth"
        done_marker = REPORTS / f"{tag}_subject4_metrics.json"
        if model_path.exists() and done_marker.exists():
            print(f"skip {tag}", flush=True)
            continue
        print(f"RUN {tag}", flush=True)
        with (REPORTS / f"queue_{tag}.log").open("w", encoding="utf-8") as log:
            subprocess.run([str(PYTHON), str(RUN_S30), *args], cwd=r"E:\rray\12.18-2",
                           stdout=log, stderr=subprocess.STDOUT)
        print(f"DONE {tag}", flush=True)
    print("QUEUE COMPLETE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
