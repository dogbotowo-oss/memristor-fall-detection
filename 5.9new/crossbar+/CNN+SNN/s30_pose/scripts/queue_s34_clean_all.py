"""Sequential queue: s34 on data1_clean (retention-window trimmed).

Main experiment: 5 seeds full s34 (c1_s34_clean_s*).
Follow-up: 5 seeds no-SNN paired ablation (c1_s34_clean_nosnn_s*).
Raw data1 runs (c1_s34_data1_*) stay as the retention-failure stress test.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PYTHON = Path(r"D:\software\anaconde\envs\my_yizu_3.10\python.exe")
S30 = Path(r"E:\rray\12.18-2\5.9new\crossbar+\CNN+SNN\s30_pose")
RUN_S30 = S30 / "scripts" / "run_s30.py"
MODELS = S30 / "models"
REPORTS = S30 / "reports"
CLEAN = Path(r"E:\rray\12.18-2\data1_clean\data")

SEEDS = (1, 3, 5, 11, 42)
JOBS = [
    ["--floor", "0.2", "--preserve", "0.3", "--preserve-margin", "0.5",
     "--seed", str(s), "--data-dir", str(CLEAN), "--tag", f"c1_s34_clean_s{s}"]
    for s in SEEDS
] + [
    ["--no-snn", "--seed", str(s), "--data-dir", str(CLEAN),
     "--tag", f"c1_s34_clean_nosnn_s{s}"]
    for s in SEEDS
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
