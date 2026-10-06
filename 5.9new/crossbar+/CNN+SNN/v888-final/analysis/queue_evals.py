"""Sequential evaluation queue: pose-off ablations + extended re-scans."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PYTHON = Path(r"D:\software\anaconde\envs\my_yizu_3.10\python.exe")
HERE = Path(__file__).resolve().parent
S30 = HERE.parent
MODELS = S30 / "models"
REPORTS = S30 / "reports"
EVAL_AT = HERE / "eval_at.py"
STRICT = HERE / "strict_eval_s30.py"
V9_MODEL = (
    Path(r"E:\rray\12.18-2\5.9new\crossbar+\CNN+SNN\v9\models")
    / "automatic_fall_event_detector_cnn_snn_v9_hnweight100.pth"
)

EVAL_AT_JOBS = [
    (MODELS / "automatic_fall_event_detector_c1_s32_posefull_floor020.pth", "c1_s32_at070_poseOFF", "0.70", True),
    (MODELS / "automatic_fall_event_detector_c1_s32_posefull_floor020.pth", "c1_s32_at080_poseOFF", "0.80", True),
    (MODELS / "automatic_fall_event_detector_c1_s33_posefull_floor020_lam020_aux030.pth", "c1_s33_at080_poseOFF", "0.80", True),
    (MODELS / "automatic_fall_event_detector_c1_s33_posefull_floor020_lam020_aux030.pth", "c1_s33_at060_poseON", "0.60", False),
]

RESCAN_JOBS = [
    (MODELS / "automatic_fall_event_detector_c1_s30_pose_floor020.pth", "c1_s30_pose_floor020_scan90"),
    (MODELS / "automatic_fall_event_detector_c1_s30_pose.pth", "c1_s30_pose_scan90"),
    (V9_MODEL, "v9_hnweight100_scan90"),
]


def main() -> None:
    for model, tag, thr, pose_off in EVAL_AT_JOBS:
        out_json = REPORTS / f"{tag}.json"
        if out_json.exists():
            print(f"skip {tag}", flush=True)
            continue
        cmd = [str(PYTHON), str(EVAL_AT), str(model), tag, thr]
        if pose_off:
            cmd.append("--pose-off")
        print(f"RUN {tag}", flush=True)
        with (REPORTS / f"{tag}.log").open("w", encoding="utf-8") as log:
            subprocess.run(cmd, cwd=str(S30), stdout=log, stderr=subprocess.STDOUT)
    for model, tag in RESCAN_JOBS:
        out_json = REPORTS / f"{tag}_subject4_metrics.json"
        if out_json.exists():
            print(f"skip {tag}", flush=True)
            continue
        print(f"RUN {tag}", flush=True)
        with (REPORTS / f"{tag}.log").open("w", encoding="utf-8") as log:
            subprocess.run([str(PYTHON), str(STRICT), str(model), tag], cwd=str(S30),
                           stdout=log, stderr=subprocess.STDOUT)
    print("QUEUE DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
