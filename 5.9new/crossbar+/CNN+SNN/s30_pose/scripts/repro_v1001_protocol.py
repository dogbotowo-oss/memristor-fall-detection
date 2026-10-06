"""Reproduce the v1001 main-table protocol and apply it to the bw models.

v1001 protocol (from v1001/reports JSONs):
  "F1-max threshold on validation (tie -> higher threshold, cap 0.70)"

Step 1: verify on the v1000 full models that some candidate validation set
reproduces the user's main table (thr 0.50/0.45/0.60/0.70, acc
96.88/91.67/81.40/75.68). Candidates:
  A) Zenodo val only
  B) Zenodo val + one in-fold GMDCSA subject (S3 for folds 1,2,4; S4 for fold 3)
Step 2: apply the winning protocol to the bw checkpoints.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(r"F:\12.18-2")
S30 = ROOT / "5.9new" / "crossbar+" / "CNN+SNN" / "s30_pose"
EVAL_PATH = S30 / "analysis" / "strict_eval_s30.py"
MODELS = S30 / "models"
DATA1 = ROOT / "data1"
POSE_CACHE = ROOT / "pose_cache"
ZENODO_VAL = ROOT / "zenodo_falldb_video_split" / "val"
GMDCSA = (
    ROOT / "测试集" / "13354453" / "ekramalam"
    / "GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-v2.1"
    / "ekramalam-GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-5abac76"
)
SUBJECT_DIRS = {i: GMDCSA / f"Subject {i}" for i in (1, 2, 3)}
SUBJECT_DIRS[4] = ROOT / "gmdcsa_subject4_test" / "Subject 4"
CAL_SUBJECT = {1: 3, 2: 3, 3: 4, 4: 3}
CAP = 0.70


def load_eval_module():
    spec = importlib.util.spec_from_file_location("strict_eval_repro", EVAL_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    mod.ROOT = ROOT
    mod.S30_DIR = S30
    mod.CODE_PATH = S30 / "code" / "automatic_fall_detector.py"
    mod.DATA_DIR = DATA1
    mod.POSE_CACHE_DIR = POSE_CACHE
    return mod


def f1_max_threshold(scan_rows, cap=CAP):
    best = None
    for r in scan_rows:
        thr = float(r["threshold"])
        if thr > cap + 1e-9:
            continue
        rec = float(r["fall_recall"])
        prec = float(r.get("fall_precision", r.get("precision", 0.0)))
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        key = (f1, thr)  # tie -> higher threshold
        if best is None or key > best[0]:
            best = (key, thr)
    return best[1]


def run_for_model(sev, model_path: Path, test_subject: int, val_videos, device):
    mod = sev.load_module()
    model, config = mod.load_model_checkpoint(model_path, device)
    scan_rows, _ = sev.scan_thresholds(mod, model, config, val_videos, device)
    thr = f1_max_threshold(scan_rows)
    test_videos = sorted(mod.list_video_files(SUBJECT_DIRS[test_subject]))
    _, summary = sev.evaluate_at_threshold(mod, model, config, test_videos, thr, device)
    return thr, summary


def main() -> None:
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sev = load_eval_module()
    mod = sev.load_module()
    zenodo = sorted(mod.list_video_files(ZENODO_VAL))

    expected = {1: (0.50, 96.88), 2: (0.45, 91.67), 3: (0.60, 81.40), 4: (0.70, 75.68)}

    for val_name in ("A_zenodo", "B_zenodo+gmdcsa"):
        print(f"=== validation set {val_name} ===", flush=True)
        for ts in (1, 2, 3, 4):
            val_videos = list(zenodo)
            if val_name.startswith("B"):
                val_videos += sorted(mod.list_video_files(SUBJECT_DIRS[CAL_SUBJECT[ts]]))
            model_path = MODELS / f"automatic_fall_event_detector_c1_s34_data1_loso_ts{ts}_s11.pth"
            thr, summary = run_for_model(sev, model_path, ts, val_videos, device)
            exp_thr, exp_acc = expected[ts]
            mark = "MATCH" if abs(summary["accuracy"] * 100 - exp_acc) < 0.01 else "diff"
            print(f"full ts{ts}: thr={thr:.2f} (expect {exp_thr}) acc={summary['accuracy']*100:.2f} "
                  f"(expect {exp_acc}) {mark}", flush=True)


if __name__ == "__main__":
    main()
