"""v1002 bw_s: binary weights + learnable per-channel scale, seed 11.

Conditions share the s34 recipe; only the SNN-branch weight mapping differs:
  full : full-precision SNN weights
  bw   : binary conductance states (--snn-binary-weights)
  bw4  : four measured compliance states (--snn-weight-levels 0.0094,0.021,0.104,1.0)

Evaluation uses the locked main-table protocol (reproduced 2026-09-25):
  validation = Zenodo val + one in-fold GMDCSA subject (S3 for folds 1,2,4;
  S4 for fold 3), F1-max threshold (tie -> higher, cap 0.70), frozen one-shot
  evaluation on the held-out subject.

All code/models/reports live under v1002/; s30_pose originals are untouched.
seed-11 runs already exist elsewhere (s30_pose/reports and v1002/reports);
this queue only adds seeds 23 and 42.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"F:\12.18-2")
PYTHON = Path(r"F:\ANACONDA\envs\my_yizu_3.10\python.exe")
S30 = ROOT / "5.9new" / "crossbar+" / "CNN+SNN" / "s30_pose"
V1002 = S30 / "v1002"
CODE_PATH = V1002 / "code" / "automatic_fall_detector.py"
EVAL_PATH = V1002 / "analysis" / "strict_eval_s30.py"
MODELS = V1002 / "models"
REPORTS = V1002 / "reports"
IMAGE = V1002 / "image"
DATA1 = ROOT / "data1"
POSE_CACHE = ROOT / "pose_cache"
ZENODO_TRAIN = ROOT / "zenodo_falldb_video_split" / "train"
ZENODO_VAL = ROOT / "zenodo_falldb_video_split" / "val"
V1_INIT = ROOT / "5.9new" / "crossbar+" / "CNN+SNN" / "models" / "automatic_fall_event_detector_cnn_snn_v1.pth"
GMDCSA = (
    ROOT / "测试集" / "13354453" / "ekramalam"
    / "GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-v2.1"
    / "ekramalam-GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-5abac76"
)
SUBJECT_DIRS = {i: GMDCSA / f"Subject {i}" for i in (1, 2, 3)}
SUBJECT_DIRS[4] = ROOT / "gmdcsa_subject4_test" / "Subject 4"
CAL_SUBJECT = {1: 3, 2: 3, 3: 4, 4: 3}
CAP = 0.70
LEVELS = "0.0094,0.021,0.104,1.0"

CONDITIONS = {
    "bw_s": ["--snn-binary-weights", "--snn-learnable-scale"],
}
SEEDS = (11,)


def load_eval_module():
    spec = importlib.util.spec_from_file_location("strict_eval_v1002ms", EVAL_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    mod.DATA_DIR = DATA1
    mod.POSE_CACHE_DIR = POSE_CACHE
    return mod


def train(tag: str, test_subject: int, cond: str, seed: int) -> Path:
    model_path = MODELS / f"automatic_fall_event_detector_{tag}.pth"
    if model_path.exists():
        print(f"model exists, skip training: {model_path}", flush=True)
        return model_path
    train_subjects = [s for s in (1, 2, 3, 4) if s != test_subject]
    cmd = [
        str(PYTHON), str(CODE_PATH), "--mode", "train",
        "--skip-main-video-training", "--disable-auxiliary-data",
        "--external-labeled-dir",
        *[str(SUBJECT_DIRS[s]) for s in train_subjects],
        str(ZENODO_TRAIN),
        "--external-val-labeled-dir", str(ZENODO_VAL),
        "--data-dir", str(DATA1),
        "--init-model-path", str(V1_INIT),
        "--model-path", str(model_path),
        "--report-json", str(REPORTS / f"train_report_{tag}.json"),
        "--training-plot-path", str(IMAGE / f"training_curve_{tag}.png"),
        "--freeze-cnn-gru-epochs", "2",
        "--teacher-student-noise", "--student-noise-mode", "gaussian",
        "--use-device-dynamics",
        "--crossbar-readout-noise-scale", "0.25",
        "--hard-negative-normal-weight", "1.00",
        "--hard-negative-min-percentile", "0.88",
        "--hard-negative-score-gamma", "2.00",
        "--normal-positive-penalty", "0.00",
        "--external-labeled-stride", "16",
        "--adl-context-path-keyword", "zenodo_falldb_video_split/train/adl",
        "--adl-context-min-percentile", "0.95",
        "--adl-context-extra-copies", "1",
        "--image-size", "64", "--skip-visual-outputs",
        "--seed", str(seed),
        "--use-snn-temporal-branch", "--use-snn-fusion-gate",
        "--snn-fusion-gate-mode", "group_channel",
        "--learnable-snn-dynamics",
        "--snn-adl-gate-penalty", "0.06",
        "--snn-event-score-threshold", "0.55",
        "--snn-event-score-sharpness", "10.0",
        "--use-snn-pose-input",
        "--pose-cache-dir", str(POSE_CACHE),
        "--snn-fusion-gate-floor", "0.20",
        "--late-fusion-lambda", "0.0",
        "--snn-class-aux-weight", "0.0",
        "--snn-fall-preserve-weight", "0.30",
        "--snn-fall-preserve-margin", "0.50",
        *CONDITIONS[cond],
    ]
    with (REPORTS / f"train_{tag}.stdout.log").open("w", encoding="utf-8") as out, \
         (REPORTS / f"train_{tag}.stderr.log").open("w", encoding="utf-8") as err:
        subprocess.run(cmd, cwd=str(ROOT), stdout=out, stderr=err, check=True)
    return model_path


def f1_max_threshold(scan_rows, cap=CAP):
    best = None
    for r in scan_rows:
        thr = float(r["threshold"])
        if thr > cap + 1e-9:
            continue
        rec = float(r["fall_recall"])
        prec = float(r.get("precision", 0.0))
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        key = (f1, thr)
        if best is None or key > best[0]:
            best = (key, thr)
    return best[1]


def evaluate(sev, tag: str, test_subject: int, model_path: Path) -> dict:
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mod = sev.load_module()
    model, config = mod.load_model_checkpoint(model_path, device)
    config["pose_cache_dir"] = str(POSE_CACHE)

    val_videos = sorted(mod.list_video_files(ZENODO_VAL)) + \
        sorted(mod.list_video_files(SUBJECT_DIRS[CAL_SUBJECT[test_subject]]))
    scan_rows, _ = sev.scan_thresholds(mod, model, config, val_videos, device)
    best_thr = f1_max_threshold(scan_rows)
    val_summary = {"tag": tag, "best_threshold": float(best_thr),
                   "source": "Zenodo val + in-fold GMDCSA subject, F1-max cap 0.70"}
    (REPORTS / f"{tag}_v1001proto_val_summary.json").write_text(
        json.dumps(val_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    test_videos = sorted(mod.list_video_files(SUBJECT_DIRS[test_subject]))
    rows, summary = sev.evaluate_at_threshold(mod, model, config, test_videos, best_thr, device)
    with (REPORTS / f"{tag}_v1001proto_test_video_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video_path", "true_label", "pred_label"])
        w.writeheader()
        w.writerows(rows)
    summary["source"] = f"LOSO fold test=Subject{test_subject} (v1001 protocol)"
    summary["tag"] = tag
    summary["val_threshold"] = best_thr
    (REPORTS / f"{tag}_v1001proto_test_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{tag}: thr={best_thr:.2f} acc={summary['accuracy']*100:.2f} "
          f"BA={summary['balanced_accuracy']*100:.2f} spec={summary['adl_specificity']*100:.1f} "
          f"rec={summary['fall_recall']*100:.2f}", flush=True)
    return summary


def main() -> None:
    sev = load_eval_module()
    for seed in SEEDS:
        for cond in CONDITIONS:
            for test_subject in (1, 2, 3, 4):
                tag = f"c1_s34_data1_loso_ts{test_subject}_{cond}_s{seed}"
                marker = REPORTS / f"{tag}_v1001proto_test_metrics.json"
                if marker.exists():
                    print(f"skip {tag}", flush=True)
                    continue
                print(f"RUN {tag}", flush=True)
                model_path = train(tag, test_subject, cond, seed)
                evaluate(sev, tag, test_subject, model_path)
                print(f"DONE {tag}", flush=True)
    print("MULTI-SEED QUEUE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
