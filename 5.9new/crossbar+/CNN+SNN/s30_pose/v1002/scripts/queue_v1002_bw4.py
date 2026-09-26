"""v1002: SNN branch weights quantized onto measured multilevel conductance states.

Same training recipe / seed / LOSO protocol as the v1000 full condition and the
v1001 bw condition; the only difference is --snn-weight-levels carrying the
measured compliance-state conductance ratios:

  compliance 5 mA / 1 mA / 500 uA / 100 uA
  -> LRS 847 Ohm / 8.1 kOhm / 40 kOhm / 90 kOhm (medians, read at +0.1 V)
  -> conductance ratios {1.0, 0.104, 0.021, 0.0094}

Outputs (models/reports/image) live under v1002/, leaving s30_pose untouched.
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

SEED = 11
LEVELS = "0.0094,0.021,0.104,1.0"


def load_eval_module():
    spec = importlib.util.spec_from_file_location("strict_eval_v1002", EVAL_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    mod.DATA_DIR = DATA1
    mod.POSE_CACHE_DIR = POSE_CACHE
    return mod


def train(tag: str, test_subject: int) -> Path:
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
        "--seed", str(SEED),
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
        "--snn-weight-levels", LEVELS,
    ]
    with (REPORTS / f"train_{tag}.stdout.log").open("w", encoding="utf-8") as out, \
         (REPORTS / f"train_{tag}.stderr.log").open("w", encoding="utf-8") as err:
        subprocess.run(cmd, cwd=str(ROOT), stdout=out, stderr=err, check=True)
    return model_path


def evaluate(sev, tag: str, test_subject: int, model_path: Path) -> dict:
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mod = sev.load_module()
    model, config = mod.load_model_checkpoint(model_path, device)

    val_videos = sorted(mod.list_video_files(ZENODO_VAL))
    scan_rows, best_thr = sev.scan_thresholds(mod, model, config, val_videos, device)
    best_row = next(r for r in scan_rows if abs(float(r["threshold"]) - float(best_thr)) < 1e-9)
    val_summary = {"tag": tag, "best_threshold": float(best_thr),
                   "source": "Zenodo val only (LOSO)", **best_row}
    (REPORTS / f"{tag}_val_summary.json").write_text(
        json.dumps(val_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    test_videos = sorted(mod.list_video_files(SUBJECT_DIRS[test_subject]))
    rows, summary = sev.evaluate_at_threshold(mod, model, config, test_videos, best_thr, device)
    with (REPORTS / f"{tag}_test_video_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video_path", "true_label", "pred_label"])
        w.writeheader()
        w.writerows(rows)
    summary["source"] = f"LOSO fold test=Subject{test_subject} (threshold frozen from Zenodo val)"
    summary["tag"] = tag
    (REPORTS / f"{tag}_test_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{tag}: thr={best_thr:.2f} acc={summary['accuracy']*100:.2f} "
          f"spec={summary['adl_specificity']*100:.1f} rec={summary['fall_recall']*100:.2f}", flush=True)
    return summary


def main() -> None:
    sev = load_eval_module()
    for test_subject in (1, 2, 3, 4):
        tag = f"c1_s34_data1_loso_ts{test_subject}_bw4_s11"
        marker = REPORTS / f"{tag}_test_metrics.json"
        if marker.exists():
            print(f"skip {tag}", flush=True)
            continue
        print(f"RUN {tag}", flush=True)
        model_path = train(tag, test_subject)
        evaluate(sev, tag, test_subject, model_path)
        print(f"DONE {tag}", flush=True)
    print("V1002 BW4 QUEUE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
