"""LOSO non-ideality ablation, middle conditions -> v1001.

The v1000 queue already produced, for every fold ts1..ts4 (seed 11, s34 recipe):
  - full  : measured HRS/LRS distributions + measured device dynamics
            (EPSC/PPF/LTP/LTD gain modulation) + readout noise 0.25
  - ideal : --disable-crossbar (raw frames, no conductance mapping)

This queue adds the two missing factorization steps:
  - dist      : measured HRS/LRS distributions only
                (no --use-device-dynamics, --crossbar-readout-noise-scale 0)
  - distnoise : measured HRS/LRS distributions + readout noise 0.25
                (no --use-device-dynamics)

Paths updated from E:\\rray\\12.18-2 to F:\\12.18-2; everything else
(training recipe, seed 11, Zenodo-val thresholding, frozen one-shot test)
is identical to queue_loso_v1000.py.
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
CODE_PATH = S30 / "code" / "automatic_fall_detector.py"
EVAL_PATH = S30 / "analysis" / "strict_eval_s30.py"
MODELS = S30 / "models"
REPORTS = S30 / "reports"
IMAGE = S30 / "image"
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

# condition -> (suffix, extra flags). Flags replace the full-condition
# "--use-device-dynamics --crossbar-readout-noise-scale 0.25" pair.
CONDITIONS = {
    "dist": [],
    "distnoise": ["--crossbar-readout-noise-scale", "0.25"],
    # full recipe re-run against F:\12.18-2\data1 so that the ablation shares
    # the same measured LTP/LTD curves as the paper figures (v1000 full used
    # the older E:\rray\12.18-2\data1\data files).
    "full2": ["--use-device-dynamics", "--crossbar-readout-noise-scale", "0.25"],
    # bw = full2 recipe + SNN branch linear weights binarized (QAT/STE),
    # emulating synaptic weights hosted on the two measured conductance states.
    "bw": ["--use-device-dynamics", "--crossbar-readout-noise-scale", "0.25",
           "--snn-binary-weights"],
}


def load_eval_module():
    spec = importlib.util.spec_from_file_location("strict_eval_loso", EVAL_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    mod.ROOT = ROOT
    mod.S30_DIR = S30
    mod.CODE_PATH = CODE_PATH
    mod.REPORTS_DIR = REPORTS
    mod.DATA_DIR = DATA1
    mod.POSE_CACHE_DIR = POSE_CACHE
    mod.GMDCSA_ROOT = GMDCSA
    mod.ZENODO_VAL_DIR = ZENODO_VAL
    return mod


def train(tag: str, test_subject: int, cond: str) -> Path:
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
        *CONDITIONS[cond],
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
        for cond in ("dist", "distnoise", "full2", "bw"):
            tag = f"c1_s34_data1_loso_ts{test_subject}_{cond}_s11"
            marker = REPORTS / f"{tag}_test_metrics.json"
            if marker.exists():
                print(f"skip {tag}", flush=True)
                continue
            print(f"RUN {tag}", flush=True)
            model_path = train(tag, test_subject, cond)
            evaluate(sev, tag, test_subject, model_path)
            print(f"DONE {tag}", flush=True)
    print("LOSO DIST QUEUE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
