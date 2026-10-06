"""s30 training + evaluation runner.

Recipe: identical to the v9/s30 baseline (2D scan p006_w100), with the SNN
branch input extended by MediaPipe pose displacement features:
  --use-snn-pose-input --pose-cache-dir <ROOT>/pose_cache

Baseline for ablation: v9 p006_w100 (val thr=0.70: acc 72.13 / balacc 71.99 /
ADL spec 63.33 / fall recall 80.65; Subject4: acc 67.57 / spec 60.00 /
recall 76.47).

Protocol: train on Subject1+2 + Zenodo train; threshold scan on
Subject3 + Zenodo val; one-shot frozen-threshold Subject4 test.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"E:\rray\12.18-2")
PYTHON = Path(r"D:\software\anaconde\envs\my_yizu_3.10\python.exe")
S30_DIR = ROOT / "5.9new" / "crossbar+" / "CNN+SNN" / "s30_pose"
CODE_PATH = S30_DIR / "code" / "automatic_fall_detector.py"
EVAL_PATH = S30_DIR / "analysis" / "strict_eval_s30.py"
MODELS_DIR = S30_DIR / "models"
REPORTS_DIR = S30_DIR / "reports"
IMAGE_DIR = S30_DIR / "image"
DATA_DIR = ROOT / "data"
POSE_CACHE_DIR = ROOT / "pose_cache"
ZENODO_TRAIN = ROOT / "zenodo_falldb_video_split" / "train"
ZENODO_VAL = ROOT / "zenodo_falldb_video_split" / "val"
GMDCSA_ROOT = (
    ROOT
    / "测试集"
    / "13354453"
    / "ekramalam"
    / "GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-v2.1"
    / "ekramalam-GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-5abac76"
)
SUBJECT1_DIR = GMDCSA_ROOT / "Subject 1"
SUBJECT2_DIR = GMDCSA_ROOT / "Subject 2"
SUBJECT3_DIR = GMDCSA_ROOT / "Subject 3"
V7_INIT_MODEL = (
    ROOT
    / "5.9new"
    / "crossbar+"
    / "CNN+SNN"
    / "v7"
    / "models"
    / "automatic_fall_event_detector_cnn_snn_v7_group_channel_gate.pth"
)

def _arg_value(name: str, default: float) -> float:
    if name in sys.argv:
        idx = sys.argv.index(name)
        return float(sys.argv[idx + 1])
    return default


GATE_FLOOR = _arg_value("--floor", 0.20)
LATE_FUSION_LAMBDA = _arg_value("--late-fusion", 0.0)
SNN_CLASS_AUX = _arg_value("--snn-aux", 0.0)
SEED = int(_arg_value("--seed", 42))
FALL_PRESERVE_W = _arg_value("--preserve", 0.0)
FALL_PRESERVE_M = _arg_value("--preserve-margin", 0.5)
NO_SNN = "--no-snn" in sys.argv
DISABLE_CROSSBAR = "--disable-crossbar" in sys.argv


def _str_value(name: str) -> str | None:
    if name in sys.argv:
        idx = sys.argv.index(name)
        return str(sys.argv[idx + 1])
    return None


_data_dir_override = _str_value("--data-dir")
if _data_dir_override:
    DATA_DIR = Path(_data_dir_override)

_tag_parts = ["c1_s30_pose"]
if GATE_FLOOR > 0:
    _tag_parts.append(f"floor{int(round(GATE_FLOOR * 100)):03d}")
if LATE_FUSION_LAMBDA > 0 or SNN_CLASS_AUX > 0:
    _tag_parts = ["c1_s31_pose"] + _tag_parts[1:]
    if LATE_FUSION_LAMBDA > 0:
        _tag_parts.append(f"lam{int(round(LATE_FUSION_LAMBDA * 100)):03d}")
    if SNN_CLASS_AUX > 0:
        _tag_parts.append(f"aux{int(round(SNN_CLASS_AUX * 100)):03d}")
TAG = _str_value("--tag") or "_".join(_tag_parts)

if NO_SNN and _str_value("--tag") is None:
    TAG = f"c1_s34_nosnn_s{SEED}"


def run_training(model_path: Path) -> Path:
    if model_path.exists():
        print(f"Model already exists, skip training: {model_path}")
        return model_path
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stdout_path = REPORTS_DIR / f"train_{TAG}.stdout.log"
    stderr_path = REPORTS_DIR / f"train_{TAG}.stderr.log"
    base_cmd = [
        str(PYTHON),
        str(CODE_PATH),
        "--mode",
        "train",
        "--skip-main-video-training",
        "--disable-auxiliary-data",
        "--external-labeled-dir",
        str(SUBJECT1_DIR),
        str(SUBJECT2_DIR),
        str(ZENODO_TRAIN),
        "--external-val-labeled-dir",
        str(SUBJECT3_DIR),
        str(ZENODO_VAL),
        "--data-dir",
        str(DATA_DIR),
        "--init-model-path",
        str(V7_INIT_MODEL),
        "--model-path",
        str(model_path),
        "--report-json",
        str(REPORTS_DIR / f"train_report_{TAG}.json"),
        "--training-plot-path",
        str(IMAGE_DIR / f"training_curve_{TAG}.png"),
        "--freeze-cnn-gru-epochs",
        "2",
        "--teacher-student-noise",
        "--student-noise-mode",
        "gaussian",
        "--use-device-dynamics",
        "--crossbar-readout-noise-scale",
        "0.25",
        "--hard-negative-normal-weight",
        "1.00",
        "--hard-negative-min-percentile",
        "0.88",
        "--hard-negative-score-gamma",
        "2.00",
        "--normal-positive-penalty",
        "0.00",
        "--external-labeled-stride",
        "16",
        "--adl-context-path-keyword",
        "zenodo_falldb_video_split/train/adl",
        "--adl-context-min-percentile",
        "0.95",
        "--adl-context-extra-copies",
        "1",
        "--image-size",
        "64",
        "--skip-visual-outputs",
        "--seed",
        str(SEED),
    ]
    snn_cmd = [
        "--use-snn-temporal-branch",
        "--use-snn-fusion-gate",
        "--snn-fusion-gate-mode",
        "group_channel",
        "--learnable-snn-dynamics",
        "--snn-adl-gate-penalty",
        "0.06",
        "--snn-event-score-threshold",
        "0.55",
        "--snn-event-score-sharpness",
        "10.0",
        "--use-snn-pose-input",
        "--pose-cache-dir",
        str(POSE_CACHE_DIR),
        "--snn-fusion-gate-floor",
        f"{GATE_FLOOR:.2f}",
        "--late-fusion-lambda",
        f"{LATE_FUSION_LAMBDA:.2f}",
        "--snn-class-aux-weight",
        f"{SNN_CLASS_AUX:.2f}",
        "--snn-fall-preserve-weight",
        f"{FALL_PRESERVE_W:.2f}",
        "--snn-fall-preserve-margin",
        f"{FALL_PRESERVE_M:.2f}",
    ]
    if NO_SNN:
        cmd = base_cmd
    else:
        cmd = base_cmd + snn_cmd
    if DISABLE_CROSSBAR:
        cmd = cmd + ["--disable-crossbar"]
    print("Training command:", " ".join(cmd))
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        subprocess.run(cmd, cwd=str(ROOT), stdout=out, stderr=err, check=True)
    return model_path


def run_evaluation(model_path: Path) -> dict[str, object]:
    spec = importlib.util.spec_from_file_location("strict_eval_s30", EVAL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {EVAL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module.run_evaluation(model_path, TAG, run_subject4=True, data_dir=DATA_DIR)


def main() -> None:
    model_path = MODELS_DIR / f"automatic_fall_event_detector_{TAG}.pth"
    run_training(model_path)
    result = run_evaluation(model_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
