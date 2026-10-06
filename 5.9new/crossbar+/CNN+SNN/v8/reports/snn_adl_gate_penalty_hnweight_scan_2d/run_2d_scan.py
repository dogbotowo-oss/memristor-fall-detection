from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(r"F:\12.18-2")
PYTHON = Path(r"F:\ANACONDA\envs\my_yizu_3.10\python.exe")
V8_DIR = ROOT / "5.9new" / "crossbar+" / "CNN+SNN" / "v8"
CODE_PATH = V8_DIR / "code" / "automatic_fall_detector.py"
STRICT_EVAL_PATH = V8_DIR / "reports" / "strict_eval_v8.py"
SCAN_DIR = V8_DIR / "reports" / "snn_adl_gate_penalty_hnweight_scan_2d"
V7_INIT_MODEL = V8_DIR.parent / "v7" / "models" / "automatic_fall_event_detector_cnn_snn_v7_group_channel_gate.pth"
DATA_DIR = ROOT / "data"
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

PENALTIES = [0.04, 0.06, 0.08]
HARD_NEGATIVE_WEIGHTS = [0.80, 1.00, 1.10]


def load_strict_eval():
    spec = importlib.util.spec_from_file_location("strict_eval_v8_for_2d_scan", STRICT_EVAL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {STRICT_EVAL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def tag_for(penalty: float, weight: float) -> str:
    return f"p{int(round(penalty * 100)):03d}_w{int(round(weight * 100)):03d}"


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_training(penalty: float, weight: float, run_dir: Path) -> Path:
    tag = tag_for(penalty, weight)
    model_path = run_dir / "models" / f"automatic_fall_event_detector_v8_2d_{tag}.pth"
    if model_path.exists():
        return model_path

    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "image").mkdir(parents=True, exist_ok=True)
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / f"train_{tag}.stdout.log"
    stderr_path = run_dir / f"train_{tag}.stderr.log"
    cmd = [
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
        str(run_dir / "reports" / f"train_report_{tag}.json"),
        "--training-plot-path",
        str(run_dir / "image" / f"training_curve_{tag}.png"),
        "--use-snn-temporal-branch",
        "--use-snn-fusion-gate",
        "--snn-fusion-gate-mode",
        "group_channel",
        "--learnable-snn-dynamics",
        "--freeze-cnn-gru-epochs",
        "2",
        "--teacher-student-noise",
        "--student-noise-mode",
        "gaussian",
        "--use-device-dynamics",
        "--crossbar-readout-noise-scale",
        "0.25",
        "--hard-negative-normal-weight",
        f"{weight:.2f}",
        "--hard-negative-min-percentile",
        "0.88",
        "--hard-negative-score-gamma",
        "2.00",
        "--normal-positive-penalty",
        "0.00",
        "--snn-adl-gate-penalty",
        f"{penalty:.2f}",
        "--snn-event-score-threshold",
        "0.55",
        "--snn-event-score-sharpness",
        "10.0",
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
    ]
    with stdout_path.open("w", encoding="utf-8") as stdout_obj, stderr_path.open("w", encoding="utf-8") as stderr_obj:
        subprocess.run(cmd, cwd=str(ROOT), stdout=stdout_obj, stderr=stderr_obj, check=True)
    return model_path


def evaluate_validation(model_path: Path, run_dir: Path, penalty: float, weight: float) -> dict[str, object]:
    strict_eval = load_strict_eval()
    mod = strict_eval.load_module()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validation_videos = sorted(dict.fromkeys(mod.list_video_files(SUBJECT3_DIR) + mod.list_video_files(ZENODO_VAL)))
    model, config = mod.load_model_checkpoint(model_path, device)
    scan_rows, best_threshold = strict_eval.scan_thresholds(mod, model, config, validation_videos, device)
    tag = tag_for(penalty, weight)
    scan_csv = run_dir / "reports" / f"strict_validation_threshold_scan_{tag}.csv"
    write_csv(scan_csv, scan_rows, list(scan_rows[0].keys()))
    best_row = next(row for row in scan_rows if abs(float(row["threshold"]) - float(best_threshold)) < 1e-9)
    summary = {
        "tag": tag,
        "snn_adl_gate_penalty": float(penalty),
        "hard_negative_normal_weight": float(weight),
        "best_threshold": float(best_threshold),
        "model_path": str(model_path),
        "scan_csv": str(scan_csv),
        "source": "Subject3 + Zenodo val only",
        **best_row,
    }
    (run_dir / "reports" / f"strict_validation_summary_{tag}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    SCAN_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for weight in HARD_NEGATIVE_WEIGHTS:
        for penalty in PENALTIES:
            tag = tag_for(penalty, weight)
            run_dir = SCAN_DIR / tag
            model_path = run_training(penalty, weight, run_dir)
            rows.append(evaluate_validation(model_path, run_dir, penalty, weight))

    summary_csv = SCAN_DIR / "strict_validation_2d_scan_summary.csv"
    write_csv(
        summary_csv,
        rows,
        [
            "tag",
            "snn_adl_gate_penalty",
            "hard_negative_normal_weight",
            "best_threshold",
            "total",
            "accuracy",
            "balanced_accuracy",
            "adl_specificity",
            "fall_recall",
            "precision",
            "tp",
            "tn",
            "fp",
            "fn",
            "model_path",
            "scan_csv",
            "source",
            "threshold",
        ],
    )
    print(json.dumps({"summary_csv": str(summary_csv), "rows": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
