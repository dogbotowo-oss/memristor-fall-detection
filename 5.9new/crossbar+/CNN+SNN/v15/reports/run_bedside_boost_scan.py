from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"F:\12.18-2")
EXP_DIR = ROOT / "5.9new" / "crossbar+" / "CNN+SNN" / "v15"
CODE_PATH = EXP_DIR / "code" / "automatic_fall_detector.py"
STRICT_EVAL_PATH = EXP_DIR / "reports" / "strict_eval_v15.py"
SCAN_DIR = EXP_DIR / "reports" / "bedside_boost_scan"
PYTHON = Path(sys.executable)

TRAIN_DIRS = [
    ROOT / "测试集" / "13354453" / "ekramalam" / "GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-v2.1" / "ekramalam-GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-5abac76" / "Subject 1",
    ROOT / "测试集" / "13354453" / "ekramalam" / "GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-v2.1" / "ekramalam-GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-5abac76" / "Subject 2",
    ROOT / "zenodo_falldb_video_split" / "train",
]
VAL_DIRS = [
    ROOT / "测试集" / "13354453" / "ekramalam" / "GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-v2.1" / "ekramalam-GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-5abac76" / "Subject 3",
    ROOT / "zenodo_falldb_video_split" / "val",
]

COMMON_ARGS = [
    "--mode",
    "train",
    "--external-labeled-stride",
    "16",
    "--data-dir",
    str(ROOT / "data"),
    "--disable-auxiliary-data",
    "--skip-main-video-training",
    "--skip-visual-outputs",
    "--window-size",
    "16",
    "--image-size",
    "64",
    "--hidden-dim",
    "64",
    "--epochs",
    "14",
    "--batch-size",
    "8",
    "--lr",
    "0.0008",
    "--weight-decay",
    "0.0001",
    "--use-snn-temporal-branch",
    "--snn-hidden-dim",
    "32",
    "--snn-decay",
    "0.82",
    "--snn-threshold",
    "0.55",
    "--snn-surrogate-scale",
    "8.0",
    "--learnable-snn-dynamics",
    "--use-snn-fusion-gate",
    "--snn-fusion-gate-mode",
    "group_channel",
    "--snn-event-score-threshold",
    "0.55",
    "--snn-event-score-sharpness",
    "10.0",
    "--use-adl-subtype-head",
    "--adl-subtype-loss-weight",
    "0.30",
    "--adl-subtype-hard-only",
    "--adl-subtype-hard-score-threshold",
    "0.22",
    "--targeted-adl-fall-penalty",
    "0.12",
    "--targeted-adl-subtypes",
    "sit_descent",
    "bend_reach",
    "squat_kneel",
    "bedside_forward_bend",
    "--freeze-cnn-gru-epochs",
    "2",
    "--use-device-dynamics",
    "--crossbar-readout-noise-scale",
    "0.25",
    "--teacher-student-noise",
    "--student-noise-mode",
    "gaussian",
    "--student-noise-std",
    "0.035",
    "--student-noise-prob",
    "0.75",
    "--student-frame-drop-prob",
    "0.08",
    "--device-noise-scale",
    "1.0",
    "--teacher-ema",
    "0.996",
    "--consistency-weight",
    "0.25",
    "--infer-stride",
    "4",
    "--video-window-stride",
    "1",
    "--max-aux-fall",
    "60",
    "--external-no-fall-weight",
    "0.35",
    "--hard-negative-normal-weight",
    "1.0",
    "--hard-negative-score-mode",
    "quantile",
    "--hard-negative-min-percentile",
    "0.88",
    "--hard-negative-score-gamma",
    "2.0",
    "--hard-negative-max-multiplier",
    "4.0",
    "--snn-adl-gate-penalty",
    "0.06",
    "--adl-context-path-keyword",
    "zenodo_falldb_video_split/train/adl",
    "--adl-context-min-percentile",
    "0.95",
    "--adl-context-radius-windows",
    "0",
    "--adl-context-extra-copies",
    "1",
    "--normal-positive-penalty",
    "0.0",
]


def run_command(command: list[str], log_path: Path | None = None) -> None:
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as handle:
            process = subprocess.run(
                command,
                cwd=str(ROOT),
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=True,
            )
        return
    subprocess.run(command, cwd=str(ROOT), check=True)


def train_variant(tag: str, boost_weight: float) -> dict[str, object]:
    model_path = EXP_DIR / "models" / f"automatic_fall_event_detector_cnn_snn_{tag}.pth"
    report_json = SCAN_DIR / tag / "train_report.json"
    plot_path = SCAN_DIR / tag / "train_main_plot.png"
    curve_path = SCAN_DIR / tag / "training_curve.png"
    train_log = SCAN_DIR / tag / "train.stdout.log"

    command = [
        str(PYTHON),
        str(CODE_PATH),
        "--external-labeled-dir",
        *[str(path) for path in TRAIN_DIRS],
        "--external-val-labeled-dir",
        *[str(path) for path in VAL_DIRS],
        "--init-model-path",
        str(EXP_DIR / "models" / "automatic_fall_event_detector_cnn_snn_v15_first_round.pth"),
        "--model-path",
        str(model_path),
        "--training-plot-path",
        str(curve_path),
        "--report-json",
        str(report_json),
        "--plot-path",
        str(plot_path),
        "--adl-subtype-sample-boost-weight",
        f"{boost_weight:.2f}",
        "--adl-subtype-sample-boost-subtypes",
        "bedside_forward_bend",
        *COMMON_ARGS,
    ]
    run_command(command, train_log)
    return {
        "tag": tag,
        "boost_weight": boost_weight,
        "model_path": str(model_path),
        "train_log": str(train_log),
    }


def eval_variant(tag: str, model_path: Path) -> dict[str, object]:
    run_command(
        [
            str(PYTHON),
            str(STRICT_EVAL_PATH),
            "--model-path",
            str(model_path),
            "--prefix",
            tag,
        ]
    )
    metrics_path = EXP_DIR / "reports" / f"{tag}_subject4_final_metrics.json"
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def main() -> None:
    SCAN_DIR.mkdir(parents=True, exist_ok=True)
    variants = [
        ("v15_bedside_boost025", 0.25),
        ("v15_bedside_boost050", 0.50),
    ]
    summary_rows: list[dict[str, object]] = []

    for tag, boost_weight in variants:
        train_info = train_variant(tag, boost_weight)
        metrics = eval_variant(tag, Path(train_info["model_path"]))
        summary_rows.append(
            {
                "tag": tag,
                "boost_weight": boost_weight,
                "threshold": float(metrics["threshold"]),
                "accuracy": float(metrics["accuracy"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "adl_specificity": float(metrics["adl_specificity"]),
                "fall_recall": float(metrics["fall_recall"]),
                "precision": float(metrics["precision"]),
                "tp": int(metrics["tp"]),
                "tn": int(metrics["tn"]),
                "fp": int(metrics["fp"]),
                "fn": int(metrics["fn"]),
            }
        )

    # Reuse the already-finished 0.75 result to complete the scan table.
    existing_metrics = json.loads(
        (EXP_DIR / "reports" / "v15_second_round_adlboost_subject4_final_metrics.json").read_text(encoding="utf-8")
    )
    summary_rows.append(
        {
            "tag": "v15_second_round_adlboost",
            "boost_weight": 0.75,
            "threshold": float(existing_metrics["threshold"]),
            "accuracy": float(existing_metrics["accuracy"]),
            "balanced_accuracy": float(existing_metrics["balanced_accuracy"]),
            "adl_specificity": float(existing_metrics["adl_specificity"]),
            "fall_recall": float(existing_metrics["fall_recall"]),
            "precision": float(existing_metrics["precision"]),
            "tp": int(existing_metrics["tp"]),
            "tn": int(existing_metrics["tn"]),
            "fp": int(existing_metrics["fp"]),
            "fn": int(existing_metrics["fn"]),
        }
    )

    summary_rows.sort(key=lambda row: float(row["boost_weight"]))
    summary_path = SCAN_DIR / "bedside_boost_scan_summary.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary_rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
