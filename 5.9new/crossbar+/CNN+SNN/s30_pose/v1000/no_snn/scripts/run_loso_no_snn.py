"""Train and evaluate four matched no-SNN v1000 LOSO folds."""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"E:\rray\12.18-2")
PYTHON = Path(r"D:\software\anaconde\envs\my_yizu_3.10\python.exe")
S30 = ROOT / "5.9new" / "crossbar+" / "CNN+SNN" / "s30_pose"
V1000_NO_SNN = S30 / "v1000" / "no_snn"
CODE_PATH = S30 / "code" / "automatic_fall_detector.py"
EVAL_PATH = S30 / "analysis" / "strict_eval_s30.py"
MODELS = V1000_NO_SNN / "models"
REPORTS = V1000_NO_SNN / "reports"
IMAGE = V1000_NO_SNN / "image"
DATA1 = ROOT / "data1" / "data"
ZENODO_TRAIN = ROOT / "zenodo_falldb_video_split" / "train"
ZENODO_VAL = ROOT / "zenodo_falldb_video_split" / "val"
V1_INIT = (
    ROOT
    / "5.9new"
    / "crossbar+"
    / "CNN+SNN"
    / "models"
    / "automatic_fall_event_detector_cnn_snn_v1.pth"
)
GMDCSA = (
    ROOT
    / "测试集"
    / "13354453"
    / "ekramalam"
    / "GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-v2.1"
    / "ekramalam-GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-5abac76"
)
SUBJECT_DIRS = {subject: GMDCSA / f"Subject {subject}" for subject in (1, 2, 3)}
SUBJECT_DIRS[4] = ROOT / "gmdcsa_subject4_test" / "Subject 4"
CAL_SUBJECT = {
    1: SUBJECT_DIRS[3],
    2: SUBJECT_DIRS[3],
    3: SUBJECT_DIRS[4],
    4: SUBJECT_DIRS[3],
}
SEED = 11


def tag_for(test_subject: int) -> str:
    return f"c1_s34_data1_v1000_loso_ts{test_subject}_nosnn_s{SEED}"


def ensure_directories() -> None:
    for path in (MODELS, REPORTS, IMAGE, V1000_NO_SNN / "comparison"):
        path.mkdir(parents=True, exist_ok=True)


def load_eval_module():
    spec = importlib.util.spec_from_file_location("strict_eval_v1000_nosnn", EVAL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {EVAL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.DATA_DIR = DATA1
    return module


def train(test_subject: int) -> Path:
    tag = tag_for(test_subject)
    model_path = MODELS / f"automatic_fall_event_detector_{tag}.pth"
    if model_path.exists():
        print(f"model exists, skip training: {model_path}", flush=True)
        return model_path

    train_subjects = [subject for subject in (1, 2, 3, 4) if subject != test_subject]
    command = [
        str(PYTHON),
        str(CODE_PATH),
        "--mode",
        "train",
        "--skip-main-video-training",
        "--disable-auxiliary-data",
        "--external-labeled-dir",
        *[str(SUBJECT_DIRS[subject]) for subject in train_subjects],
        str(ZENODO_TRAIN),
        "--external-val-labeled-dir",
        str(ZENODO_VAL),
        "--data-dir",
        str(DATA1),
        "--init-model-path",
        str(V1_INIT),
        "--model-path",
        str(model_path),
        "--report-json",
        str(REPORTS / f"train_report_{tag}.json"),
        "--training-plot-path",
        str(IMAGE / f"training_curve_{tag}.png"),
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
    (REPORTS / f"train_command_{tag}.json").write_text(
        json.dumps(command, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"TRAIN {tag}", flush=True)
    with (REPORTS / f"train_{tag}.stdout.log").open("w", encoding="utf-8") as stdout, (
        REPORTS / f"train_{tag}.stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        subprocess.run(command, cwd=str(ROOT), stdout=stdout, stderr=stderr, check=True)
    return model_path


def write_video_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["video_path", "true_label", "pred_label"]
        )
        writer.writeheader()
        writer.writerows(rows)


def evaluate_at_rule(
    evaluator,
    module,
    model,
    config,
    device,
    tag: str,
    test_subject: int,
    validation_videos,
    suffix: str,
    source: str,
    recall_priority: bool = False,
) -> dict[str, object]:
    scan_rows, best_threshold = evaluator.scan_thresholds(
        module, model, config, validation_videos, device
    )
    if recall_priority:
        grid = [row for row in scan_rows if float(row["threshold"]) <= 0.70 + 1e-9]
        candidates = [row for row in grid if float(row["fall_recall"]) >= 0.85]
        if not candidates:
            max_recall = max(float(row["fall_recall"]) for row in grid)
            candidates = [row for row in grid if float(row["fall_recall"]) == max_recall]
        selected_row = max(
            candidates,
            key=lambda row: (float(row["adl_specificity"]), float(row["threshold"])),
        )
        best_threshold = float(selected_row["threshold"])
    else:
        selected_row = next(
            row
            for row in scan_rows
            if abs(float(row["threshold"]) - float(best_threshold)) < 1e-9
        )

    validation_summary = {
        "tag": tag,
        "best_threshold": float(best_threshold),
        "source": source,
        **selected_row,
    }
    validation_name = {
        "test": "val_summary",
        "gtest": "gval_summary",
        "gtestR": "gvalR_summary",
    }[suffix]
    (REPORTS / f"{tag}_{validation_name}.json").write_text(
        json.dumps(validation_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    test_videos = sorted(module.list_video_files(SUBJECT_DIRS[test_subject]))
    rows, summary = evaluator.evaluate_at_threshold(
        module, model, config, test_videos, best_threshold, device
    )
    write_video_rows(REPORTS / f"{tag}_{suffix}_video_metrics.csv", rows)
    summary["source"] = source.replace("validation", f"test Subject{test_subject}")
    summary["tag"] = tag
    (REPORTS / f"{tag}_{suffix}_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"{tag} [{suffix}]: thr={best_threshold:.2f} "
        f"acc={summary['accuracy'] * 100:.2f} "
        f"spec={summary['adl_specificity'] * 100:.2f} "
        f"rec={summary['fall_recall'] * 100:.2f}",
        flush=True,
    )
    return summary


def evaluate(test_subject: int, model_path: Path) -> None:
    import torch

    tag = tag_for(test_subject)
    required = [
        REPORTS / f"{tag}_test_metrics.json",
        REPORTS / f"{tag}_gtest_metrics.json",
        REPORTS / f"{tag}_gtestR_metrics.json",
    ]
    if all(path.exists() for path in required):
        print(f"evaluation exists, skip: {tag}", flush=True)
        return

    evaluator = load_eval_module()
    module = evaluator.load_module()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = module.load_model_checkpoint(model_path, device)

    if not required[0].exists():
        zenodo_validation = sorted(module.list_video_files(ZENODO_VAL))
        evaluate_at_rule(
            evaluator,
            module,
            model,
            config,
            device,
            tag,
            test_subject,
            zenodo_validation,
            "test",
            "Zenodo-val-only validation; frozen LOSO test threshold",
        )

    gmdcsa_validation = sorted(module.list_video_files(CAL_SUBJECT[test_subject]))
    gmdcsa_validation += sorted(module.list_video_files(ZENODO_VAL))
    gmdcsa_validation = sorted(dict.fromkeys(gmdcsa_validation))
    if not required[1].exists():
        evaluate_at_rule(
            evaluator,
            module,
            model,
            config,
            device,
            tag,
            test_subject,
            gmdcsa_validation,
            "gtest",
            f"GMDCSA-informed validation ({CAL_SUBJECT[test_subject].name} + Zenodo val)",
        )
    if not required[2].exists():
        evaluate_at_rule(
            evaluator,
            module,
            model,
            config,
            device,
            tag,
            test_subject,
            gmdcsa_validation,
            "gtestR",
            f"Recall-priority validation ({CAL_SUBJECT[test_subject].name} + Zenodo val)",
            recall_priority=True,
        )


def main() -> int:
    ensure_directories()
    for test_subject in (1, 2, 3, 4):
        tag = tag_for(test_subject)
        print(f"RUN {tag}", flush=True)
        model_path = train(test_subject)
        evaluate(test_subject, model_path)
        print(f"DONE {tag}", flush=True)
    print("V1000 NO-SNN LOSO COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
