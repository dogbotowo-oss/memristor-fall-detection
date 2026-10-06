from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

import torch


ROOT = Path(r"F:\12.18-2")
V8_DIR = ROOT / "5.9new" / "crossbar+" / "CNN+SNN" / "v8"
SCAN_DIR = V8_DIR / "reports" / "snn_adl_gate_penalty_scan_event_norm"
STRICT_EVAL_PATH = V8_DIR / "reports" / "strict_eval_v8.py"
OUTPUT_DIR = SCAN_DIR / "subject4_compare_p002_p006"


def load_strict_eval():
    spec = importlib.util.spec_from_file_location("strict_eval_v8_subject4_compare", STRICT_EVAL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {STRICT_EVAL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    strict_eval = load_strict_eval()
    mod = strict_eval.load_module()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subject4_videos = mod.list_video_files(strict_eval.SUBJECT4_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for penalty, tag in [("0.02", "p002"), ("0.06", "p006")]:
        run_dir = SCAN_DIR / tag
        model_path = run_dir / f"automatic_fall_event_detector_v8_event_norm_gate_penalty_{tag}.pth"
        validation_summary_path = run_dir / f"strict_validation_summary_event_norm_{tag}.json"
        validation_summary = json.loads(validation_summary_path.read_text(encoding="utf-8"))
        threshold = float(validation_summary["best_threshold"])
        model, config = mod.load_model_checkpoint(model_path, device)
        video_rows, subject4_summary = strict_eval.evaluate_at_threshold(
            mod,
            model,
            config,
            subject4_videos,
            threshold,
            device,
        )
        video_csv = OUTPUT_DIR / f"subject4_video_metrics_{tag}.csv"
        write_csv(video_csv, video_rows, ["video_path", "true_label", "pred_label"])
        payload = {
            "penalty": penalty,
            "tag": tag,
            "model_path": str(model_path),
            "validation_threshold": threshold,
            "validation_source": "Subject3 + Zenodo val only",
            "subject4_source": "Subject4 final independent test",
            "video_metrics_csv": str(video_csv),
            **subject4_summary,
        }
        summary_json = OUTPUT_DIR / f"subject4_summary_{tag}.json"
        summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        rows.append(payload)

    summary_csv = OUTPUT_DIR / "subject4_compare_p002_p006_summary.csv"
    write_csv(
        summary_csv,
        rows,
        [
            "penalty",
            "tag",
            "validation_threshold",
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
            "false_alarm_count",
            "missed_fall_count",
            "model_path",
            "video_metrics_csv",
            "validation_source",
            "subject4_source",
        ],
    )
    print(json.dumps({"summary_csv": str(summary_csv), "rows": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
