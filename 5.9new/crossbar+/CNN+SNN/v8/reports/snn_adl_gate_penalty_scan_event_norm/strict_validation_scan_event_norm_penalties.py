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


def load_strict_eval():
    spec = importlib.util.spec_from_file_location("strict_eval_v8_event_norm_scan", STRICT_EVAL_PATH)
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
    validation_videos = mod.list_video_files(strict_eval.SUBJECT3_DIR) + mod.list_video_files(strict_eval.ZENODO_VAL_DIR)
    rows: list[dict[str, object]] = []
    for penalty, tag in [
        ("0.00", "p000"),
        ("0.02", "p002"),
        ("0.04", "p004"),
        ("0.06", "p006"),
        ("0.08", "p008"),
    ]:
        run_dir = SCAN_DIR / tag
        model_path = run_dir / f"automatic_fall_event_detector_v8_event_norm_gate_penalty_{tag}.pth"
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        model, config = mod.load_model_checkpoint(model_path, device)
        scan_rows, best_threshold = strict_eval.scan_thresholds(mod, model, config, validation_videos, device)
        scan_csv = run_dir / f"strict_validation_threshold_scan_event_norm_{tag}.csv"
        write_csv(scan_csv, scan_rows, list(scan_rows[0].keys()))
        best_row = next(row for row in scan_rows if abs(float(row["threshold"]) - float(best_threshold)) < 1e-9)
        summary = {
            "penalty": penalty,
            "tag": tag,
            "model_path": str(model_path),
            "best_threshold": float(best_threshold),
            "scan_csv": str(scan_csv),
            "source": "Subject3 + Zenodo val only",
            **best_row,
        }
        (run_dir / f"strict_validation_summary_event_norm_{tag}.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        rows.append(summary)

    summary_csv = SCAN_DIR / "strict_validation_event_norm_penalty_scan_summary.csv"
    summary_fields = [
        "penalty",
        "tag",
        "best_threshold",
        "threshold",
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
    ]
    write_csv(summary_csv, rows, summary_fields)
    print(json.dumps({"summary_csv": str(summary_csv), "rows": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
