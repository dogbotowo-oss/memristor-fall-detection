from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import torch


ROOT = Path(r"F:\12.18-2")
V9_DIR = ROOT / "5.9new" / "crossbar+" / "CNN+SNN" / "v9"
CODE_PATH = V9_DIR / "code" / "automatic_fall_detector.py"
MODEL_PATH = V9_DIR / "models" / "automatic_fall_event_detector_cnn_snn_v9_final.pth"
REPORTS_DIR = V9_DIR / "reports"
STRICT_EVAL_V8_PATH = ROOT / "5.9new" / "crossbar+" / "CNN+SNN" / "v8" / "reports" / "strict_eval_v8.py"
THRESHOLD = 0.70


def load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    strict_eval = load_module_from_path("strict_eval_v8_helpers_for_v9", STRICT_EVAL_V8_PATH)
    mod = load_module_from_path("automatic_fall_detector_v9", CODE_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = mod.load_model_checkpoint(MODEL_PATH, device)
    subject4_videos = mod.list_video_files(strict_eval.SUBJECT4_DIR)
    rows, summary = strict_eval.evaluate_at_threshold(mod, model, config, subject4_videos, THRESHOLD, device)
    video_csv = REPORTS_DIR / "v9_subject4_final_video_metrics.csv"
    strict_eval.write_csv(video_csv, rows, ["video_path", "true_label", "pred_label"])
    summary.update(
        {
            "version": "v9",
            "threshold_source": "Subject3 + Zenodo val only",
            "subject4_source": "Subject4 final independent test",
            "model_path": str(MODEL_PATH),
            "video_metrics_csv": str(video_csv),
        }
    )
    (REPORTS_DIR / "v9_subject4_final_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
