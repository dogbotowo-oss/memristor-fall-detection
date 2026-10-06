"""s34 5-seed probability-ensemble evaluation.

Averages frame-level fall probabilities across the five trained s34 seeds
(1, 3, 5, 7, 42). The ensemble rule is fixed a priori; threshold selection
follows the same frozen protocol as strict_eval_s30 (scan on Subject3 +
Zenodo val only, selection range <= 0.70), then a one-shot Subject4 test.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(r"E:\rray\12.18-2")
S30_DIR = ROOT / "5.9new" / "crossbar+" / "CNN+SNN" / "s30_pose"
EVAL_PATH = S30_DIR / "analysis" / "strict_eval_s30.py"
MODELS_DIR = S30_DIR / "models"
REPORTS_DIR = S30_DIR / "reports"
SEEDS = [1, 3, 5, 7, 42]
TAG = "c1_s34_ensemble5"


def load_eval_module():
    spec = importlib.util.spec_from_file_location("strict_eval_s30", EVAL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {EVAL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def ensemble_predict(ev, models, videos, device):
    """Return per-video dicts with frame_probabilities averaged over seeds."""
    per_model_cache: list[list[dict[str, object]]] = []
    for model_path, model, config in models:
        print(f"predicting with {model_path.name}", flush=True)
        per_model_cache.append(
            [ev.predict_video(ev.mod, model, config, video_path, device) for video_path in videos]
        )
    rows: list[dict[str, object]] = []
    for idx, video_path in enumerate(videos):
        items = [cache[idx] for cache in per_model_cache]
        lengths = {len(it["frame_probabilities"]) for it in items}
        if len(lengths) != 1:
            raise RuntimeError(f"frame_probabilities length mismatch for {video_path}: {lengths}")
        avg_probs = np.mean(
            [np.asarray(it["frame_probabilities"], dtype=np.float32) for it in items], axis=0
        )
        rows.append(
            {
                "video_path": str(video_path),
                "fps": float(items[0]["fps"]),
                "frame_probabilities": avg_probs,
                "true_label": int(items[0]["true_label"]),
            }
        )
    return rows


def scan_and_select(ev, cached):
    """Same selection rule as strict_eval_s30.scan_thresholds (<=0.70)."""
    thresholds = [round(v, 2) for v in np.arange(0.35, 0.91, 0.05)]
    rows: list[dict[str, object]] = []
    best_threshold = thresholds[0]
    best_score = -1.0
    best_spec = -1.0
    tol = 1e-9
    for threshold in thresholds:
        eval_rows = [
            {
                "video_path": item["video_path"],
                "true_label": item["true_label"],
                "pred_label": int(ev.classify_video(ev.mod, item, threshold)),
            }
            for item in cached
        ]
        metrics = ev.compute_metrics(eval_rows)
        rows.append({"threshold": threshold, **metrics})
        if threshold > 0.70:
            continue
        score = float(metrics["balanced_accuracy"])
        spec = float(metrics["adl_specificity"])
        is_better = score > best_score + tol or (
            abs(score - best_score) <= tol
            and (spec > best_spec + tol or (abs(spec - best_spec) <= tol and threshold > best_threshold))
        )
        if is_better:
            best_score, best_spec, best_threshold = score, spec, threshold
    return rows, best_threshold


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ev = load_eval_module()
    ev.mod = ev.load_module()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = []
    for seed in SEEDS:
        model_path = MODELS_DIR / f"automatic_fall_event_detector_c1_s34_preserve_s{seed}.pth"
        model, config = ev.mod.load_model_checkpoint(model_path, device)
        models.append((model_path, model, config))

    val_videos = ev.mod.list_video_files(ev.SUBJECT3_DIR) + ev.mod.list_video_files(ev.ZENODO_VAL_DIR)
    val_videos = sorted(dict.fromkeys(val_videos))
    val_cache = ensemble_predict(ev, models, val_videos, device)
    scan_rows, best_threshold = scan_and_select(ev, val_cache)
    ev.write_csv(
        REPORTS_DIR / f"{TAG}_val_threshold_scan.csv",
        scan_rows,
        ["threshold", "total", "accuracy", "balanced_accuracy", "adl_specificity",
         "fall_recall", "precision", "tp", "tn", "fp", "fn"],
    )
    best_row = next(r for r in scan_rows if abs(float(r["threshold"]) - float(best_threshold)) < 1e-9)
    val_summary = {
        "tag": TAG,
        "seeds": SEEDS,
        "best_threshold": float(best_threshold),
        "source": "Subject3 + Zenodo val only (5-seed probability ensemble)",
        **best_row,
    }
    (REPORTS_DIR / f"{TAG}_val_summary.json").write_text(
        json.dumps(val_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    s4_videos = ev.mod.list_video_files(ev.SUBJECT4_DIR)
    s4_cache = ensemble_predict(ev, models, s4_videos, device)
    rows, false_alarms, missed_falls = [], [], []
    for item in s4_cache:
        pred = int(ev.classify_video(ev.mod, item, best_threshold))
        rows.append({"video_path": item["video_path"], "true_label": item["true_label"], "pred_label": pred})
        if item["true_label"] == 0 and pred == 1:
            false_alarms.append(item["video_path"])
        elif item["true_label"] == 1 and pred == 0:
            missed_falls.append(item["video_path"])
    metrics = ev.compute_metrics(rows)
    s4_summary = {
        "threshold": float(best_threshold),
        **metrics,
        "false_alarms": "; ".join(false_alarms),
        "missed_falls": "; ".join(missed_falls),
        "false_alarm_count": len(false_alarms),
        "missed_fall_count": len(missed_falls),
        "source": "Subject4 final independent test (5-seed ensemble, threshold frozen from val)",
    }
    ev.write_csv(
        REPORTS_DIR / f"{TAG}_subject4_video_metrics.csv",
        rows,
        ["video_path", "true_label", "pred_label"],
    )
    (REPORTS_DIR / f"{TAG}_subject4_metrics.json").write_text(
        json.dumps(s4_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"val": val_summary, "subject4": s4_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
