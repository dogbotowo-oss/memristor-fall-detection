"""LOSO threshold re-calibration with in-distribution val (v1000 rule).

Rule (fixed, leak-free): for each fold, the threshold is selected on
Zenodo val + one GMDCSA subject that is INSIDE that fold's training set:
  folds 1, 2, 4 -> Subject3 (+ Zenodo val)
  fold 3        -> Subject4 (+ Zenodo val)
The held-out test subject is never touched. Mirrors the v999 protocol
structure (one GMDCSA subject + Zenodo val). No retraining; reuses the
already-trained LOSO models, re-scans thresholds, and re-evaluates once.

Outputs per tag: *_gval_summary.json, *_gtest_metrics.json,
*_gtest_video_metrics.csv under s30_pose/reports.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(r"E:\rray\12.18-2")
S30 = ROOT / "5.9new" / "crossbar+" / "CNN+SNN" / "s30_pose"
EVAL_PATH = S30 / "analysis" / "strict_eval_s30.py"
MODELS = S30 / "models"
REPORTS = S30 / "reports"
DATA1 = ROOT / "data1" / "data"
ZENODO_VAL = ROOT / "zenodo_falldb_video_split" / "val"
GMDCSA = (
    ROOT / "测试集" / "13354453" / "ekramalam"
    / "GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-v2.1"
    / "ekramalam-GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-5abac76"
)
S3_DIR = GMDCSA / "Subject 3"
S4_DIR = ROOT / "gmdcsa_subject4_test" / "Subject 4"
SUBJECT_DIRS = {1: GMDCSA / "Subject 1", 2: GMDCSA / "Subject 2", 3: S3_DIR, 4: S4_DIR}
CAL_SUBJECT = {1: S3_DIR, 2: S3_DIR, 3: S4_DIR, 4: S3_DIR}


def load_eval_module():
    spec = importlib.util.spec_from_file_location("strict_eval_loso2", EVAL_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    mod.DATA_DIR = DATA1
    return mod


def pick_recall_priority_threshold(scan_rows, recall_target: float = 0.85, cap: float = 0.70):
    """Highest-specificity threshold whose val recall >= target (grid <= cap).

    Standard sensitivity-first operating rule for fall detection; uses only
    the validation scan table (no test contact). Falls back to the
    highest-recall grid point if the target is unreachable.
    """
    grid = [r for r in scan_rows if float(r["threshold"]) <= cap + 1e-9]
    ok = [r for r in grid if float(r["fall_recall"]) >= recall_target]
    pool = ok if ok else sorted(grid, key=lambda r: -float(r["fall_recall"]))[:1]
    best = max(pool, key=lambda r: (float(r["adl_specificity"]), float(r["threshold"])))
    return float(best["threshold"]), best


def main() -> None:
    import torch
    sev = load_eval_module()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for test_subject in (1, 2, 3, 4):
        for ideal in (False, True):
            tag = f"c1_s34_data1_loso_ts{test_subject}{'_ideal' if ideal else ''}_s11"
            if (REPORTS / f"{tag}_gtest_metrics.json").exists() and (
                REPORTS / f"{tag}_gtestR_metrics.json"
            ).exists():
                print(f"skip {tag}", flush=True)
                continue
            model_path = MODELS / f"automatic_fall_event_detector_{tag}.pth"
            if not model_path.exists():
                print(f"model missing, skip {tag}", flush=True)
                continue
            mod = sev.load_module()
            model, config = mod.load_model_checkpoint(model_path, device)
            val_videos = sorted(mod.list_video_files(CAL_SUBJECT[test_subject])) + sorted(
                mod.list_video_files(ZENODO_VAL))
            val_videos = sorted(dict.fromkeys(val_videos))
            scan_rows, best_thr = sev.scan_thresholds(mod, model, config, val_videos, device)
            best_row = next(r for r in scan_rows if abs(float(r["threshold"]) - float(best_thr)) < 1e-9)
            val_summary = {"tag": tag, "best_threshold": float(best_thr),
                           "source": f"GMDCSA-informed val ({CAL_SUBJECT[test_subject].name} + Zenodo val)",
                           **best_row}
            (REPORTS / f"{tag}_gval_summary.json").write_text(
                json.dumps(val_summary, indent=2, ensure_ascii=False), encoding="utf-8")
            test_videos = sorted(mod.list_video_files(SUBJECT_DIRS[test_subject]))
            rows, summary = sev.evaluate_at_threshold(mod, model, config, test_videos, best_thr, device)
            with (REPORTS / f"{tag}_gtest_video_metrics.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["video_path", "true_label", "pred_label"])
                w.writeheader()
                w.writerows(rows)
            summary["source"] = f"LOSO fold test=Subject{test_subject} (GMDCSA-informed frozen threshold)"
            summary["tag"] = tag
            (REPORTS / f"{tag}_gtest_metrics.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"{tag} [bal]: thr={best_thr:.2f} acc={summary['accuracy']*100:.2f} "
                  f"spec={summary['adl_specificity']*100:.1f} rec={summary['fall_recall']*100:.2f}", flush=True)

            # recall-prioritized operating rule (val-only, fixed for all folds)
            rec_thr, rec_row = pick_recall_priority_threshold(scan_rows)
            val_summary_r = {"tag": tag, "best_threshold": rec_thr,
                             "source": f"recall-priority rule (val recall>=0.85) on {CAL_SUBJECT[test_subject].name} + Zenodo val",
                             **rec_row}
            (REPORTS / f"{tag}_gvalR_summary.json").write_text(
                json.dumps(val_summary_r, indent=2, ensure_ascii=False), encoding="utf-8")
            rows_r, summary_r = sev.evaluate_at_threshold(mod, model, config, test_videos, rec_thr, device)
            with (REPORTS / f"{tag}_gtestR_video_metrics.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["video_path", "true_label", "pred_label"])
                w.writeheader()
                w.writerows(rows_r)
            summary_r["source"] = f"LOSO fold test=Subject{test_subject} (recall-priority frozen threshold)"
            summary_r["tag"] = tag
            (REPORTS / f"{tag}_gtestR_metrics.json").write_text(
                json.dumps(summary_r, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"{tag} [rec]: thr={rec_thr:.2f} acc={summary_r['accuracy']*100:.2f} "
                  f"spec={summary_r['adl_specificity']*100:.1f} rec={summary_r['fall_recall']*100:.2f}", flush=True)
    print("RESCAN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
