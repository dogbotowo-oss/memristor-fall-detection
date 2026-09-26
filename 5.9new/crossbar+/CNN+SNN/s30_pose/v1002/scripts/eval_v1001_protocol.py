"""Evaluate full / bw / bw4 LOSO checkpoints under the v1001 main-table protocol.

Protocol (confirmed by user): validation set = Zenodo val only;
rule = F1-max threshold on validation, tie -> higher threshold, cap 0.70.

full checkpoints: s30_pose/models (trained by queue_loso_v1000.py) — used to
verify reproduction of the user's main table (96.88/91.67/81.40/75.68).
bw  checkpoints: s30_pose/models (*_bw_s11) — binary conductance weights.
bw4 checkpoints: v1002/models (*_bw4_s11) — 4-level measured conductance weights.

bw/bw4 checkpoints must be evaluated with the v1002 code (weight quantization
lives there); full models are unaffected since their config has no quantization
keys.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(r"F:\12.18-2")
S30 = ROOT / "5.9new" / "crossbar+" / "CNN+SNN" / "s30_pose"
V1002 = S30 / "v1002"
EVAL_PATH = S30 / "analysis" / "strict_eval_s30.py"
CODE_PATH = V1002 / "code" / "automatic_fall_detector.py"
REPORTS = V1002 / "reports"
DATA1 = ROOT / "data1"
POSE_CACHE = ROOT / "pose_cache"
ZENODO_VAL = ROOT / "zenodo_falldb_video_split" / "val"
GMDCSA = (
    ROOT / "测试集" / "13354453" / "ekramalam"
    / "GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-v2.1"
    / "ekramalam-GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-5abac76"
)
SUBJECT_DIRS = {i: GMDCSA / f"Subject {i}" for i in (1, 2, 3)}
SUBJECT_DIRS[4] = ROOT / "gmdcsa_subject4_test" / "Subject 4"

CAP = 0.70
GROUPS = {
    "bw":   (S30 / "models", "automatic_fall_event_detector_c1_s34_data1_loso_ts{}_bw_s11.pth"),
    "bw4":  (V1002 / "models", "automatic_fall_event_detector_c1_s34_data1_loso_ts{}_bw4_s11.pth"),
}
CAL_SUBJECT = {1: 3, 2: 3, 3: 4, 4: 3}
EXPECTED_FULL = {1: (0.50, 96.88), 2: (0.45, 91.67), 3: (0.60, 81.40), 4: (0.70, 75.68)}


def load_eval_module():
    spec = importlib.util.spec_from_file_location("strict_eval_v1001p", EVAL_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    mod.ROOT = ROOT
    mod.S30_DIR = S30
    mod.CODE_PATH = CODE_PATH
    mod.DATA_DIR = DATA1
    mod.POSE_CACHE_DIR = POSE_CACHE
    return mod


def f1_max_threshold(scan_rows, cap=CAP):
    best = None
    for r in scan_rows:
        thr = float(r["threshold"])
        if thr > cap + 1e-9:
            continue
        rec = float(r["fall_recall"])
        prec = float(r.get("fall_precision", r.get("precision", 0.0)))
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        key = (f1, thr)
        if best is None or key > best[0]:
            best = (key, thr)
    return best[1]


def main() -> None:
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sev = load_eval_module()
    mod = sev.load_module()
    zenodo = sorted(mod.list_video_files(ZENODO_VAL))

    all_out = {}
    for group, (mdir, fmt) in GROUPS.items():
        all_out[group] = []
        for ts in (1, 2, 3, 4):
            model_path = mdir / fmt.format(ts)
            model, config = mod.load_model_checkpoint(model_path, device)
            # E-era checkpoints store pose_cache_dir=E:\rray\... (gone); override
            config["pose_cache_dir"] = str(POSE_CACHE)
            val_videos = list(zenodo) + sorted(mod.list_video_files(SUBJECT_DIRS[CAL_SUBJECT[ts]]))
            scan_rows, _ = sev.scan_thresholds(mod, model, config, val_videos, device)
            thr = f1_max_threshold(scan_rows)
            test_videos = sorted(mod.list_video_files(SUBJECT_DIRS[ts]))
            rows, summary = sev.evaluate_at_threshold(mod, model, config, test_videos, thr, device)
            summary["tag"] = f"{group}_ts{ts}_v1001proto"
            summary["val_threshold"] = thr
            summary["protocol"] = "F1-max on Zenodo val (tie->higher, cap 0.70)"
            (REPORTS / f"{group}_ts{ts}_v1001proto_test_metrics.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            with (REPORTS / f"{group}_ts{ts}_v1001proto_test_video_metrics.csv").open(
                    "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["video_path", "true_label", "pred_label"])
                w.writeheader()
                w.writerows(rows)
            extra = ""
            if group == "full":
                exp_thr, exp_acc = EXPECTED_FULL[ts]
                extra = f"  expect thr={exp_thr} acc={exp_acc} -> {'MATCH' if abs(summary['accuracy']*100-exp_acc)<0.01 else 'DIFF'}"
            print(f"{group} ts{ts}: thr={thr:.2f} acc={summary['accuracy']*100:.2f} "
                  f"BA={summary['balanced_accuracy']*100:.2f} spec={summary['adl_specificity']*100:.1f} "
                  f"rec={summary['fall_recall']*100:.2f}{extra}", flush=True)
            all_out[group].append(summary)

    import statistics as st
    print("=== macro averages (v1001 protocol) ===", flush=True)
    for group, rows in all_out.items():
        for m in ("accuracy", "balanced_accuracy", "adl_specificity", "fall_recall"):
            v = [r[m] * 100 for r in rows]
            print(f"{group} {m}: {st.mean(v):.2f} +/- {st.stdev(v):.2f}", flush=True)


if __name__ == "__main__":
    main()
