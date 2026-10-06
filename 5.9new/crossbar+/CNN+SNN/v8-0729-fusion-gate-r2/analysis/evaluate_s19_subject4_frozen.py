"""Subject4 one-shot test for the S19 lineage with a FROZEN calibrated decision rule.

Frozen before this run (from LOSO val folds only, Subject4 never used):
  - affine calibration per model: score = (peak - med_adl) / (med_fall - med_adl)
      params are taken from each model's own held-out fold video peaks.
  - primary rule: calibrated ensemble mean(S18, S19a) >= 0.80
  - singles: S19a >= 0.80 (raw ~0.487), S19b >= 0.20 (raw ~0.319)

Protocol disclosure: Subject4 was previously observed during S10 selection and
failure diagnosis. S18/S19a/S19b never trained on Subject4, but family-level
choices were informed by earlier diagnostics. Report these numbers as
post-leakage final measurements, not as a pristine one-shot test.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

EXP_ROOT = Path(r"E:\rray\12.18-2\5.9new\crossbar+\CNN+SNN\v8-0729-fusion-gate-r2")
RUNNER_PATH = EXP_ROOT / "code" / "run_refined_gate_r2.py"
OUT_DIR = EXP_ROOT / "reports" / "s19_subject4_frozen"

ROOT = Path(r"E:\rray\12.18-2")
DATA_DIR = ROOT / "data"
SUBJECT4_DIR = ROOT / "gmdcsa_subject4_test" / "Subject 4"
STRIDE = 4

# Frozen calibration constants from held-out-fold val video peaks (no Subject4).
MODELS = {
    "s18": {
        "ckpt": EXP_ROOT / "models" / "c1_ema0999_s18_evtlbl" / "automatic_fall_event_detector_c1_ema0999_s18_evtlbl_ema.pth",
        "med_adl": 0.770, "med_fall": 0.962, "cal_thr": 0.80,
    },
    "s19a": {
        "ckpt": EXP_ROOT / "models" / "c1_ema0999_s19a_evtlbl_s2val" / "automatic_fall_event_detector_c1_ema0999_s19a_evtlbl_s2val_ema.pth",
        "med_adl": 0.347, "med_fall": 0.522, "cal_thr": 0.80,
    },
    "s19b": {
        "ckpt": EXP_ROOT / "models" / "c1_ema0999_s19b_noslow_s2val" / "automatic_fall_event_detector_c1_ema0999_s19b_noslow_s2val_ema.pth",
        "med_adl": 0.183, "med_fall": 0.863, "cal_thr": 0.20,
    },
}
ENSEMBLE = {"members": ["s18", "s19a"], "cal_thr": 0.80}


def log(msg: str) -> None:
    print(msg, flush=True)


def load_model(r2, base, ckpt_path: Path, device: torch.device):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = dict(checkpoint.get("config", {}))
    r2.REFINED_DROP_FES_CONTEXT = bool(cfg.get("refined_drop_fes_context", True))
    r2.REFINED_LATE_FUSION_LAMBDA = float(cfg.get("snn_late_fusion_lambda", 0.0))
    model = r2.RefinedFallEventDetectorR2(
        hidden_dim=int(cfg.get("hidden_dim", 64)),
        num_classes=len(base.LABEL_NAME_TO_ID),
        use_snn_temporal_branch=True,
        snn_hidden_dim=int(cfg.get("snn_hidden_dim", 32)),
        snn_decay=float(cfg.get("snn_decay", 0.82)),
        snn_threshold=float(cfg.get("snn_threshold", 0.30)),
        snn_surrogate_scale=float(cfg.get("snn_surrogate_scale", 8.0)),
        learnable_snn_dynamics=bool(cfg.get("learnable_snn_dynamics", True)),
        use_snn_fusion_gate=True,
        snn_fusion_gate_mode="group",
        snn_event_score_threshold=float(cfg.get("snn_event_score_threshold", 0.55)),
        snn_event_score_sharpness=float(cfg.get("snn_event_score_sharpness", 10.0)),
        snn_input_mode=str(cfg.get("snn_input_mode", "scalar7")),
        snn_pool_mode=str(cfg.get("snn_pool_mode", "global3")),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def metrics(rows, score_key, thr):
    tp = sum(1 for r in rows if r["true_label"] == 1 and r[score_key] >= thr)
    fn = sum(1 for r in rows if r["true_label"] == 1 and r[score_key] < thr)
    tn = sum(1 for r in rows if r["true_label"] == 0 and r[score_key] < thr)
    fp = sum(1 for r in rows if r["true_label"] == 0 and r[score_key] >= thr)
    spec = tn / max(tn + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"acc": (tp + tn) / len(rows), "bal": (spec + rec) / 2, "spec": spec,
            "rec": rec, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    sys.argv = ["run_refined_gate_r2.py"]
    spec = importlib.util.spec_from_file_location("s19_subject4_runner", RUNNER_PATH)
    r2 = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = r2
    spec.loader.exec_module(r2)
    base = r2.base
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hrs_values, lrs_values = base.load_conductance_pair(DATA_DIR)
    device_dynamics = base.load_device_dynamics(DATA_DIR)
    windows, labels, summary = base.load_external_labeled_windows(
        [SUBJECT4_DIR], image_size=64, window_size=16, stride=STRIDE,
        max_videos=0, fall_name_pattern="fall", use_pixel_human=False,
        use_silhouette_human=False, pixel_grid_size=20,
    )
    dataset = base.CombinedFallDataset(
        video_windows=np.asarray(windows, dtype=np.float32), video_labels=labels,
        auxiliary_records=[], window_size=16, image_size=64, use_crossbar=True,
        use_pixel_human=False, use_silhouette_human=False, pixel_grid_size=20,
        hrs_values=hrs_values, lrs_values=lrs_values, device_dynamics=device_dynamics,
        crossbar_readout_noise_scale=0.25, augment=False, base_seed=10041,
    )
    fall_id = base.LABEL_NAME_TO_ID["fall"]

    window_probs: dict[str, np.ndarray] = {}
    for name, info in MODELS.items():
        model = load_model(r2, base, info["ckpt"], device)
        parts = []
        with torch.no_grad():
            for offset in range(0, len(dataset), 32):
                batch = torch.stack([dataset[i][0] for i in range(offset, min(offset + 32, len(dataset)))])
                logits, _ = model.forward_with_gate(batch.to(device))
                parts.append(torch.softmax(logits, dim=1)[:, fall_id].cpu())
        window_probs[name] = torch.cat(parts).numpy()
        del model
        torch.cuda.empty_cache()
        log(f"scored {name}: {len(parts[0]) if parts else 0}+ windows")

    rows = []
    offset = 0
    for item in summary["videos"]:
        count = int(item["num_windows"])
        video_path = Path(item["video_path"])
        row = {"video": video_path.name,
               "true_label": int(base.infer_video_level_label_id(video_path, "fall") == fall_id)}
        for name, info in MODELS.items():
            peak = float(window_probs[name][offset:offset + count].max())
            cal = (peak - info["med_adl"]) / max(info["med_fall"] - info["med_adl"], 1e-6)
            row[f"{name}_peak"] = round(peak, 4)
            row[f"{name}_cal"] = round(cal, 4)
        rows.append(row)
        offset += count
    assert offset == len(next(iter(window_probs.values())))

    members = ENSEMBLE["members"]
    for row in rows:
        row["ensemble_cal"] = round(float(np.mean([row[f"{m}_cal"] for m in members])), 4)

    results = {"protocol": {
        "disclosure": "Subject4 previously observed during S10 selection/diagnosis; "
                      "models never trained on Subject4; decision rule frozen from LOSO val folds.",
        "calibration": "affine per model from own held-out-fold val video peaks",
        "models": {k: {kk: vv for kk, vv in v.items() if kk != "ckpt"} for k, v in MODELS.items()},
        "ensemble": ENSEMBLE,
    }, "results": {}}
    for name, info in MODELS.items():
        results["results"][name] = metrics(rows, f"{name}_cal", info["cal_thr"])
    results["results"]["ensemble_s18_s19a"] = metrics(rows, "ensemble_cal", ENSEMBLE["cal_thr"])

    with (OUT_DIR / "subject4_video_scores.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT_DIR / "subject4_frozen_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    log(json.dumps(results["results"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
