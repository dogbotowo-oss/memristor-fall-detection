"""One-time independent Subject4 test for the frozen S10 candidate.

Frozen rule selected on Subject3 + Zenodo val only:
  model = c1_ema0999_s10up best EMA checkpoint
  inference = stride-4 windows, video fall score = max window fall probability
  decision = fall when video score >= 0.90

No Subject4 score is used to alter the model, threshold, or decision rule.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

V90_ROOT = Path(r"E:\rray\12.18-2\5.9new\crossbar+\CNN+SNN\v90")
RUNNER_PATH = V90_ROOT / "code" / "run_refined_gate_r2.py"
CKPT = V90_ROOT / "models" / "automatic_fall_event_detector_v90.pth"
OUT_DIR = V90_ROOT / "reports" / "subject4_frozen"

ROOT = Path(r"E:\rray\12.18-2")
DATA_DIR = ROOT / "data"
SUBJECT4_DIR = ROOT / "gmdcsa_subject4_test" / "Subject 4"
THRESHOLD = 0.90
STRIDE = 4


def log(message: str) -> None:
    print(message, flush=True)


def load_model(r2, base, device: torch.device):
    checkpoint = torch.load(CKPT, map_location=device, weights_only=False)
    cfg = dict(checkpoint.get("config", {}))
    r2.REFINED_MODE = str(cfg.get("snn_fusion_gate_mode", "group"))
    r2.REFINED_FLOOR = float(cfg.get("snn_fusion_gate_floor", 0.2))
    r2.REFINED_GROUP_BIAS = float(cfg.get("refined_group_bias", 0.0))
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
    return model, cfg


def main() -> None:
    if OUT_DIR.exists():
        raise RuntimeError(f"refusing to overwrite existing one-time test output: {OUT_DIR}")
    if not CKPT.exists():
        raise FileNotFoundError(CKPT)
    if not SUBJECT4_DIR.exists():
        raise FileNotFoundError(SUBJECT4_DIR)

    sys.argv = ["run_refined_gate_r2.py"]
    spec = importlib.util.spec_from_file_location("s10_subject4_runner", RUNNER_PATH)
    r2 = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = r2
    spec.loader.exec_module(r2)
    base = r2.base
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, cfg = load_model(r2, base, device)
    hrs_values, lrs_values = base.load_conductance_pair(DATA_DIR)
    device_dynamics = base.load_device_dynamics(DATA_DIR)
    windows, labels, summary = base.load_external_labeled_windows(
        [SUBJECT4_DIR],
        image_size=int(cfg.get("image_size", 64)),
        window_size=int(cfg.get("window_size", 16)),
        stride=STRIDE,
        max_videos=0,
        fall_name_pattern="fall",
        use_pixel_human=False,
        use_silhouette_human=False,
        pixel_grid_size=20,
    )
    if not len(windows):
        raise RuntimeError("Subject4 produced zero windows")
    dataset = base.CombinedFallDataset(
        video_windows=np.asarray(windows, dtype=np.float32),
        video_labels=labels,
        auxiliary_records=[],
        window_size=int(cfg.get("window_size", 16)),
        image_size=int(cfg.get("image_size", 64)),
        use_crossbar=True,
        use_pixel_human=False,
        use_silhouette_human=False,
        pixel_grid_size=20,
        hrs_values=hrs_values,
        lrs_values=lrs_values,
        device_dynamics=device_dynamics,
        crossbar_readout_noise_scale=float(cfg.get("crossbar_readout_noise_scale", 0.25)),
        augment=False,
        base_seed=42 + 9999,
    )
    fall_id = base.LABEL_NAME_TO_ID["fall"]
    parts = []
    with torch.no_grad():
        for offset in range(0, len(dataset), 32):
            batch = torch.stack([dataset[i][0] for i in range(offset, min(offset + 32, len(dataset)))])
            logits, _ = model.forward_with_gate(batch.to(device))
            parts.append(torch.softmax(logits, dim=1)[:, fall_id].cpu())
    probabilities = torch.cat(parts).numpy()

    rows = []
    offset = 0
    for item in summary["videos"]:
        count = int(item["num_windows"])
        video_path = Path(item["video_path"])
        score = float(probabilities[offset:offset + count].max())
        offset += count
        true_fall = int(base.infer_video_level_label_id(video_path, "fall") == fall_id)
        pred_fall = int(score >= THRESHOLD)
        rows.append({
            "video_path": str(video_path),
            "true_label": true_fall,
            "video_peak_fall_probability": score,
            "pred_label": pred_fall,
            "num_stride4_windows": count,
        })
    assert offset == len(probabilities)

    tp = sum(r["true_label"] == 1 and r["pred_label"] == 1 for r in rows)
    tn = sum(r["true_label"] == 0 and r["pred_label"] == 0 for r in rows)
    fp = sum(r["true_label"] == 0 and r["pred_label"] == 1 for r in rows)
    fn = sum(r["true_label"] == 1 and r["pred_label"] == 0 for r in rows)
    spec_v = tn / max(tn + fp, 1)
    rec_v = tp / max(tp + fn, 1)
    result = {
        "model": str(CKPT),
        "source_epoch": cfg.get("weight_ema_source_epoch"),
        "source": "S10 frozen candidate, no Subject4 tuning",
        "decision_rule": "stride4 video peak fall probability >= 0.90",
        "threshold": THRESHOLD,
        "total": len(rows),
        "accuracy": (tp + tn) / len(rows),
        "balanced_accuracy": (spec_v + rec_v) / 2,
        "adl_specificity": spec_v,
        "fall_recall": rec_v,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
    OUT_DIR.mkdir(parents=True)
    with (OUT_DIR / "subject4_video_metrics.csv").open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT_DIR / "subject4_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
