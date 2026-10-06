"""Video-level evaluation on the validation set.

Window-level metrics cap around 0.75 bal for the whole S-series; the final
Subject4 protocol is video-level (33 videos, TP/TN/FP/FN=13/16/4/4), so this
aggregates per-video mean fall probability and reports video-level metrics.
Thresholds calibrated on val (in-sample, same as p006 protocol).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

EXP_ROOT = Path(r"E:\rray\12.18-2\5.9new\crossbar+\CNN+SNN\v8-0729-fusion-gate-r2")
RUNNER_PATH = EXP_ROOT / "code" / "run_refined_gate_r2.py"
CKPTS = {
    "s3_scalar7": EXP_ROOT / "models" / "c1_ema0999_s3" / "automatic_fall_event_detector_c1_ema0999_s3_ema.pth",
    "s5_grid": EXP_ROOT / "models" / "c1_ema0999_s5" / "automatic_fall_event_detector_c1_ema0999_s5_ema.pth",
    "s6_flow": EXP_ROOT / "models" / "c1_ema0999_s6flow" / "automatic_fall_event_detector_c1_ema0999_s6flow_ema.pth",
    "s7_segflow": EXP_ROOT / "models" / "c1_ema0999_s7seg" / "automatic_fall_event_detector_c1_ema0999_s7seg_ema.pth",
}

ROOT = Path(r"E:\rray\12.18-2")
DATA_DIR = ROOT / "data"
GMDCSA_ROOT = (
    ROOT
    / "测试集"
    / "13354453"
    / "ekramalam"
    / "GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-v2.1"
    / "ekramalam-GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-5abac76"
)


def log(msg: str) -> None:
    print(msg, flush=True)


sys.argv = ["run_refined_gate_r2.py"]
spec = importlib.util.spec_from_file_location("r2", RUNNER_PATH)
r2 = importlib.util.module_from_spec(spec)
sys.modules["r2"] = r2
spec.loader.exec_module(r2)
base = r2.base

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hrs_values, lrs_values = base.load_conductance_pair(DATA_DIR)
device_dynamics = base.load_device_dynamics(DATA_DIR)
windows, labels, summary = base.load_external_labeled_windows(
    [GMDCSA_ROOT / "Subject 3", ROOT / "zenodo_falldb_video_split" / "val"],
    image_size=64,
    window_size=16,
    stride=16,
    max_videos=0,
    fall_name_pattern="fall",
    use_pixel_human=False,
    use_silhouette_human=False,
    pixel_grid_size=20,
)
assert len(windows) == 1076, f"protocol violation: {len(windows)}"

videos = summary["videos"]
bounds = []
offset = 0
for row in videos:
    n = int(row["num_windows"])
    label_text = str(row.get("assigned_label"))
    if label_text in base.LABEL_NAME_TO_ID:
        label_id = int(base.LABEL_NAME_TO_ID[label_text])
    else:
        # Zenodo rows carry 'metadata' as assigned_label; infer from window labels.
        label_id = int(np.bincount(labels[offset : offset + n].astype(np.int64)).argmax())
    bounds.append((row["video_name"], label_id, offset, offset + n))
    offset += n
assert offset == len(windows), f"video boundary mismatch: {offset} vs {len(windows)}"
log(f"videos={len(bounds)} windows={len(windows)}")

val_dataset = base.CombinedFallDataset(
    video_windows=np.asarray(windows, dtype=np.float32),
    video_labels=labels,
    auxiliary_records=[],
    window_size=16,
    image_size=64,
    use_crossbar=True,
    use_pixel_human=False,
    use_silhouette_human=False,
    pixel_grid_size=20,
    hrs_values=hrs_values,
    lrs_values=lrs_values,
    device_dynamics=device_dynamics,
    crossbar_readout_noise_scale=0.25,
    augment=False,
    base_seed=42 + 9999,
)

fall_id = base.LABEL_NAME_TO_ID["fall"]
normal_id = base.LABEL_NAME_TO_ID["normal"]
video_true = np.asarray([b[1] for b in bounds], dtype=np.int64)
video_true_binary = (video_true == fall_id).astype(np.int64)
log(f"video class counts: fall={int(video_true_binary.sum())} adl={int((1 - video_true_binary).sum())}")


def load_model(ckpt_path: Path):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = dict(checkpoint.get("config", {}))
    r2.REFINED_DROP_FES_CONTEXT = bool(cfg.get("refined_drop_fes_context", False))
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


def window_fall_probs(model) -> np.ndarray:
    out = []
    BATCH = 32
    with torch.no_grad():
        for offset in range(0, len(val_dataset), BATCH):
            batch_x = torch.stack([val_dataset[i][0] for i in range(offset, min(offset + BATCH, len(val_dataset)))])
            logits = model(batch_x.to(device))
            out.append(torch.softmax(logits, dim=1)[:, fall_id].cpu())
    return torch.cat(out).numpy()


def video_metrics(video_score: np.ndarray, thr: float) -> tuple[float, float, float, float]:
    pred = (video_score >= thr).astype(np.int64)
    y = video_true_binary
    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    spec = tn / max(tn + fp, 1)
    rec = tp / max(tp + fn, 1)
    acc = (tp + tn) / len(y)
    return 0.5 * (spec + rec), spec, rec, acc


def best_thr(video_score: np.ndarray) -> tuple[float, float, float, float, float]:
    best = (-1.0, 0.5, 0.0, 0.0, 0.0)
    for thr in np.arange(0.20, 0.901, 0.025):
        bal, spec, rec, acc = video_metrics(video_score, float(thr))
        if bal > best[0]:
            best = (bal, float(thr), spec, rec, acc)
    return best


probs: dict[str, np.ndarray] = {}
for name, path in CKPTS.items():
    model = load_model(path)
    probs[name] = window_fall_probs(model)
    del model
    torch.cuda.empty_cache()

# v88 = s3 + late fusion lambda 0.2 on probs is not exact; use s3 probabilities
# and note v88 argmax-equivalent is covered by the s3 curve within noise.

combos = {
    "s3_scalar7": ["s3_scalar7"],
    "s5_grid": ["s5_grid"],
    "s6_flow": ["s6_flow"],
    "s7_segflow": ["s7_segflow"],
    "ens_s356": ["s3_scalar7", "s5_grid", "s6_flow"],
    "ens_all4": ["s3_scalar7", "s5_grid", "s6_flow", "s7_segflow"],
}
for name, members in combos.items():
    win_prob = np.mean([probs[m] for m in members], axis=0)
    video_score = np.asarray([win_prob[s:e].mean() for _, _, s, e in bounds])
    bal5, spec5, rec5, acc5 = video_metrics(video_score, 0.5)
    bbal, bthr, bspec, brec, bacc = best_thr(video_score)
    log(f"{name:12s} thr=0.50: bal={bal5:.4f} spec={spec5:.4f} rec={rec5:.4f} acc={acc5:.4f} | "
        f"best-thr={bthr:.3f}: bal={bbal:.4f} spec={bspec:.4f} rec={brec:.4f} acc={bacc:.4f}")
