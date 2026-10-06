"""Quantify why the SNN branch leans toward ADL.

Three measurements on the 1076-window validation set (Subject3 + Zenodo val,
protocol-identical to training; read-only):

A. Motion budget: how much of a fall window is actually moving. If most fall
   windows are dominated by static (post-fall hold) segments, the LIF branch
   receives almost no input on falls while hard ADL keeps stimulating it.
B. fall_event_score separability: distribution + AUC of the hand-crafted
   event score between ADL and Fall.
C. SNN response correlation (revive checkpoint, where LIF actually fires):
   does snn activity track motion (class-agnostic) or the fall label?

Outputs: snn_bias_diagnosis.csv (per-window rows) + printed summary.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

EXP_ROOT = Path(r"E:\rray\12.18-2\5.9new\crossbar+\CNN+SNN\v8-0729-fusion-gate-r2")
RUNNER_PATH = EXP_ROOT / "code" / "run_refined_gate_r2.py"
REVIVE_CKPT = (
    EXP_ROOT / "models" / "snn_revive_thr030_clsaux030"
    / "automatic_fall_event_detector_snn_revive_thr030_clsaux030.pth"
)
OUT_CSV = EXP_ROOT / "analysis" / "snn_bias_diagnosis.csv"

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
SUBJECT3_DIR = GMDCSA_ROOT / "Subject 3"
ZENODO_VAL_DIR = ROOT / "zenodo_falldb_video_split" / "val"


def log(msg: str) -> None:
    print(msg, flush=True)


def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC of `scores` for the positive class (labels == 1)."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start
        while end + 1 < len(scores) and sorted_scores[end + 1] == sorted_scores[start]:
            end += 1
        if end > start:
            ranks[order[start : end + 1]] = ranks[order[start : end + 1]].mean()
        start = end + 1
    pos = labels == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


sys.argv = ["run_refined_gate_r2.py"]
spec = importlib.util.spec_from_file_location("r2", RUNNER_PATH)
r2 = importlib.util.module_from_spec(spec)
sys.modules["r2"] = r2
spec.loader.exec_module(r2)
base = r2.base

r2.REFINED_MODE = "group"
r2.REFINED_FLOOR = 0.2
r2.REFINED_GROUP_BIAS = 0.0
r2.REFINED_CHANNEL_BIAS = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(REVIVE_CKPT, map_location=device, weights_only=False)
cfg = dict(checkpoint.get("config", {}))
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
).to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()
log(f"loaded revive checkpoint, snn_threshold={cfg.get('snn_threshold')}")

hrs_values, lrs_values = base.load_conductance_pair(DATA_DIR)
device_dynamics = base.load_device_dynamics(DATA_DIR)

windows, labels, summary = base.load_external_labeled_windows(
    [SUBJECT3_DIR, ZENODO_VAL_DIR],
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
log(f"external_val_labeled_windows={len(windows)}")

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

true_binary = (labels == base.LABEL_NAME_TO_ID["fall"]).astype(np.int64)

rows: list[dict[str, float]] = []
BATCH = 32
with torch.no_grad():
    for offset in range(0, len(val_dataset), BATCH):
        batch_x = torch.stack([val_dataset[i][0] for i in range(offset, min(offset + BATCH, len(val_dataset)))])
        batch_x = batch_x.to(device)
        features = model.extract_temporal_event_features(batch_x)
        motion = features[:, :, 1].clamp_min(0.0)
        fes = model.extract_fall_event_score(features).squeeze(1)
        snn_pooled = model.run_lif_temporal_branch(batch_x, features)
        snn_hidden = snn_pooled.size(1) // 3
        spike_mean = snn_pooled[:, :snn_hidden].mean(dim=1)

        peak_motion = motion.amax(dim=1)
        peak_index = motion.argmax(dim=1)
        time_index = torch.arange(motion.size(1), device=motion.device).view(1, -1)
        post_mask = time_index > peak_index.view(-1, 1)
        static_thresh = 0.1 * peak_motion.clamp_min(1e-6).view(-1, 1)
        post_count = post_mask.sum(dim=1).clamp_min(1).float()
        post_static_frac = ((post_mask & (motion < static_thresh)).sum(dim=1).float() / post_count)
        no_motion = (peak_motion < 1e-3).float()

        for j in range(batch_x.size(0)):
            rows.append(
                {
                    "index": offset + j,
                    "label": int(true_binary[offset + j]),
                    "peak_motion": float(peak_motion[j].cpu()),
                    "mean_motion": float(motion[j].mean().cpu()),
                    "post_peak_static_frac": float(post_static_frac[j].cpu()),
                    "no_motion_window": float(no_motion[j].cpu()),
                    "fall_event_score": float(fes[j].cpu()),
                    "snn_spike_mean": float(spike_mean[j].cpu()),
                }
            )
        if offset % (BATCH * 10) == 0:
            log(f"processed {offset}/{len(val_dataset)}")

with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
log(f"wrote {OUT_CSV.name} rows={len(rows)}")

arr = {k: np.asarray([r[k] for r in rows], dtype=np.float64) for k in rows[0]}
y = arr["label"].astype(np.int64)


def report(name: str, values: np.ndarray) -> None:
    adl = values[y == 0]
    fall = values[y == 1]
    log(
        f"{name:24s} | ADL mean={adl.mean():.4f} p50={np.median(adl):.4f} "
        f"| Fall mean={fall.mean():.4f} p50={np.median(fall):.4f} "
        f"| AUC(fall+)={auc_score(values, y):.3f}"
    )


log("---- summary (ADL=0, Fall=1) ----")
report("peak_motion", arr["peak_motion"])
report("mean_motion", arr["mean_motion"])
report("post_peak_static_frac", arr["post_peak_static_frac"])
report("fall_event_score", arr["fall_event_score"])
report("snn_spike_mean", arr["snn_spike_mean"])

log(
    f"no_motion_window rate | ADL={arr['no_motion_window'][y == 0].mean():.3f} "
    f"Fall={arr['no_motion_window'][y == 1].mean():.3f}"
)

snn_act = arr["snn_spike_mean"]
for ref_name in ("peak_motion", "mean_motion", "fall_event_score"):
    ref = arr[ref_name]
    if ref.std() > 0 and snn_act.std() > 0:
        corr = float(np.corrcoef(ref, snn_act)[0, 1])
    else:
        corr = float("nan")
    log(f"corr(snn_spike_mean, {ref_name}) = {corr:.3f}")

hard = np.asarray([float(s["hard_negative_score"]) for s in val_dataset.samples], dtype=np.float64)
hard_adl = (y == 0) & (hard >= np.quantile(hard[y == 0], 0.88))
log(f"snn_spike_mean on hard ADL={snn_act[hard_adl].mean():.4f} vs fall={snn_act[y == 1].mean():.4f}")
